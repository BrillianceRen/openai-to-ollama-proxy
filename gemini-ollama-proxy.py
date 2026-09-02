#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemini-ollama-proxy
===================

基于 Google 官方 google-genai SDK 把 Google Gemini API (Developer API)
转换为 Ollama API, 同时提供 OpenAI 兼容接口 /v1/chat/completions 与 /v1/models。
支持流式 NDJSON、多模态识图、工具调用 (Function Calling) 与思考流 (Thinking)。

端点
----
GET  /                      返回 "Ollama is running"
GET  /api/version           返回模拟的 Ollama 版本号
GET  /api/tags              汇总 Gemini 模型列表, 优先匹配 models/gemini.json, 未命中自动生成
POST /api/show              优先读取 models/gemini.json, 未命中自动生成模型详情
POST /api/chat              转换为 Gemini API 转发 (支持流式 NDJSON)
POST /api/generate          转换为 Gemini API 转发 (支持流式 NDJSON 与多模态图片)
GET  /api/ps                返回空模型列表 (兼容 Ollama 状态轮询)
GET  /v1/models             OpenAI 兼容模型列表
POST /v1/chat/completions   OpenAI 兼容请求透传 (支持流式 SSE)

用法
----
    python gemini-ollama-proxy.py
    python gemini-ollama-proxy.py --config config.json
    python gemini-ollama-proxy.py --api-key <API_KEY>
    python gemini-ollama-proxy.py --port 11434 --verbose
"""

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_MODEL = "gemini-3.6-flash"
OLLAMA_VERSION = "0.5.4"
__version__ = "1.3.0"


def now_iso():
    """Ollama 风格 RFC3339 时间戳, 如 2026-09-01T12:00:00.123Z"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def ndjson(obj):
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


class ModelNotFoundError(Exception):
    pass


class BodyTooLargeError(Exception):
    pass


class UpstreamError(Exception):
    def __init__(self, status, body):
        super().__init__("upstream http %s: %s" % (status, body))
        self.status = status
        self.body = body


class Config:
    def __init__(self, data=None, base_dir=None):
        data = data or {}
        base_dir = base_dir or SCRIPT_DIR
        self.host = str(data.get("host", "127.0.0.1"))
        self.port = int(data.get("port", 11434))

        gemini_cfg = data.get("gemini", {}) if isinstance(data.get("gemini"), dict) else {}
        self.api_url = str(gemini_cfg.get("api_url") or data.get("api_url") or "").rstrip("/")

        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        cfg_key = gemini_cfg.get("api_key") or data.get("api_key")
        if not cfg_key and isinstance(data.get("providers"), list):
            for p in data["providers"]:
                if p.get("name") == "gemini" and p.get("api_key"):
                    cfg_key = p["api_key"]
                    break

        self.api_key = str(env_key or cfg_key or "").strip()
        self.default_model = str(gemini_cfg.get("default_model") or data.get("default_model") or DEFAULT_MODEL).strip()
        self.timeout = float(data.get("timeout", 180))
        self.cache_ttl = float(data.get("cache_ttl", 300))
        self.fetch_wait_timeout = float(data.get("fetch_wait_timeout", 30))
        self.max_body_bytes = int(data.get("max_body_bytes", 64 * 1024 * 1024))
        self.default_num_ctx = int(data.get("default_num_ctx", 1048576))
        self.models_dir = os.path.abspath(os.path.join(base_dir, str(data.get("models_dir", "models"))))
        self.use_env_proxy = bool(data.get("use_env_proxy", True))
        self.log_level = str(data.get("log_level", "info")).lower()
        if self.log_level not in ("quiet", "info", "debug"):
            self.log_level = "info"
        self.mapping = data.get("mapping", {}) or {}
        self.custom_models = list(data.get("models", []))


class GeminiClient:
    """基于 google-genai SDK 的 Google Gemini API 客户端。"""

    def __init__(self, config):
        self.config = config
        self.log_level = config.log_level
        self.lock = threading.RLock()
        self._sdk_client = None
        self.model_cache = []
        self.fetched_at = 0
        self.models_entries = None
        self.models_defaults = {}
        self._thought_signatures = {}
        self._latest_valid_signature = None
        self._sig_cache_file = os.path.join(SCRIPT_DIR, "tmp", "thought_signatures.json")
        self._load_signatures_from_disk()

        if genai is None:
            raise RuntimeError("未安装 google-genai SDK。请运行: pip install google-genai")

    def _load_signatures_from_disk(self):
        try:
            if os.path.exists(self._sig_cache_file):
                with open(self._sig_cache_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    for k, b64_val in raw.items():
                        try:
                            sig_bytes = base64.b64decode(b64_val)
                            self._thought_signatures[k] = sig_bytes
                            self._latest_valid_signature = sig_bytes
                        except Exception:
                            pass
        except Exception:
            pass

    def _save_signatures_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self._sig_cache_file), exist_ok=True)
            data = {}
            items = list(self._thought_signatures.items())[-500:]
            for k, val in items:
                if isinstance(val, (bytes, bytearray)):
                    data[k] = base64.b64encode(val).decode("ascii")
            with open(self._sig_cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def cache_thought_signature(self, name, args, sig, fn_id=None):
        if not sig:
            return
        if not isinstance(sig, (bytes, bytearray)):
            try:
                sig = bytes(sig)
            except Exception:
                return
        with self.lock:
            self._latest_valid_signature = sig
            if len(self._thought_signatures) > 1000:
                self._thought_signatures.clear()
            if fn_id:
                self._thought_signatures[f"id:{fn_id}"] = sig
            args_key = json.dumps(args, sort_keys=True, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
            self._thought_signatures[f"fn:{name}:{args_key}"] = sig
            self._thought_signatures[f"latest:{name}"] = sig
            self._save_signatures_to_disk()

    def get_thought_signature(self, name, args, fn_id=None):
        with self.lock:
            if fn_id and f"id:{fn_id}" in self._thought_signatures:
                return self._thought_signatures[f"id:{fn_id}"]
            args_key = json.dumps(args, sort_keys=True, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
            if f"fn:{name}:{args_key}" in self._thought_signatures:
                return self._thought_signatures[f"fn:{name}:{args_key}"]
            if f"latest:{name}" in self._thought_signatures:
                return self._thought_signatures[f"latest:{name}"]
            return self._latest_valid_signature

    @staticmethod
    def extract_text_tool_calls(text):
        """解析文本中的伪工具调用, 如 [Call tool `grep` with arguments: {...}]"""
        if not text or "[Call tool " not in text:
            return text, []
        pattern = re.compile(r"\[Call tool [`']?([a-zA-Z0-9_.:-]+)[`']? with arguments:\s*(\{.*?\})\]", re.DOTALL)
        tool_calls = []
        for match in pattern.finditer(text):
            name = match.group(1)
            raw_args = match.group(2)
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {"raw": raw_args}
            tool_calls.append({
                "function": {
                    "name": name,
                    "arguments": args
                }
            })
        cleaned_text = pattern.sub("", text).strip()
        return cleaned_text, tool_calls

    def log(self, msg, level="info"):
        levels = {"quiet": 0, "info": 1, "debug": 2}
        want = levels.get(self.log_level, 1)
        if want < levels.get(level, 1):
            return
        stream = sys.stderr if level == "error" else sys.stdout
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), file=stream, flush=True)

    def get_sdk_client(self):
        if self._sdk_client is None:
            if not self.config.api_key:
                raise ValueError("未配置 Gemini API Key。请在 config.json 中配置 gemini.api_key 或通过环境变量 GEMINI_API_KEY / 启动参数 --api-key 指定。")
            http_options = {}
            if self.config.api_url:
                http_options["base_url"] = self.config.api_url
            self._sdk_client = genai.Client(
                api_key=self.config.api_key,
                http_options=http_options if http_options else None
            )
        return self._sdk_client

    def fetch_models(self):
        if self.config.custom_models:
            return list(self.config.custom_models)

        now = time.time()
        with self.lock:
            if self.model_cache and now - self.fetched_at < self.config.cache_ttl:
                return list(self.model_cache)

        try:
            client = self.get_sdk_client()
            models = []
            for m in client.models.list():
                raw_name = str(getattr(m, "name", "") or "")
                clean = raw_name.split("/")[-1] if raw_name.startswith("models/") else raw_name
                # 过滤掉非文本生成模型
                if clean and clean not in models:
                    models.append(clean)
            if models:
                self.log("通过 google-genai 成功拉取 %d 个 Gemini 模型" % len(models))
                with self.lock:
                    self.model_cache = models
                    self.fetched_at = now
                return models
        except Exception as exc:
            self.log("google-genai 拉取模型列表失败，使用本地模板: %s" % exc, level="debug")

        templates = self.load_models_templates()
        if templates:
            ids = []
            for t in templates:
                name = t.get("model") or t.get("name")
                if name and name not in ids:
                    ids.append(name)
            if ids:
                return ids

        return ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.7-flash", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]

    def warm_models(self):
        if not self.config.api_key:
            return

        def _probe():
            try:
                models = self.fetch_models()
                self.log("Gemini 初始化探测成功, 可用模型数: %d" % len(models))
                if self._latest_valid_signature is None:
                    try:
                        client = self.get_sdk_client()
                        probe_tool = [types.Tool(function_declarations=[
                            types.FunctionDeclaration(name="probe", description="probe", parameters={"type": "object", "properties": {"x": {"type": "string"}}})
                        ])]
                        resp = client.models.generate_content(
                            model=self.normalize_model_name(self.config.default_model),
                            contents=[types.Content(role="user", parts=[types.Part.from_text(text="call probe x=1")])],
                            config=types.GenerateContentConfig(
                                tools=probe_tool,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                            )
                        )
                        if resp.candidates and resp.candidates[0].content:
                            for p in resp.candidates[0].content.parts:
                                sig = getattr(p, "thought_signature", None)
                                if sig:
                                    self.cache_thought_signature("probe", {"x": "1"}, sig)
                                    self.log("Gemini 思考签名预热成功", level="debug")
                                    break
                    except Exception as sig_err:
                        self.log("Gemini 思考签名预热跳过: %s" % sig_err, level="debug")
            except Exception as e:
                self.log("Gemini 初始化探测: %s" % e, level="debug")
        threading.Thread(target=_probe, daemon=True, name="gemini-warm").start()

    def normalize_model_name(self, name):
        name = (name or "").strip()
        if not name:
            return self.config.default_model

        if name in self.config.mapping:
            target = self.config.mapping[name]
            if isinstance(target, str):
                return target
            if isinstance(target, dict) and target.get("model"):
                return target["model"]

        base = name.split(":", 1)[0]
        if base.startswith("models/"):
            base = base[len("models/"):]

        alias_map = {
            "gemini": self.config.default_model,
            "flash": "gemini-3.6-flash",
            "flash-lite": "gemini-3.5-flash-lite",
            "pro": "gemini-3.1-pro-preview",
            "gemini-flash": "gemini-3.6-flash",
            "gemini-pro": "gemini-3.1-pro-preview",
            "gemini-2.5-flash": "gemini-3.6-flash",
        }
        if base.lower() in alias_map:
            return alias_map[base.lower()]
        return base

    def load_models_templates(self):
        if self.models_entries is None:
            entries = []
            defaults = {}
            if os.path.isdir(self.config.models_dir):
                for root, _dirs, filenames in os.walk(self.config.models_dir):
                    for filename in sorted(filenames):
                        if filename.lower().endswith(".json"):
                            path = os.path.join(root, filename)
                            try:
                                with open(path, "r", encoding="utf-8") as fh:
                                    data = json.load(fh)
                                    if isinstance(data, dict):
                                        if data.get("provider") == "gemini" or "gemini" in filename.lower():
                                            defaults = data.get("defaults") or defaults
                                            for m in data.get("models", []):
                                                if isinstance(m, dict):
                                                    entries.append(m)
                            except Exception:
                                pass
            self.models_defaults = defaults
            self.models_entries = entries
        return self.models_entries

    def find_template(self, model_name):
        entries = self.load_models_templates()
        clean = self.normalize_model_name(model_name).lower()
        for e in entries:
            name = str(e.get("name") or e.get("model") or "").lower()
            if name == clean or name == clean.split(":", 1)[0]:
                return e
        return None

    def get_tags(self):
        model_ids = self.fetch_models()
        tags_list = []
        seen = set()

        for mid in model_ids:
            clean = mid.split(":", 1)[0]
            if clean in seen:
                continue
            seen.add(clean)

            tmpl = self.find_template(clean)
            tag_name = clean + ":latest"
            size = (tmpl.get("size") if tmpl else 0) or 70000000000
            family = "gemma" if "gemma" in clean.lower() else "gemini"
            param_size = (tmpl.get("details", {}).get("parameter_size") if tmpl else "") or "70B"

            tags_list.append({
                "name": tag_name,
                "model": tag_name,
                "modified_at": (tmpl.get("modified_at") if tmpl else None) or now_iso(),
                "size": size,
                "digest": (tmpl.get("digest") if tmpl else None) or uuid.uuid5(uuid.NAMESPACE_URL, "gemini:" + clean).hex[:12],
                "details": {
                    "parent_model": "",
                    "format": "",
                    "family": family,
                    "families": None,
                    "parameter_size": param_size,
                    "quantization_level": (tmpl.get("details", {}).get("quantization_level") if tmpl else "") or "none",
                }
            })

        return {"models": tags_list}

    def get_show(self, name):
        clean = self.normalize_model_name(name)
        tmpl = self.find_template(clean)

        family = "gemma" if "gemma" in clean.lower() else "gemini"
        param_size = (tmpl.get("details", {}).get("parameter_size") if tmpl else "") or "70B"
        context_len = (tmpl.get("model_info", {}).get("%s.context_length" % family) if tmpl else None) or self.config.default_num_ctx

        return {
            "capabilities": (tmpl.get("capabilities") if tmpl else None) or ["completion", "tools", "thinking", "vision"],
            "details": {
                "parent_model": "",
                "format": "",
                "family": family,
                "families": None,
                "parameter_size": param_size,
                "quantization_level": (tmpl.get("details", {}).get("quantization_level") if tmpl else "") or "none",
            },
            "model_info": {
                "general.architecture": family,
                "general.parameter_count": (tmpl.get("size") if tmpl else None) or 70000000000,
                "%s.context_length" % family: context_len,
                "%s.embedding_length" % family: 4096,
            },
            "modified_at": (tmpl.get("modified_at") if tmpl else None) or now_iso(),
        }

    @staticmethod
    def _get_image_mime(data_bytes):
        if data_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data_bytes[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if data_bytes[:4] == b"RIFF" and data_bytes[8:12] == b"WEBP":
            return "image/webp"
        if data_bytes[:2] == b"BM":
            return "image/bmp"
        return "image/png"

    @staticmethod
    def sanitize_schema(raw):
        if not isinstance(raw, dict):
            return raw
        disallowed = {
            "additionalProperties", "additional_properties",
            "$schema", "$defs", "definitions", "title", "default"
        }
        cleaned = {}
        for k, v in raw.items():
            if k in disallowed:
                continue
            if k == "properties" and isinstance(v, dict):
                cleaned["properties"] = {pk: GeminiClient.sanitize_schema(pv) for pk, pv in v.items()}
            elif k == "items":
                if isinstance(v, dict):
                    cleaned["items"] = GeminiClient.sanitize_schema(v)
                elif isinstance(v, list):
                    cleaned["items"] = [GeminiClient.sanitize_schema(x) for x in v]
                else:
                    cleaned["items"] = v
            elif k in ("anyOf", "any_of") and isinstance(v, list):
                cleaned["any_of"] = [GeminiClient.sanitize_schema(x) for x in v]
            elif k in ("oneOf", "one_of") and isinstance(v, list):
                cleaned["any_of"] = [GeminiClient.sanitize_schema(x) for x in v]
            elif k in ("allOf", "all_of") and isinstance(v, list):
                if v:
                    cleaned["any_of"] = [GeminiClient.sanitize_schema(x) for x in v]
            elif isinstance(v, dict):
                cleaned[k] = GeminiClient.sanitize_schema(v)
            elif isinstance(v, list):
                cleaned[k] = [GeminiClient.sanitize_schema(x) for x in v]
            else:
                cleaned[k] = v
        if not cleaned:
            return {"type": "object"}
        return cleaned

    def convert_messages_and_config(self, messages, options=None, tools=None, system=None):
        contents = []
        system_instruction_text = str(system) if system else None
        tool_call_id_to_name = {}

        for msg in messages or []:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_images = msg.get("images") or []

            if role == "system":
                if isinstance(content, str):
                    system_instruction_text = (system_instruction_text + "\n" + content) if system_instruction_text else content
                continue

            genai_role = "model" if role == "assistant" else "user"
            parts = []

            if role not in ("tool", "function"):
                if isinstance(content, str) and content:
                    parts.append(types.Part.from_text(text=content))
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append(types.Part.from_text(text=item.get("text", "")))
                            elif item.get("type") == "image_url":
                                img_url = (item.get("image_url") or {}).get("url", "")
                                if img_url.startswith("data:"):
                                    mime = img_url.split(";")[0].replace("data:", "")
                                    b64 = img_url.split(",", 1)[1] if "," in img_url else ""
                                    try:
                                        raw = base64.b64decode(b64)
                                        parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
                                    except Exception:
                                        pass
                        elif isinstance(item, str):
                            parts.append(types.Part.from_text(text=item))

                for img in msg_images:
                    val = str(img).strip()
                    if "," in val and val.lstrip().lower().startswith("data:"):
                        val = val.split(",", 1)[1]
                    try:
                        raw = base64.b64decode(val)
                        mime = self._get_image_mime(raw)
                        parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
                    except Exception:
                        pass

            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                fn_name = fn.get("name") or tc.get("name") or ""
                fn_id = tc.get("id") or (tc.get("function", {}).get("id") if isinstance(tc, dict) else None) or ""
                if fn_id and fn_name:
                    tool_call_id_to_name[fn_id] = fn_name
                args = fn.get("arguments", {}) if "arguments" in fn else tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                if not isinstance(args, dict):
                    args = {"value": args}

                sig = self.get_thought_signature(fn_name, args, fn_id)
                if sig:
                    parts.append(types.Part(
                        function_call=types.FunctionCall(name=fn_name, args=args, id=fn_id if fn_id else None),
                        thought_signature=sig
                    ))
                else:
                    parts.append(types.Part.from_function_call(name=fn_name, args=args))

            if role in ("tool", "function"):
                name = msg.get("name") or tool_call_id_to_name.get(msg.get("tool_call_id", "")) or "function_name"
                resp_obj = content
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        resp_obj = parsed if isinstance(parsed, dict) else {"result": parsed}
                    except Exception:
                        resp_obj = {"result": content}
                elif not isinstance(content, dict):
                    resp_obj = {"result": content}

                parts.append(types.Part.from_function_response(name=name, response=resp_obj))
                genai_role = "user"

            if not parts:
                parts.append(types.Part.from_text(text=""))
            contents.append(types.Content(role=genai_role, parts=parts))

        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="")]))

        opts = options or {}
        sdk_tools = None
        if tools:
            decls = []
            for t in tools:
                if t.get("type") == "function":
                    fn = t.get("function", {})
                    params = self.sanitize_schema(fn.get("parameters") or {})
                    decls.append(types.FunctionDeclaration(
                        name=fn.get("name"),
                        description=fn.get("description"),
                        parameters=params
                    ))
                elif t.get("name"):
                    params = self.sanitize_schema(t.get("parameters") or {})
                    decls.append(types.FunctionDeclaration(
                        name=t.get("name"),
                        description=t.get("description"),
                        parameters=params
                    ))
            if decls:
                sdk_tools = [types.Tool(function_declarations=decls)]

        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction_text,
            temperature=float(opts["temperature"]) if opts.get("temperature") is not None else None,
            top_p=float(opts["top_p"]) if opts.get("top_p") is not None else None,
            top_k=int(opts["top_k"]) if opts.get("top_k") is not None else None,
            max_output_tokens=int(opts["num_predict"] if opts.get("num_predict") and int(opts["num_predict"]) > 0 else (opts.get("max_tokens") or 0)) or None,
            stop_sequences=[opts["stop"]] if isinstance(opts.get("stop"), str) else (list(opts["stop"]) if opts.get("stop") else None),
            tools=sdk_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if sdk_tools else None,
        )

        return contents, gen_config

    def chat(self, body):
        t0 = time.time()
        model_name = body.get("model", "")
        clean_model = self.normalize_model_name(model_name)
        contents, config = self.convert_messages_and_config(
            messages=body.get("messages", []),
            options=body.get("options", {}),
            tools=body.get("tools")
        )
        self.log("chat model=%s target=%s stream=false" % (model_name, clean_model))
        client = self.get_sdk_client()
        try:
            resp = client.models.generate_content(
                model=clean_model,
                contents=contents,
                config=config
            )
        except Exception as exc:
            self.log("Gemini SDK 错误: %s" % exc, level="error")
            raise UpstreamError(500, str(exc)) from exc

        text_parts = []
        tool_calls = []
        if resp.candidates:
            cand = resp.candidates[0]
            if cand.content:
                for p in cand.content.parts:
                    if getattr(p, "text", None):
                        text_parts.append(p.text)
                    elif getattr(p, "function_call", None):
                        sig = getattr(p, "thought_signature", None)
                        fn_id = getattr(p.function_call, "id", None)
                        args = getattr(p.function_call, "args", {})
                        self.cache_thought_signature(p.function_call.name, args, sig, fn_id)
                        tool_calls.append({
                            "function": {
                                "name": p.function_call.name,
                                "arguments": args
                            }
                        })

        full_text = "".join(text_parts)
        if not tool_calls and full_text:
            cleaned_text, fallback_tcs = self.extract_text_tool_calls(full_text)
            if fallback_tcs:
                tool_calls.extend(fallback_tcs)
                full_text = cleaned_text

        prompt_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) if resp.usage_metadata else 0
        completion_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) if resp.usage_metadata else 0

        omsg = {"role": "assistant", "content": full_text}
        done_reason = "stop"
        if tool_calls:
            omsg["tool_calls"] = tool_calls
            done_reason = "tool_calls"

        out = {
            "model": model_name,
            "created_at": now_iso(),
            "message": omsg,
            "done": True,
            "done_reason": done_reason,
            "total_duration": int((time.time() - t0) * 1e9),
            "load_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }
        self.log("chat 完成 model=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            model_name, prompt_tokens, completion_tokens, time.time() - t0))
        return out

    def chat_stream(self, body, write):
        t0 = time.time()
        model_name = body.get("model", "")
        clean_model = self.normalize_model_name(model_name)
        contents, config = self.convert_messages_and_config(
            messages=body.get("messages", []),
            options=body.get("options", {}),
            tools=body.get("tools")
        )
        self.log("chat model=%s target=%s stream=true" % (model_name, clean_model))
        client = self.get_sdk_client()

        tool_calls = []
        stream_text_parts = []
        prompt_tokens = 0
        completion_tokens = 0
        final_reason = "stop"

        try:
            for chunk in client.models.generate_content_stream(
                model=clean_model,
                contents=contents,
                config=config
            ):
                if chunk.candidates:
                    cand = chunk.candidates[0]
                    if cand.content:
                        for p in cand.content.parts:
                            if getattr(p, "text", None) and p.text:
                                stream_text_parts.append(p.text)
                                write(ndjson({
                                    "model": model_name,
                                    "created_at": now_iso(),
                                    "message": {"role": "assistant", "content": p.text},
                                    "done": False,
                                }))
                            elif getattr(p, "function_call", None):
                                sig = getattr(p, "thought_signature", None)
                                fn_id = getattr(p.function_call, "id", None)
                                args = getattr(p.function_call, "args", {})
                                self.cache_thought_signature(p.function_call.name, args, sig, fn_id)
                                tc = {
                                    "function": {
                                        "name": p.function_call.name,
                                        "arguments": args
                                    }
                                }
                                tool_calls.append(tc)
                                write(ndjson({
                                    "model": model_name,
                                    "created_at": now_iso(),
                                    "message": {"role": "assistant", "content": "", "tool_calls": [tc]},
                                    "done": False,
                                }))
                if chunk.usage_metadata:
                    prompt_tokens = getattr(chunk.usage_metadata, "prompt_token_count", prompt_tokens)
                    completion_tokens = getattr(chunk.usage_metadata, "candidates_token_count", completion_tokens)
        except Exception as exc:
            self.log("Gemini 流式错误: %s" % exc, level="error")
            raise UpstreamError(500, str(exc)) from exc

        if not tool_calls and stream_text_parts:
            _, fallback_tcs = self.extract_text_tool_calls("".join(stream_text_parts))
            if fallback_tcs:
                tool_calls.extend(fallback_tcs)

        final_msg = {"role": "assistant", "content": ""}
        if tool_calls:
            final_msg["tool_calls"] = tool_calls
            final_reason = "tool_calls"

        write(ndjson({
            "model": model_name,
            "created_at": now_iso(),
            "message": final_msg,
            "done": True,
            "done_reason": final_reason,
            "total_duration": int((time.time() - t0) * 1e9),
            "load_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }))
        self.log("chat 流式完成 model=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            model_name, prompt_tokens, completion_tokens, time.time() - t0))

    def generate(self, body):
        t0 = time.time()
        model_name = body.get("model", "")
        clean_model = self.normalize_model_name(model_name)
        prompt = body.get("prompt", "")
        images = body.get("images") or []
        system = body.get("system")
        messages = [{"role": "user", "content": prompt, "images": images}]

        contents, config = self.convert_messages_and_config(
            messages=messages,
            options=body.get("options", {}),
            system=system
        )
        self.log("generate model=%s target=%s stream=false" % (model_name, clean_model))
        client = self.get_sdk_client()
        try:
            resp = client.models.generate_content(
                model=clean_model,
                contents=contents,
                config=config
            )
        except Exception as exc:
            self.log("Gemini SDK 错误: %s" % exc, level="error")
            raise UpstreamError(500, str(exc)) from exc

        text_parts = []
        if resp.candidates and resp.candidates[0].content:
            for p in resp.candidates[0].content.parts:
                if getattr(p, "text", None):
                    text_parts.append(p.text)

        prompt_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) if resp.usage_metadata else 0
        completion_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) if resp.usage_metadata else 0

        out = {
            "model": model_name,
            "created_at": now_iso(),
            "response": "".join(text_parts),
            "done": True,
            "done_reason": "stop",
            "context": [],
            "total_duration": int((time.time() - t0) * 1e9),
            "load_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }
        self.log("generate 完成 model=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            model_name, prompt_tokens, completion_tokens, time.time() - t0))
        return out

    def generate_stream(self, body, write):
        t0 = time.time()
        model_name = body.get("model", "")
        clean_model = self.normalize_model_name(model_name)
        prompt = body.get("prompt", "")
        images = body.get("images") or []
        system = body.get("system")
        messages = [{"role": "user", "content": prompt, "images": images}]

        contents, config = self.convert_messages_and_config(
            messages=messages,
            options=body.get("options", {}),
            system=system
        )
        self.log("generate model=%s target=%s stream=true" % (model_name, clean_model))
        client = self.get_sdk_client()

        prompt_tokens = 0
        completion_tokens = 0

        try:
            for chunk in client.models.generate_content_stream(
                model=clean_model,
                contents=contents,
                config=config
            ):
                if chunk.candidates and chunk.candidates[0].content:
                    for p in chunk.candidates[0].content.parts:
                        if getattr(p, "text", None) and p.text:
                            write(ndjson({
                                "model": model_name,
                                "created_at": now_iso(),
                                "response": p.text,
                                "done": False,
                            }))
                if chunk.usage_metadata:
                    prompt_tokens = getattr(chunk.usage_metadata, "prompt_token_count", prompt_tokens)
                    completion_tokens = getattr(chunk.usage_metadata, "candidates_token_count", completion_tokens)
        except Exception as exc:
            self.log("Gemini 流式错误: %s" % exc, level="error")
            raise UpstreamError(500, str(exc)) from exc

        write(ndjson({
            "model": model_name,
            "created_at": now_iso(),
            "response": "",
            "done": True,
            "done_reason": "stop",
            "context": [],
            "total_duration": int((time.time() - t0) * 1e9),
            "load_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }))
        self.log("generate 流式完成 model=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            model_name, prompt_tokens, completion_tokens, time.time() - t0))

    def v1_models(self):
        model_ids = self.fetch_models()
        data = []
        for mid in model_ids:
            data.append({
                "id": mid,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google",
            })
        return {"object": "list", "data": data}

    def v1_chat(self, body, stream=False):
        model_name = body.get("model", "")
        clean_model = self.normalize_model_name(model_name)
        options = {}
        for key in ("temperature", "top_p", "top_k", "max_tokens", "stop"):
            if body.get(key) is not None:
                options[key] = body[key]

        if not stream:
            chat_res = self.chat({"model": clean_model, "messages": body.get("messages", []), "options": options, "tools": body.get("tools")})
            omsg = chat_res.get("message", {})
            tool_calls = []
            for i, tc in enumerate(omsg.get("tool_calls") or []):
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args_str}
                })

            msg = {"role": "assistant", "content": omsg.get("content") or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "finish_reason": chat_res.get("done_reason", "stop"),
                }],
                "usage": {
                    "prompt_tokens": chat_res.get("prompt_eval_count", 0),
                    "completion_tokens": chat_res.get("eval_count", 0),
                    "total_tokens": chat_res.get("prompt_eval_count", 0) + chat_res.get("eval_count", 0),
                }
            }

        return self._v1_chat_stream(body, clean_model, model_name, options)

    def _v1_chat_stream(self, body, clean_model, model_name, options):
        base_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        finish_reason = "stop"
        tool_calls = []
        prompt_tokens = 0
        completion_tokens = 0

        contents, config = self.convert_messages_and_config(
            messages=body.get("messages", []),
            options=options,
            tools=body.get("tools")
        )
        client = self.get_sdk_client()

        for chunk in client.models.generate_content_stream(
            model=clean_model,
            contents=contents,
            config=config
        ):
            if chunk.candidates and chunk.candidates[0].content:
                for p in chunk.candidates[0].content.parts:
                    if getattr(p, "text", None) and p.text:
                        item = {
                            "id": base_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": p.text}, "finish_reason": None}],
                        }
                        yield ("data: " + json.dumps(item, ensure_ascii=False) + "\n\n").encode("utf-8")
                    elif getattr(p, "function_call", None):
                        fn = p.function_call
                        args = getattr(fn, "args", {})
                        sig = getattr(p, "thought_signature", None)
                        fn_id = getattr(fn, "id", None) or f"call_{len(tool_calls)}"
                        self.cache_thought_signature(getattr(fn, "name", ""), args, sig, fn_id)
                        args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                        tc_chunk = {
                            "id": base_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"tool_calls": [{
                                "index": len(tool_calls),
                                "id": fn_id,
                                "type": "function",
                                "function": {"name": getattr(fn, "name", ""), "arguments": args_str},
                            }]}, "finish_reason": None}],
                        }
                        tool_calls.append(fn)
                        finish_reason = "tool_calls"
                        yield ("data: " + json.dumps(tc_chunk, ensure_ascii=False) + "\n\n").encode("utf-8")
            if chunk.usage_metadata:
                prompt_tokens = getattr(chunk.usage_metadata, "prompt_token_count", prompt_tokens)
                completion_tokens = getattr(chunk.usage_metadata, "candidates_token_count", completion_tokens)

        final_chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        if prompt_tokens or completion_tokens:
            final_chunk["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        yield ("data: " + json.dumps(final_chunk, ensure_ascii=False) + "\n\n").encode("utf-8")
        yield b"data: [DONE]\n\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gemini-ollama-proxy/" + __version__

    @property
    def client(self):
        return self.server.client

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def log_message(self, format, *args):
        pass

    def read_body(self):
        length_header = self.headers.get("Content-Length")
        limit = self.client.config.max_body_bytes
        if length_header:
            try:
                length = int(length_header)
            except ValueError:
                raise ValueError("非法的 Content-Length 头")
            if limit and length > limit:
                raise BodyTooLargeError("请求体大小 %d 超过上限 %d" % (length, limit))
            return self.rfile.read(length)
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            data = bytearray()
            while True:
                line = self.rfile.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                try:
                    size = int(line.strip().split(b";", 1)[0], 16)
                except ValueError:
                    break
                if size == 0:
                    self.rfile.readline()
                    break
                if limit and len(data) + size > limit:
                    raise BodyTooLargeError("分块请求体超过上限 %d" % limit)
                data.extend(self.rfile.read(size))
                self.rfile.readline()
            return bytes(data)
        return b""

    def json_body(self):
        raw = self.read_body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("请求体不是合法 JSON: %s" % exc)

    def send_json(self, obj, status=200, headers=None):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def send_text(self, text, status=200):
        body = text.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def start_stream(self, content_type):
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def write_chunk(self, data):
        try:
            self.wfile.write(("%X\r\n" % len(data)).encode("ascii"))
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def end_stream(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    def send_error_json(self, message, status=500):
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/v1/"):
            obj = {"error": {"message": message, "type": "proxy_error", "code": status}}
        else:
            obj = {"error": message}
        self.send_json(obj, status=status)

    def do_OPTIONS(self):
        try:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-goog-api-key")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/":
                self.send_text("Ollama is running")
            elif path == "/api/version":
                self.send_json({"version": OLLAMA_VERSION})
            elif path == "/api/tags":
                self.send_json(self.client.get_tags())
            elif path == "/api/ps":
                self.send_json({"models": []})
            elif path == "/v1/models":
                self.send_json(self.client.v1_models())
            else:
                self.send_error_json("not found: " + path, 404)
        except UpstreamError as exc:
            self.send_error_json("upstream error: %s" % (exc.body or exc), 502)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self.client.log("GET %s 失败: %s" % (path, exc), level="error")
            self.send_error_json("internal error: %s" % exc, 500)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self.json_body()
            if path == "/api/chat":
                if body.get("stream"):
                    self.start_stream("application/x-ndjson")
                    try:
                        self.client.chat_stream(body, self.write_chunk)
                    except Exception as stream_exc:
                        self.client.log("API chat 流式异常: %s" % stream_exc, level="error")
                        self.write_chunk(ndjson({"error": str(stream_exc), "done": True}))
                    finally:
                        self.end_stream()
                else:
                    self.send_json(self.client.chat(body))
            elif path == "/api/generate":
                if body.get("stream"):
                    self.start_stream("application/x-ndjson")
                    try:
                        self.client.generate_stream(body, self.write_chunk)
                    except Exception as stream_exc:
                        self.client.log("API generate 流式异常: %s" % stream_exc, level="error")
                        self.write_chunk(ndjson({"error": str(stream_exc), "done": True}))
                    finally:
                        self.end_stream()
                else:
                    self.send_json(self.client.generate(body))
            elif path == "/api/show":
                name = body.get("name") or body.get("model")
                if not name:
                    raise ValueError("请求缺少模型 name")
                self.send_json(self.client.get_show(name))
            elif path == "/v1/chat/completions":
                if body.get("stream"):
                    self.start_stream("text/event-stream")
                    try:
                        for chunk in self.client.v1_chat(body, stream=True):
                            self.write_chunk(chunk)
                    except Exception as stream_exc:
                        self.client.log("OpenAI v1_chat 流式异常: %s" % stream_exc, level="error")
                        err_chunk = {
                            "error": {
                                "message": str(stream_exc),
                                "type": "upstream_error",
                                "code": 500
                            }
                        }
                        self.write_chunk(("data: " + json.dumps(err_chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))
                        self.write_chunk(b"data: [DONE]\n\n")
                    finally:
                        self.end_stream()
                else:
                    self.send_json(self.client.v1_chat(body, stream=False))
            else:
                self.send_error_json("not found: " + path, 404)
        except BodyTooLargeError as exc:
            self.send_error_json(str(exc), 413)
        except UpstreamError as exc:
            self.send_error_json("upstream error: %s" % (exc.body or exc), 502)
        except ValueError as exc:
            self.send_error_json(str(exc), 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self.client.log("POST %s 失败: %s" % (path, exc), level="error")
            self.send_error_json("internal error: %s" % exc, 500)


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, client):
        super().__init__(server_address, RequestHandlerClass)
        self.client = client


def main():
    parser = argparse.ArgumentParser(description="Google Gemini API -> Ollama API 代理 (基于 google-genai SDK)")
    parser.add_argument("--api-key", default=None, help="Gemini API Key (也可通过 config.json 或环境变量 GEMINI_API_KEY 配置)")
    parser.add_argument("--default-model", default=None, help="默认模型 (默认 gemini-3.6-flash)")
    parser.add_argument("--config", default=None, help="配置文件路径 (默认查找 config.json)")
    parser.add_argument("--host", default=None, help="监听地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="监听端口 (默认 11434)")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    parser.add_argument("--version", action="version", version="gemini-ollama-proxy %s" % __version__)
    args = parser.parse_args()

    cfg_data = {}
    cfg_base_dir = SCRIPT_DIR
    cfg_path = os.path.abspath(args.config) if args.config else DEFAULT_CONFIG
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                cfg_base_dir = os.path.dirname(cfg_path)
        except Exception as exc:
            print("[warn] 读取配置文件 %s 失败: %s" % (cfg_path, exc), file=sys.stderr)

    config = Config(cfg_data, cfg_base_dir)
    if args.api_key:
        config.api_key = args.api_key
    if args.default_model:
        config.default_model = args.default_model
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.verbose:
        config.log_level = "debug"

    client = GeminiClient(config)
    client.warm_models()

    server = ProxyServer((config.host, config.port), Handler, client)

    print("gemini-ollama-proxy 已启动: http://%s:%d (引擎: google-genai SDK)" % (config.host, config.port))
    print("  - 默认模型: %s" % config.default_model)
    key_info = (config.api_key[:6] + "..." + config.api_key[-4:]) if len(config.api_key) > 10 else (config.api_key or "未配置")
    print("  - API Key: %s" % key_info)
    print("日志级别: %s (--verbose 可开启 debug)" % config.log_level)
    print("按 Ctrl+C 停止")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
