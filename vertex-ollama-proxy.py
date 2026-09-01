#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vertex-ollama-proxy
===================

把 Google Cloud Vertex AI Agent Platform (generateContent / streamGenerateContent)
转换为 Ollama API, 同时提供 OpenAI 兼容接口 /v1/chat/completions 与 /v1/models。
支持 API Key (x-goog-api-key) 与 OAuth2 Bearer Token (Authorization: Bearer) 双鉴权模式。
仅依赖 Python 标准库，零第三方依赖。敏感信息（如 API Key, Project ID）从 config.json 或环境变量中读取。

端点
----
GET  /                      返回 "Ollama is running"
GET  /api/version           返回模拟的 Ollama 版本号
GET  /api/tags              汇总 Vertex AI 模型列表, 优先匹配 models/vertex.json, 未命中自动生成
POST /api/show              优先读取 models/vertex.json, 未命中自动生成模型详情
POST /api/chat              转换为 Vertex AI API 转发 (支持流式 NDJSON)
POST /api/generate          转换为 Vertex AI API 转发 (支持流式 NDJSON 与多模态图片)
GET  /api/ps                返回空模型列表 (兼容 Ollama 状态轮询)
GET  /v1/models             OpenAI 兼容模型列表
POST /v1/chat/completions   OpenAI 兼容请求透传 (支持流式 SSE)

用法
----
    python vertex-ollama-proxy.py
    python vertex-ollama-proxy.py --config config.json
    python vertex-ollama-proxy.py --api-key <API_KEY> --vertex-project <PROJECT_ID>
    python vertex-ollama-proxy.py --bearer-token <OAUTH_TOKEN> --vertex-project <PROJECT_ID>
    python vertex-ollama-proxy.py --port 11434 --verbose
"""

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"
OLLAMA_VERSION = "0.5.4"
__version__ = "1.2.0"


def now_iso():
    """Ollama 风格 RFC3339 时间戳, 如 2026-09-01T12:00:00.123Z"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def ndjson(obj):
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def iter_sse_events(resp):
    """逐条解析 Vertex AI streamGenerateContent 的 SSE 事件, 产出 (event_type, data_dict/raw) 元组。"""
    event_type = None
    data_lines = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif line == "" and data_lines:
            payload = "\n".join(data_lines)
            if payload.lower() == "[done]":
                yield ("done", "[DONE]")
            else:
                try:
                    data = json.loads(payload)
                except Exception:
                    data = {"raw": payload}
                yield (event_type or data.get("event_type", ""), data)
            event_type = None
            data_lines = []
    if data_lines:
        payload = "\n".join(data_lines)
        if payload.lower() == "[done]":
            yield ("done", "[DONE]")
        else:
            try:
                data = json.loads(payload)
            except Exception:
                data = {"raw": payload}
            yield (event_type or data.get("event_type", ""), data)


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

        # 优先读取 vertex 专属节，若无则从顶层或 providers 提取
        vertex_cfg = data.get("vertex", {}) if isinstance(data.get("vertex"), dict) else {}

        # 1. API Key 鉴权
        env_key = os.environ.get("VERTEX_API_KEY")
        cfg_key = vertex_cfg.get("api_key") or data.get("api_key")
        if not cfg_key and isinstance(data.get("providers"), list):
            for p in data["providers"]:
                if p.get("name") == "vertex" and p.get("api_key"):
                    cfg_key = p["api_key"]
                    break
        raw_key = str(env_key or cfg_key or "").strip()

        # 2. OAuth2 Bearer Token 鉴权
        env_token = os.environ.get("VERTEX_BEARER_TOKEN") or os.environ.get("GOOGLE_OAUTH_TOKEN")
        cfg_token = vertex_cfg.get("bearer_token") or data.get("bearer_token")
        self.bearer_token = str(env_token or cfg_token or "").strip()

        # 3. ADC / 凭据文件
        env_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        cfg_cred = vertex_cfg.get("credentials_file") or data.get("credentials_file")
        if not cfg_cred and raw_key and (raw_key.lower().endswith(".json") or os.path.exists(os.path.join(base_dir, raw_key))):
            cfg_cred = raw_key
            self.api_key = ""
        else:
            self.api_key = raw_key

        self.credentials_file = str(env_cred or cfg_cred or "").strip()
        if self.credentials_file and not os.path.isabs(self.credentials_file):
            self.credentials_file = os.path.abspath(os.path.join(base_dir, self.credentials_file))

        env_proj = os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        cfg_proj = vertex_cfg.get("vertex_project") or vertex_cfg.get("project") or data.get("vertex_project") or data.get("project")
        self.vertex_project = str(env_proj or cfg_proj or "").strip()

        env_loc = os.environ.get("VERTEX_LOCATION")
        cfg_loc = vertex_cfg.get("vertex_location") or vertex_cfg.get("location") or data.get("vertex_location") or data.get("location")
        self.vertex_location = str(env_loc or cfg_loc or DEFAULT_VERTEX_LOCATION).strip()

        self.default_model = str(vertex_cfg.get("default_model") or data.get("default_model") or DEFAULT_MODEL).strip()
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


class VertexClient:
    """Google Cloud Vertex AI Agent Platform 客户端 (支持 API Key 与 OAuth2 Token)。"""

    def __init__(self, config):
        self.config = config
        self.log_level = config.log_level
        self.lock = threading.RLock()
        self._opener_cache = None
        self.model_cache = []
        self.model_details_cache = {}
        self.fetched_at = 0
        self.models_entries = None
        self.models_defaults = {}
        self._adc_cached_token = None
        self._adc_token_expiry = 0

    def log(self, msg, level="info"):
        levels = {"quiet": 0, "info": 1, "debug": 2}
        want = levels.get(self.log_level, 1)
        if want < levels.get(level, 1):
            return
        stream = sys.stderr if level == "error" else sys.stdout
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), file=stream, flush=True)

    def _opener(self):
        if self._opener_cache is None:
            if self.config.use_env_proxy:
                self._opener_cache = urllib.request.build_opener()
            else:
                self._opener_cache = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return self._opener_cache

    def _get_auth_headers(self):
        """生成鉴权请求头：优先 OAuth2 Bearer Token / ADC 刷新，其次 API Key。"""
        headers = {}
        if self.config.vertex_project:
            headers["X-Goog-User-Project"] = self.config.vertex_project

        # 1. 直接提供 Bearer Token
        if self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
            return headers

        # 2. 如果配置了凭据文件 (ADC JSON 或 service_account.json)，自动刷新 OAuth2 Token
        if self.config.credentials_file or not self.config.api_key:
            cred_path = self.config.credentials_file
            if not cred_path:
                appdata = os.environ.get("APPDATA", "")
                home = os.path.expanduser("~")
                candidate_adcs = [
                    os.path.join(appdata, "gcloud", "application_default_credentials.json"),
                    os.path.join(home, ".config", "gcloud", "application_default_credentials.json"),
                ]
                for p in candidate_adcs:
                    if os.path.exists(p):
                        cred_path = p
                        break

            if cred_path and os.path.exists(cred_path):
                now = time.time()
                if self._adc_cached_token and now < self._adc_token_expiry - 60:
                    headers["Authorization"] = f"Bearer {self._adc_cached_token}"
                    return headers

                try:
                    with open(cred_path, "r", encoding="utf-8") as f:
                        cdata = json.load(f)
                    if cdata.get("type") == "authorized_user" and cdata.get("refresh_token"):
                        token_payload = {
                            "client_id": cdata.get("client_id"),
                            "client_secret": cdata.get("client_secret"),
                            "refresh_token": cdata.get("refresh_token"),
                            "grant_type": "refresh_token",
                        }
                        t_req = urllib.request.Request(
                            "https://oauth2.googleapis.com/token",
                            data=json.dumps(token_payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"}
                        )
                        with self._opener().open(t_req, timeout=15) as t_resp:
                            t_data = json.loads(t_resp.read().decode("utf-8"))
                            self._adc_cached_token = t_data["access_token"]
                            self._adc_token_expiry = now + int(t_data.get("expires_in", 3600))
                            self.log("成功通过 ADC 刷新 OAuth2 Access Token (有效时间 %ds)" % t_data.get("expires_in", 3600), level="debug")
                            headers["Authorization"] = f"Bearer {self._adc_cached_token}"
                            return headers
                except Exception as exc:
                    self.log("ADC Token 刷新失败: %s" % exc, level="debug")

        # 3. 使用 API Key
        if self.config.api_key:
            return {"x-goog-api-key": self.config.api_key}

        raise ValueError(
            "未配置有效的 Vertex AI 凭据！请在 config.json 或环境变量中提供以下任意一项：\n"
            "  1. API Key: vertex.api_key 或环境变量 VERTEX_API_KEY\n"
            "  2. OAuth2 Bearer Token: vertex.bearer_token 或环境变量 VERTEX_BEARER_TOKEN\n"
            "  3. 运行 'gcloud auth application-default login' 生成应用默认凭据 (ADC)"
        )

    def base_url(self):
        proj = self.config.vertex_project or "default"
        loc = self.config.vertex_location or DEFAULT_VERTEX_LOCATION
        return f"https://{loc}-aiplatform.googleapis.com/v1beta1/projects/{proj}/locations/{loc}"

    def request(self, endpoint, payload=None, method=None, stream=False):
        if not self.config.vertex_project:
            raise ValueError("未配置 Vertex Project ID / Number。请在 config.json 中配置 vertex.vertex_project 或通过环境变量 VERTEX_PROJECT / 启动参数 --vertex-project 指定。")

        url = self.base_url() + "/" + endpoint.lstrip("/")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        headers.update(self._get_auth_headers())

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req_method = method or ("POST" if payload is not None else "GET")
        req = urllib.request.Request(url, data=data, headers=headers, method=req_method)

        started = time.time()
        self.log("Vertex 请求 %s %s bytes=%d" % (req_method, url, len(data or b"")), level="debug")
        try:
            resp = self._opener().open(req, timeout=self.config.timeout)
            self.log("Vertex 响应 status=%s type=%s elapsed=%.2fs" % (
                resp.status, resp.headers.get("Content-Type", ""), time.time() - started), level="debug")
            return resp
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", "replace")
            self.log("Vertex HTTP错误 status=%s body=%s elapsed=%.2fs" % (
                exc.code, err_body[:300], time.time() - started), level="error")
            raise UpstreamError(exc.code, err_body) from exc
        except urllib.error.URLError as exc:
            self.log("Vertex 连接错误 reason=%s elapsed=%.2fs" % (exc.reason, time.time() - started), level="error")
            raise UpstreamError(0, str(exc.reason)) from exc

    def fetch_models(self):
        if self.config.custom_models:
            return list(self.config.custom_models)

        now = time.time()
        with self.lock:
            if self.model_cache and now - self.fetched_at < self.config.cache_ttl:
                return list(self.model_cache)

        # 1. 尝试通过 OAuth2 动态获取 Model Garden 模型列表
        try:
            auth_headers = self._get_auth_headers()
            if "Authorization" in auth_headers:
                loc = self.config.vertex_location or DEFAULT_VERTEX_LOCATION
                urls = [
                    "https://aiplatform.googleapis.com/v1beta1/publishers/google/models",
                    f"https://{loc}-aiplatform.googleapis.com/v1beta1/publishers/google/models",
                ]
                for url in urls:
                    try:
                        req = urllib.request.Request(url, headers=auth_headers)
                        with self._opener().open(req, timeout=10) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                            models = []
                            for m in data.get("publisherModels", []):
                                name = m.get("name", "").split("/")[-1]
                                if name and name not in models:
                                    models.append(name)
                            if models:
                                self.log("通过 OAuth2 动态发现 %d 个 Vertex 模型: %s" % (
                                    len(models), ", ".join(models[:6]) + ("..." if len(models) > 6 else "")))
                                with self.lock:
                                    self.model_cache = models
                                    self.fetched_at = now
                                return models
                    except Exception as e:
                        self.log("动态拉取 %s 跳过: %s" % (url, e), level="debug")
        except Exception as exc:
            self.log("Vertex 动态拉取模型列表跳过: %s" % exc, level="debug")

        # 2. 从本地元数据模板中读取
        templates = self.load_models_templates()
        if templates:
            ids = []
            for t in templates:
                name = t.get("model") or t.get("name")
                if name and name not in ids:
                    ids.append(name)
            if ids:
                return ids

        return ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

    def warm_models(self):
        if not self.config.vertex_project:
            return

        def _probe():
            try:
                resp = self.request("reasoningEngines")
                resp.close()
                self.log("Vertex Reasoning Engines 探测成功")
            except Exception as e:
                self.log("Vertex 初始化探测: %s" % e, level="debug")
        threading.Thread(target=_probe, daemon=True, name="vertex-warm").start()

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
            "flash": "gemini-2.5-flash",
            "flash-lite": "gemini-2.5-flash-lite",
            "pro": "gemini-2.5-pro",
            "gemini-flash": "gemini-2.5-flash",
            "gemini-pro": "gemini-2.5-pro",
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
                                        if data.get("provider") == "vertex" or "vertex" in filename.lower():
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
            family = "gemini"
            param_size = (tmpl.get("details", {}).get("parameter_size") if tmpl else "") or "70B"

            tags_list.append({
                "name": tag_name,
                "model": tag_name,
                "modified_at": (tmpl.get("modified_at") if tmpl else None) or now_iso(),
                "size": size,
                "digest": (tmpl.get("digest") if tmpl else None) or uuid.uuid5(uuid.NAMESPACE_URL, "vertex:" + clean).hex[:12],
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

        family = "gemini"
        param_size = (tmpl.get("details", {}).get("parameter_size") if tmpl else "") or "70B"
        context_len = (tmpl.get("model_info", {}).get("%s.context_length" % family) if tmpl else None) or self.config.default_num_ctx

        return {
            "capabilities": (tmpl.get("capabilities") if tmpl else None) or ["completion", "tools", "thinking"],
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
    def _get_image_mime(data):
        try:
            raw = base64.b64decode(data.encode("ascii") if isinstance(data, str) else data)
        except Exception:
            return "image/png"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
        if raw[:2] == b"BM":
            return "image/bmp"
        return "image/png"

    @staticmethod
    def _strip_data_uri(images):
        out = []
        for img in images or []:
            value = str(img).strip()
            if "," in value and value.lstrip().lower().startswith("data:"):
                value = value.split(",", 1)[1]
            out.append(value)
        return out

    def build_payload(self, messages, options=None, tools=None, system=None):
        contents = []
        system_instruction_text = str(system) if system else None

        for msg in messages or []:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_images = self._strip_data_uri(msg.get("images") or [])

            if role == "system":
                if isinstance(content, str):
                    system_instruction_text = (system_instruction_text + "\n" + content) if system_instruction_text else content
                continue

            if role == "user":
                parts = []
                if isinstance(content, str) and content:
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append({"text": part.get("text", "")})
                            elif part.get("type") == "image_url":
                                img_url = (part.get("image_url") or {}).get("url", "")
                                if img_url.startswith("data:"):
                                    mime = img_url.split(";")[0].replace("data:", "")
                                    b64_data = img_url.split(",", 1)[1] if "," in img_url else ""
                                    parts.append({"inlineData": {"mimeType": mime, "data": b64_data}})
                        elif isinstance(part, str):
                            parts.append({"text": part})

                for img in msg_images:
                    parts.append({"inlineData": {"mimeType": self._get_image_mime(img), "data": img}})

                if not parts:
                    parts.append({"text": ""})
                contents.append({"role": "user", "parts": parts})

            elif role == "assistant":
                parts = []
                if content:
                    parts.append({"text": str(content)})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
                if not parts:
                    parts.append({"text": ""})
                contents.append({"role": "model", "parts": parts})

            elif role in ("tool", "function"):
                name = msg.get("name", "function_name")
                result_val = content
                if isinstance(result_val, str):
                    try:
                        result_val = json.loads(result_val)
                    except Exception:
                        result_val = {"result": result_val}
                elif not isinstance(result_val, dict):
                    result_val = {"result": result_val}
                contents.append({
                    "role": "function",
                    "parts": [{"functionResponse": {"name": name, "response": result_val}}]
                })

        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        gen_cfg = {}
        opts = options or {}
        if opts.get("temperature") is not None:
            gen_cfg["temperature"] = float(opts["temperature"])
        if opts.get("top_p") is not None:
            gen_cfg["topP"] = float(opts["top_p"])
        if opts.get("top_k") is not None:
            gen_cfg["topK"] = int(opts["top_k"])
        if opts.get("num_predict") is not None and int(opts["num_predict"]) > 0:
            gen_cfg["maxOutputTokens"] = int(opts["num_predict"])
        elif opts.get("max_tokens") is not None and int(opts["max_tokens"]) > 0:
            gen_cfg["maxOutputTokens"] = int(opts["max_tokens"])
        if opts.get("stop"):
            stop = opts["stop"]
            gen_cfg["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)

        payload = {"contents": contents}
        if system_instruction_text:
            payload["systemInstruction"] = {"role": "system", "parts": [{"text": system_instruction_text}]}
        if gen_cfg:
            payload["generationConfig"] = gen_cfg

        if tools:
            decls = []
            for t in tools:
                if t.get("type") == "function":
                    fn = t.get("function", {})
                    decls.append({
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    })
                elif t.get("name"):
                    decls.append(t)
            if decls:
                payload["tools"] = [{"functionDeclarations": decls}]

        return payload

    def chat(self, body):
        t0 = time.time()
        model_name = body.get("model", "")
        clean_model = self.normalize_model_name(model_name)
        endpoint = f"publishers/google/models/{clean_model}:generateContent"
        payload = self.build_payload(
            messages=body.get("messages", []),
            options=body.get("options", {}),
            tools=body.get("tools")
        )
        self.log("chat model=%s target=%s stream=false" % (model_name, clean_model))
        resp = self.request(endpoint, payload=payload)
        try:
            raw = resp.read().decode("utf-8", "replace")
        finally:
            resp.close()
        data = json.loads(raw)

        text_parts = []
        tool_calls = []
        candidates = data.get("candidates", [])
        if candidates:
            for p in candidates[0].get("content", {}).get("parts", []):
                if "text" in p:
                    text_parts.append(p["text"])
                elif "functionCall" in p:
                    tool_calls.append({
                        "function": {
                            "name": p["functionCall"].get("name", ""),
                            "arguments": p["functionCall"].get("args", {})
                        }
                    })

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        omsg = {"role": "assistant", "content": "".join(text_parts)}
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
        endpoint = f"publishers/google/models/{clean_model}:streamGenerateContent?alt=sse"
        payload = self.build_payload(
            messages=body.get("messages", []),
            options=body.get("options", {}),
            tools=body.get("tools")
        )
        self.log("chat model=%s target=%s stream=true" % (model_name, clean_model))
        resp = self.request(endpoint, payload=payload, stream=True)

        tool_calls = []
        prompt_tokens = 0
        completion_tokens = 0
        final_reason = "stop"

        try:
            for event_type, data in iter_sse_events(resp):
                if event_type == "done":
                    break
                if not isinstance(data, dict):
                    continue
                candidates = data.get("candidates", [])
                if candidates:
                    for p in candidates[0].get("content", {}).get("parts", []):
                        if "text" in p and p["text"]:
                            write(ndjson({
                                "model": model_name,
                                "created_at": now_iso(),
                                "message": {"role": "assistant", "content": p["text"]},
                                "done": False,
                            }))
                        elif "functionCall" in p:
                            tool_calls.append({
                                "function": {
                                    "name": p["functionCall"].get("name", ""),
                                    "arguments": p["functionCall"].get("args", {})
                                }
                            })
                if "usageMetadata" in data:
                    prompt_tokens = data["usageMetadata"].get("promptTokenCount", prompt_tokens)
                    completion_tokens = data["usageMetadata"].get("candidatesTokenCount", completion_tokens)
        finally:
            resp.close()

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

        endpoint = f"publishers/google/models/{clean_model}:generateContent"
        payload = self.build_payload(
            messages=messages,
            options=body.get("options", {}),
            system=system
        )
        self.log("generate model=%s target=%s stream=false" % (model_name, clean_model))
        resp = self.request(endpoint, payload=payload)
        try:
            raw = resp.read().decode("utf-8", "replace")
        finally:
            resp.close()
        data = json.loads(raw)
        text_parts = []
        candidates = data.get("candidates", [])
        if candidates:
            for p in candidates[0].get("content", {}).get("parts", []):
                if "text" in p:
                    text_parts.append(p["text"])
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

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

        endpoint = f"publishers/google/models/{clean_model}:streamGenerateContent?alt=sse"
        payload = self.build_payload(
            messages=messages,
            options=body.get("options", {}),
            system=system
        )
        self.log("generate model=%s target=%s stream=true" % (model_name, clean_model))
        resp = self.request(endpoint, payload=payload, stream=True)

        prompt_tokens = 0
        completion_tokens = 0

        try:
            for event_type, data in iter_sse_events(resp):
                if event_type == "done":
                    break
                if not isinstance(data, dict):
                    continue
                candidates = data.get("candidates", [])
                if candidates:
                    for p in candidates[0].get("content", {}).get("parts", []):
                        if "text" in p and p["text"]:
                            write(ndjson({
                                "model": model_name,
                                "created_at": now_iso(),
                                "response": p["text"],
                                "done": False,
                            }))
                if "usageMetadata" in data:
                    prompt_tokens = data["usageMetadata"].get("promptTokenCount", prompt_tokens)
                    completion_tokens = data["usageMetadata"].get("candidatesTokenCount", completion_tokens)
        finally:
            resp.close()

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

        endpoint = f"publishers/google/models/{clean_model}:streamGenerateContent?alt=sse"
        payload = self.build_payload(
            messages=body.get("messages", []),
            options=options,
            tools=body.get("tools")
        )
        resp = self.request(endpoint, payload=payload, stream=True)
        try:
            for event_type, data in iter_sse_events(resp):
                if event_type == "done":
                    break
                if not isinstance(data, dict):
                    continue
                candidates = data.get("candidates", [])
                if candidates:
                    for p in candidates[0].get("content", {}).get("parts", []):
                        if "text" in p and p["text"]:
                            chunk = {
                                "id": base_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": p["text"]}, "finish_reason": None}],
                            }
                            yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8")
                        elif "functionCall" in p:
                            fn = p["functionCall"]
                            args = fn.get("args", {})
                            args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                            tc_chunk = {
                                "id": base_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {"tool_calls": [{
                                    "index": len(tool_calls),
                                    "id": f"call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {"name": fn.get("name", ""), "arguments": args_str},
                                }]}, "finish_reason": None}],
                            }
                            tool_calls.append(fn)
                            finish_reason = "tool_calls"
                            yield ("data: " + json.dumps(tc_chunk, ensure_ascii=False) + "\n\n").encode("utf-8")
                if "usageMetadata" in data:
                    prompt_tokens = data["usageMetadata"].get("promptTokenCount", prompt_tokens)
                    completion_tokens = data["usageMetadata"].get("candidatesTokenCount", completion_tokens)
        finally:
            resp.close()

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
    server_version = "vertex-ollama-proxy/" + __version__

    @property
    def client(self):
        return self.server.client

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
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text, status=200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def start_stream(self, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def write_chunk(self, data):
        self.wfile.write(("%X\r\n" % len(data)).encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

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
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-goog-api-key")
        self.send_header("Content-Length", "0")
        self.end_headers()

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
                    self.client.chat_stream(body, self.write_chunk)
                    self.end_stream()
                else:
                    self.send_json(self.client.chat(body))
            elif path == "/api/generate":
                if body.get("stream"):
                    self.start_stream("application/x-ndjson")
                    self.client.generate_stream(body, self.write_chunk)
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
                    for chunk in self.client.v1_chat(body, stream=True):
                        self.write_chunk(chunk)
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
        except (BrokenPipeError, ConnectionResetError):
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
    parser = argparse.ArgumentParser(description="Google Cloud Vertex AI Agent Platform -> Ollama API 代理")
    parser.add_argument("--api-key", default=None, help="Vertex AI API Key (也可通过 config.json 或环境变量 VERTEX_API_KEY 配置)")
    parser.add_argument("--bearer-token", default=None, help="GCP OAuth2 Bearer Token (也可通过 config.json 或环境变量 VERTEX_BEARER_TOKEN 配置)")
    parser.add_argument("--vertex-project", default=None, help="Vertex AI Project Number/ID (也可通过 config.json 或环境变量 VERTEX_PROJECT 配置)")
    parser.add_argument("--vertex-location", default=None, help="Vertex AI Location/Region (默认 us-central1)")
    parser.add_argument("--default-model", default=None, help="默认模型 (默认 gemini-2.5-flash)")
    parser.add_argument("--config", default=None, help="配置文件路径 (默认查找 config.json)")
    parser.add_argument("--host", default=None, help="监听地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="监听端口 (默认 11434)")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    parser.add_argument("--version", action="version", version="vertex-ollama-proxy %s" % __version__)
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
    if args.bearer_token:
        config.bearer_token = args.bearer_token
    if args.vertex_project:
        config.vertex_project = args.vertex_project
    if args.vertex_location:
        config.vertex_location = args.vertex_location
    if args.default_model:
        config.default_model = args.default_model
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.verbose:
        config.log_level = "debug"

    client = VertexClient(config)
    client.warm_models()

    server = ProxyServer((config.host, config.port), Handler, client)

    print("vertex-ollama-proxy 已启动: http://%s:%d" % (config.host, config.port))
    proj_display = config.vertex_project or "(未设置，请在 config.json 中配置)"
    print("  - Vertex AI: %s (Project: %s, Region: %s)" % (
        client.base_url(), proj_display, config.vertex_location))
    print("  - 默认模型: %s" % config.default_model)
    auth_info = "OAuth2 Bearer Token" if config.bearer_token else (
        ("API Key: " + config.api_key[:6] + "..." + config.api_key[-4:]) if len(config.api_key) > 10 else (config.api_key or "ADC 自动发现")
    )
    print("  - 鉴权模式: %s" % auth_info)
    print("日志级别: %s (--verbose 可开启 debug)" % config.log_level)
    print("按 Ctrl+C 停止")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
