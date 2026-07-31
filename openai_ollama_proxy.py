#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
openai-ollama-proxy
===================

把 OpenAI 兼容 API(DeepSeek / 智谱 BigModel / Kimi 等)暴露为 Ollama API,
同时透传 OpenAI 兼容接口 /v1/chat/completions,供 VS Code Copilot 本地模型、
Continue、Cline 等工具使用。仅依赖 Python 标准库。

端点
----
GET  /api/tags             汇总各 provider 的模型列表,优先匹配 models/*.json,未命中自动生成
POST /api/show             优先读取 models/*.json,未命中自动生成应答
POST /api/chat             转换为 OpenAI /v1/chat/completions 转发(支持流式)
POST /api/generate         转换为 OpenAI /v1/chat/completions 转发(支持流式)
GET  /api/version          返回模拟的 Ollama 版本号
GET  /v1/models            OpenAI 兼容模型列表
POST /v1/chat/completions  透传(支持流式)

用法
----
    python openai_ollama_proxy.py --config config.json
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
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
DEFAULT_TEMPLATE = "{{ .System }}\n{{ .Prompt }}"
OLLAMA_VERSION = "0.5.4"


def now_iso():
    """Ollama 风格 RFC3339 时间戳, 如 2026-07-31T12:00:00.123Z"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def ndjson(obj):
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def iter_sse(resp):
    """逐条解析 upstream 的 SSE 事件, 产出 'data: ' 之后的内容。"""
    buf = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("data:"):
            buf.append(line[len("data:"):].strip())
        elif line == "" and buf:
            yield "\n".join(buf)
            buf = []
    if buf:
        yield "\n".join(buf)


class ModelNotFoundError(Exception):
    pass


class UpstreamError(Exception):
    def __init__(self, status, body):
        super().__init__("upstream http %s: %s" % (status, body))
        self.status = status
        self.body = body


class Provider:
    """单个 OpenAI 兼容上游。"""

    def __init__(self, name, base_url, api_key, models, family):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.models = [str(m).strip() for m in (models or []) if str(m).strip()]
        self.family = family or name
        self.chat_url = self.base_url + "/chat/completions"
        self.models_url = self.base_url + "/models"


class Config:
    def __init__(self, data, base_dir):
        self.host = str(data.get("host", "127.0.0.1"))
        self.port = int(data.get("port", 11434))
        self.timeout = float(data.get("timeout", 300))
        self.cache_ttl = float(data.get("cache_ttl", 60))
        self.default_num_ctx = int(data.get("default_num_ctx", 4096))
        self.models_dir = os.path.abspath(
            os.path.join(base_dir, str(data.get("models_dir", "models"))))
        self.use_env_proxy = bool(data.get("use_env_proxy", True))
        self.log_level = str(data.get("log_level", "info")).lower()
        if self.log_level not in ("quiet", "info", "debug"):
            self.log_level = "info"
        self.mapping = data.get("mapping", {}) or {}
        raw = data.get("providers", [])
        if not raw:
            raise ValueError("配置文件缺少 providers 列表")
        self.providers = []
        for p in raw:
            name = str(p.get("name", "")).strip()
            base_url = str(p.get("base_url", "")).strip()
            if not name or not base_url:
                raise ValueError("每个 provider 都必须包含 name 和 base_url")
            self.providers.append(Provider(
                name=name,
                base_url=base_url,
                api_key=p.get("api_key"),
                models=p.get("models"),
                family=p.get("family"),
            ))


class Proxy:
    def __init__(self, config):
        self.config = config
        self.log_level = config.log_level
        self.lock = threading.Lock()
        self.model_owner = {}   # upstream model id(lower) -> provider name
        self.fetched_at = {}    # provider name -> 上次 /models 时间
        self.fetched_ids = {}   # provider name -> [model ids]
        self.models_files = None
        self._init_owner_from_config()

    # ---------------- 通用 ----------------
    def log(self, msg, level="info"):
        levels = {"quiet": 0, "info": 1, "debug": 2}
        want = levels.get(self.log_level, 1)
        if want < levels.get(level, 1):
            return
        stream = sys.stderr if level == "error" else sys.stdout
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), file=stream, flush=True)

    def _opener(self):
        if self.config.use_env_proxy:
            return urllib.request.build_opener()
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _init_owner_from_config(self):
        for p in self.config.providers:
            for mid in p.models:
                self.model_owner[mid.lower()] = p.name

    def get_provider(self, name):
        for p in self.config.providers:
            if p.name.lower() == str(name).lower():
                return p
        return None

    def upstream_request(self, provider, url, payload=None):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if provider.api_key:
            headers["Authorization"] = "Bearer " + provider.api_key
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url, data=data, headers=headers,
            method="POST" if payload is not None else "GET")
        try:
            return self._opener().open(req, timeout=self.config.timeout)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", "replace")
            raise UpstreamError(exc.code, err_body) from exc
        except urllib.error.URLError as exc:
            raise UpstreamError(0, str(exc.reason)) from exc

    # ---------------- 模型列表 ----------------
    def fetch_provider_models(self, provider, force=False):
        now = time.time()
        with self.lock:
            cached = self.fetched_ids.get(provider.name)
            cached_at = self.fetched_at.get(provider.name, 0)
        if not force and cached is not None and (now - cached_at) < self.config.cache_ttl:
            self.log("[%s] /models 使用缓存 %d 个模型" % (provider.name, len(cached)), level="debug")
            return list(cached)

        if provider.models:
            ids = list(provider.models)
        else:
            try:
                resp = self.upstream_request(provider, provider.models_url)
                try:
                    raw = resp.read().decode("utf-8", "replace")
                finally:
                    resp.close()
                data = json.loads(raw)
                ids = [str(m.get("id", "")).strip() for m in data.get("data", [])
                       if str(m.get("id", "")).strip()]
                self.log("[%s] /models 获取到 %d 个模型" % (provider.name, len(ids)))
            except Exception as exc:
                self.log("[%s] /models 获取失败: %s" % (provider.name, exc), level="error")
                return list(provider.models)

        with self.lock:
            self.fetched_ids[provider.name] = list(ids)
            self.fetched_at[provider.name] = time.time()
            for mid in ids:
                self.model_owner[mid.lower()] = provider.name
        return ids

    def resolve_model(self, name):
        """把 Ollama 侧模型名映射为 (provider, upstream_model_id)。"""
        name = (name or "").strip()
        if not name:
            return None, None
        base = name.split(":", 1)[0] if ":" in name else name

        # 1) 显式 mapping
        mp = self.config.mapping.get(name) or self.config.mapping.get(base)
        if mp:
            provider = self.get_provider(mp.get("provider") or mp.get("name"))
            if provider:
                return provider, str(mp.get("model") or base)

        # 2) 运行时注册表(由 /models 或配置 models 列表填充)
        with self.lock:
            owner = self.model_owner.get(name.lower()) or self.model_owner.get(base.lower())
        if owner:
            provider = self.get_provider(owner)
            if provider:
                return provider, base

        # 3) 配置的 models 列表
        for p in self.config.providers:
            for mid in p.models:
                if mid == name or mid == base or mid + ":latest" == name:
                    return p, mid

        # 4) 尚未拉取过的 provider 尝试动态获取一次
        for p in self.config.providers:
            if p.name not in self.fetched_ids:
                try:
                    self.fetch_provider_models(p)
                except Exception:
                    pass
        with self.lock:
            owner = self.model_owner.get(name.lower()) or self.model_owner.get(base.lower())
        if owner:
            provider = self.get_provider(owner)
            if provider:
                return provider, base

        # 5) 无法路由到任何 provider
        return None, None

    # ---------------- models/*.json 匹配 ----------------
    def list_models_files(self):
        if self.models_files is None:
            files = []
            if os.path.isdir(self.config.models_dir):
                for fn in sorted(os.listdir(self.config.models_dir)):
                    if fn.lower().endswith(".json"):
                        files.append(os.path.join(self.config.models_dir, fn))
            self.models_files = files
        return self.models_files

    def load_models_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def find_models_entry(self, name):
        """按 Ollama 模型名匹配 models 目录 JSON, 返回 (数据, 路径)。"""
        base = name.split(":", 1)[0] if ":" in name else name
        name_l, base_l = name.lower(), base.lower()
        for path in self.list_models_files():
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.lower() in (name_l, base_l):
                data = self.load_models_file(path)
                if data:
                    return data, path
        for path in self.list_models_files():
            data = self.load_models_file(path)
            if not data:
                continue
            tag = data.get("tag") or data.get("tag_model") or {}
            for key in ("name", "model"):
                val = str(tag.get(key, "")).lower()
                if val and val in (name_l, base_l):
                    return data, path
        return None, None

    # ---------------- /api/tags 与 /api/show 应答 ----------------
    def tags(self):
        entries, seen = [], set()
        for provider in self.config.providers:
            for mid in self.fetch_provider_models(provider):
                tag = mid if ":" in mid else mid + ":latest"
                key = tag.lower()
                if key in seen:
                    continue
                seen.add(key)
                entries.append(self.tags_entry_for(tag, mid, provider))
        self.log("tags 返回 %d 个模型(provider=%d)" % (len(entries), len(self.config.providers)))
        return {"models": entries}


    def tags_entry_for(self, ollama_name, upstream_id, provider):
        data, _path = self.find_models_entry(ollama_name)
        if data:
            tag = data.get("tag") or data.get("tag_model")
            if isinstance(tag, dict):
                return tag
    def generate_tag_entry(self, ollama_name, provider):
        return {
            "name": ollama_name,
            "model": ollama_name,
            "modified_at": now_iso(),
            "size": 0,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": provider.family,
                "families": [provider.family],
                "parameter_size": "unknown",
                "quantization_level": "unknown",
            },
        }

    def show_for(self, name):
        data, path = self.find_models_entry(name)
        if data and isinstance(data.get("show"), dict):
            show = dict(data["show"])
            show.setdefault("license", "")
            show.setdefault("modelfile", "")
            self.log("show model=%s source=file(%s)" % (name, os.path.basename(path)))
            return show
        provider, _upstream = self.resolve_model(name)
        if provider is None:
            provider = self.config.providers[0]
        self.log("show model=%s source=auto provider=%s" % (name, provider.name))
        return self.generate_show(name, provider)


    def generate_show(self, ollama_name, provider):
        return {
            "license": "",
            "modelfile": "",
            "parameters": "temperature 0.7\nnum_ctx %d" % self.config.default_num_ctx,
            "template": DEFAULT_TEMPLATE,
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": provider.family,
                "families": [provider.family],
                "parameter_size": "unknown",
                "quantization_level": "unknown",
            },
            "model_info": {
                "general.architecture": provider.family,
                "general.parameter_count": 0,
                "%s.context_length" % provider.family: self.config.default_num_ctx,
            },
            "capabilities": ["completion", "tools"],
        }

    # ---------------- 消息/参数转换 ----------------
    @staticmethod
    def convert_messages(messages):
        out = []
        for msg in messages or []:
            m = dict(msg)
            content = m.get("content")
            if content is None:
                m["content"] = ""
            elif not isinstance(content, list):
                m["content"] = str(content)
            if m.get("tool_calls"):
                m["tool_calls"] = Proxy.ollama_tool_calls_to_openai(m["tool_calls"])
            out.append(m)
        return out

    @staticmethod
    def ollama_tool_calls_to_openai(tool_calls):
        out = []
        for i, tc in enumerate(tool_calls or []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            out.append({
                "id": "call_%d" % i,
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args},
            })
        return out

    @staticmethod
    def openai_tool_calls_to_ollama(tool_calls):
        out = []
        for tc in tool_calls or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
        return out

    def apply_options(self, params, opts):
        key_map = {
            "temperature": "temperature",
            "num_predict": "max_tokens",
            "top_p": "top_p",
            "stop": "stop",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
        }
        for okey, akey in key_map.items():
            val = opts.get(okey)
            if val is None:
                continue
            if okey == "num_predict" and int(val) <= 0:
                continue
            if okey == "stop" and isinstance(val, str):
                val = [val]
            params[akey] = val

    # ---------------- /api/chat ----------------
    def build_chat_payload(self, body, upstream_model):
        params = {
            "model": upstream_model,
            "messages": self.convert_messages(body.get("messages", [])),
            "stream": bool(body.get("stream", False)),
        }
        self.apply_options(params, body.get("options", {}) or {})
        fmt = body.get("format")
        if fmt in ("json", "json_object"):
            params["response_format"] = {"type": "json_object"}
        if body.get("tools"):
            params["tools"] = body["tools"]
        if body.get("tool_choice") is not None:
            params["tool_choice"] = body["tool_choice"]
        return params

    def chat(self, body):
        t0 = time.time()
        ollama_name = body.get("model", "")
        provider, upstream = self.resolve_model(ollama_name)
        if provider is None:
            raise ModelNotFoundError("model '%s' not found, try pulling it first" % ollama_name)
        url = provider.chat_url
        self.log("chat model=%s provider=%s url=%s stream=false" % (ollama_name, provider.name, url))
        payload = self.build_chat_payload(body, upstream)
        try:
            resp = self.upstream_request(provider, provider.chat_url, payload)
        except UpstreamError as exc:
            self.log("chat 失败 model=%s provider=%s status=%s body=%s" % (
                ollama_name, provider.name, exc.status, (exc.body or "")[:200]), level="error")
            raise
        try:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        finally:
            resp.close()
        out = self.openai_chat_to_ollama(data, ollama_name)
        self.log("chat 完成 model=%s provider=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            ollama_name, provider.name,
            out.get("prompt_eval_count", 0), out.get("eval_count", 0),
            time.time() - t0))
        return out


    def chat_stream(self, body, write):
        t0 = time.time()
        ollama_name = body.get("model", "")
        provider, upstream = self.resolve_model(ollama_name)
        if provider is None:
            raise ModelNotFoundError("model '%s' not found, try pulling it first" % ollama_name)
        url = provider.chat_url
        self.log("chat model=%s provider=%s url=%s stream=true" % (ollama_name, provider.name, url))
        payload = self.build_chat_payload(body, upstream)
        try:
            resp = self.upstream_request(provider, provider.chat_url, payload)
        except UpstreamError as exc:
            self.log("chat 失败 model=%s provider=%s status=%s body=%s" % (
                ollama_name, provider.name, exc.status, (exc.body or "")[:200]), level="error")
            raise
        full_content, tool_calls, usage = [], {}, {}
        final_reason = "stop"
        try:
            for payload_text in iter_sse(resp):
                if not payload_text.strip():
                    continue
                try:
                    chunk = json.loads(payload_text)
                except Exception:
                    continue
                if chunk.get("error"):
                    write(ndjson({"error": chunk["error"], "done": False}))
                    return
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {}) or {}
                content = delta.get("content")
                if content:
                    full_content.append(content)
                    write(ndjson({
                        "model": ollama_name,
                        "created_at": now_iso(),
                        "message": {"role": "assistant", "content": content},
                        "done": False,
                    }))
                for tcd in delta.get("tool_calls") or []:
                    idx = tcd.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"name": "", "arguments": ""})
                    fn = tcd.get("function", {}) or {}
                    slot["name"] += fn.get("name", "") or ""
                    slot["arguments"] += fn.get("arguments", "") or ""
                if choice.get("finish_reason"):
                    final_reason = choice["finish_reason"]
        except Exception as exc:
            self.log("chat 流式中断 model=%s error=%s" % (ollama_name, exc), level="debug")
            raise
        finally:
            resp.close()

        final_msg = {"role": "assistant", "content": ""}
        if tool_calls:
            calls = []
            for idx in sorted(tool_calls):
                slot = tool_calls[idx]
                try:
                    args = json.loads(slot["arguments"]) if slot["arguments"] else {}
                except Exception:
                    args = {}
                calls.append({"function": {"name": slot["name"], "arguments": args}})
            final_msg["tool_calls"] = calls
            final_reason = "tool_calls"
        write(ndjson({
            "model": ollama_name,
            "created_at": now_iso(),
            "message": final_msg,
            "done": True,
            "done_reason": final_reason,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }))
        self.log("chat 完成 model=%s provider=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            ollama_name, provider.name,
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            time.time() - t0))


    @staticmethod
    def openai_chat_to_ollama(openai_resp, ollama_name):
        usage = openai_resp.get("usage", {}) or {}
        choices = openai_resp.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message", {}) or {}
        content = message.get("content")
        if content is None:
            content = ""
        omsg = {"role": message.get("role", "assistant"), "content": content}
        done_reason = choice.get("finish_reason") or "stop"
        if message.get("tool_calls"):
            omsg["tool_calls"] = Proxy.openai_tool_calls_to_ollama(message["tool_calls"])
            done_reason = "tool_calls"
        return {
            "model": ollama_name,
            "created_at": now_iso(),
            "message": omsg,
            "done": True,
            "done_reason": done_reason,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

    # ---------------- /api/generate ----------------
    def build_generate_payload(self, body, upstream_model):
        prompt = body.get("prompt", "") or ""
        messages = []
        if body.get("system"):
            messages.append({"role": "system", "content": str(body["system"])})
        images = body.get("images") or []
        if images:
            content = [{"type": "text", "text": str(prompt)}]
            for img in images:
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + img}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": str(prompt)})
        params = {
            "model": upstream_model,
            "messages": messages,
            "stream": bool(body.get("stream", False)),
        }
        self.apply_options(params, body.get("options", {}) or {})
        return params

    def generate(self, body):
        t0 = time.time()
        ollama_name = body.get("model", "")
        provider, upstream = self.resolve_model(ollama_name)
        if provider is None:
            raise ModelNotFoundError("model '%s' not found, try pulling it first" % ollama_name)
        url = provider.chat_url
        prompt = body.get("prompt", "") or ""
        self.log("generate model=%s provider=%s url=%s prompt_len=%d stream=false" % (
            ollama_name, provider.name, url, len(prompt)))
        payload = self.build_generate_payload(body, upstream)
        try:
            resp = self.upstream_request(provider, provider.chat_url, payload)
        except UpstreamError as exc:
            self.log("generate 失败 model=%s provider=%s status=%s body=%s" % (
                ollama_name, provider.name, exc.status, (exc.body or "")[:200]), level="error")
            raise
        try:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        finally:
            resp.close()
        out = self.openai_chat_to_generate(data, ollama_name)
        self.log("generate 完成 model=%s provider=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            ollama_name, provider.name,
            out.get("prompt_eval_count", 0), out.get("eval_count", 0),
            time.time() - t0))
        return out


    def generate_stream(self, body, write):
        t0 = time.time()
        ollama_name = body.get("model", "")
        provider, upstream = self.resolve_model(ollama_name)
        if provider is None:
            raise ModelNotFoundError("model '%s' not found, try pulling it first" % ollama_name)
        url = provider.chat_url
        prompt = body.get("prompt", "") or ""
        self.log("generate model=%s provider=%s url=%s prompt_len=%d stream=true" % (
            ollama_name, provider.name, url, len(prompt)))
        payload = self.build_generate_payload(body, upstream)
        try:
            resp = self.upstream_request(provider, provider.chat_url, payload)
        except UpstreamError as exc:
            self.log("generate 失败 model=%s provider=%s status=%s body=%s" % (
                ollama_name, provider.name, exc.status, (exc.body or "")[:200]), level="error")
            raise
        usage, final_reason = {}, "stop"
        try:
            for payload_text in iter_sse(resp):
                if not payload_text.strip():
                    continue
                try:
                    chunk = json.loads(payload_text)
                except Exception:
                    continue
                if chunk.get("error"):
                    write(ndjson({"error": chunk["error"], "done": False}))
                    return
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                content = (choice.get("delta", {}) or {}).get("content")
                if content:
                    write(ndjson({
                        "model": ollama_name,
                        "created_at": now_iso(),
                        "response": content,
                        "done": False,
                    }))
                if choice.get("finish_reason"):
                    final_reason = choice["finish_reason"]
        except Exception as exc:
            self.log("generate 流式中断 model=%s error=%s" % (ollama_name, exc), level="debug")
            raise
        finally:
            resp.close()
        write(ndjson({
            "model": ollama_name,
            "created_at": now_iso(),
            "response": "",
            "done": True,
            "done_reason": final_reason,
            "context": [],
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }))
        self.log("generate 完成 model=%s provider=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            ollama_name, provider.name,
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            time.time() - t0))


    @staticmethod
    def openai_chat_to_generate(openai_resp, ollama_name):
        usage = openai_resp.get("usage", {}) or {}
        choices = openai_resp.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message", {}) or {}
        return {
            "model": ollama_name,
            "created_at": now_iso(),
            "response": message.get("content") or "",
            "done": True,
            "done_reason": choice.get("finish_reason") or "stop",
            "context": [],
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

    # ---------------- /v1/* 透传 ----------------
    def v1_models(self):
        data, seen = [], set()
        for provider in self.config.providers:
            for mid in self.fetch_provider_models(provider):
                tag = mid if ":" in mid else mid + ":latest"
                if tag.lower() in seen:
                    continue
                seen.add(tag.lower())
                data.append({
                    "id": tag,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider.name,
                })
        return {"object": "list", "data": data}

    def v1_chat(self, body, stream=False):
        t0 = time.time()
        ollama_name = body.get("model", "")
        provider, upstream = self.resolve_model(ollama_name)
        if provider is None:
            raise ModelNotFoundError("model '%s' not found" % ollama_name)
        url = provider.chat_url
        self.log("v1/chat model=%s provider=%s url=%s stream=%s" % (
            ollama_name, provider.name, url, "true" if stream else "false"))
        new_body = dict(body)
        new_body["model"] = upstream
        try:
            resp = self.upstream_request(provider, provider.chat_url, new_body)
        except UpstreamError as exc:
            self.log("v1/chat 失败 model=%s provider=%s status=%s body=%s" % (
                ollama_name, provider.name, exc.status, (exc.body or "")[:200]), level="error")
            raise
        if stream:
            return resp
        try:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        finally:
            resp.close()
        data["model"] = ollama_name
        usage = data.get("usage", {}) or {}
        self.log("v1/chat 完成 model=%s provider=%s prompt_tokens=%d completion_tokens=%d elapsed=%.2fs" % (
            ollama_name, provider.name,
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            time.time() - t0))
        return data


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "OllamaProxy/1.0"
    protocol_version = "HTTP/1.1"

    @property

    def proxy(self):
        return self.server.proxy

    def log_message(self, fmt, *args):
        proxy = getattr(self.server, "proxy", None)
        if not proxy:
            return
        elapsed = ""
        start = getattr(self, "_req_start", None)
        if start:
            elapsed = " %.2fs" % (time.time() - start)
        proxy.log("%s %s%s" % (self.client_address[0], fmt % args, elapsed), level="info")

    # ---------------- 基础 IO ----------------
    def read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length > 0:
            return self.rfile.read(length)
        return self.rfile.read()

    def json_body(self):
        raw = self.read_body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
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

    # ---------------- 路由 ----------------
    def do_OPTIONS(self):
        self._req_start = time.time()
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._req_start = time.time()
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/":
                self.send_text("Ollama is running")
            elif path == "/api/version":
                self.send_json({"version": OLLAMA_VERSION})
            elif path == "/api/tags":
                self.send_json(self.proxy.tags())
            elif path == "/api/ps":
                self.send_json({"models": []})
            elif path == "/v1/models":
                self.send_json(self.proxy.v1_models())
            else:
                self.send_error_json("not found: " + path, 404)
        except UpstreamError as exc:
            self.send_error_json("upstream error: %s" % (exc.body or exc), 502)
        except Exception as exc:
            self.proxy.log("GET %s 失败: %s" % (path, exc), level="error")
            try:
                self.send_error_json("internal error: %s" % exc, 500)
            except Exception:
                pass

    def do_POST(self):
        self._req_start = time.time()
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self.json_body()
            if path == "/api/chat":
                if body.get("stream"):
                    self.start_stream("application/x-ndjson")
                    self.proxy.chat_stream(body, self.write_chunk)
                    self.end_stream()
                else:
                    self.send_json(self.proxy.chat(body))
            elif path == "/api/generate":
                if body.get("stream"):
                    self.start_stream("application/x-ndjson")
                    self.proxy.generate_stream(body, self.write_chunk)
                    self.end_stream()
                else:
                    self.send_json(self.proxy.generate(body))
            elif path == "/api/show":
                name = body.get("name") or body.get("model")
                if not name:
                    raise ValueError("请求缺少模型 name")
                self.send_json(self.proxy.show_for(name))
            elif path == "/v1/chat/completions":
                if body.get("stream"):
                    resp = self.proxy.v1_chat(body, stream=True)
                    self.start_stream("text/event-stream")
                    try:
                        for raw in resp:
                            if raw:
                                self.write_chunk(raw)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        try:
                            resp.close()
                        except Exception:
                            pass
                        self.end_stream()
                else:
                    self.send_json(self.proxy.v1_chat(body, stream=False))
            else:
                self.send_error_json("not found: " + path, 404)
        except ModelNotFoundError as exc:
            self.send_error_json(str(exc), 404)
        except UpstreamError as exc:
            self.send_error_json("upstream error: %s" % (exc.body or exc), 502)
        except ValueError as exc:
            self.send_error_json(str(exc), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.proxy.log("POST %s 失败: %s" % (path, exc), level="error")
            try:
                self.send_error_json("internal error: %s" % exc, 500)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="OpenAI 兼容 API -> Ollama API 代理")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径(默认 config.json)")
    parser.add_argument("--host", default=None, help="覆盖监听地址")
    parser.add_argument("--port", type=int, default=None, help="覆盖监听端口")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)
    if not os.path.exists(config_path):
        print("[error] 配置文件不存在: %s" % config_path, file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print("[error] 读取配置失败: %s" % exc, file=sys.stderr)
        sys.exit(1)

    try:
        config = Config(data, os.path.dirname(config_path))
    except ValueError as exc:
        print("[error] 配置错误: %s" % exc, file=sys.stderr)
        sys.exit(1)

    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    proxy = Proxy(config)
    if args.verbose:
        proxy.log_level = "debug"

    server = ProxyServer((config.host, config.port), Handler)
    server.proxy = proxy

    print("openai-ollama-proxy 已启动: http://%s:%d" % (config.host, config.port))
    for p in config.providers:
        print("  - %-12s %s  models=%s" % (
            p.name, p.base_url, p.models if p.models else "(动态获取)"))
    print("日志级别: %s (--verbose 可开启 debug)" % proxy.log_level)
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
