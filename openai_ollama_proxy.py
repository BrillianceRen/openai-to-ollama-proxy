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
import base64
import json
import os
import sys
import threading
import time
import uuid
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
__version__ = "1.1.0"


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
            payload = line[len("data:"):].strip()
            if payload.lower() == "[done]":
                if buf:
                    yield "\n".join(buf)
                return
            buf.append(payload)
        elif line == "" and buf:
            yield "\n".join(buf)
            buf = []
    if buf:
        yield "\n".join(buf)


def iter_sse_events(resp):
    """逐条解析 Responses API 的 SSE 事件, 产出 (event_type, data_dict) 元组。"""
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
            try:
                data = json.loads(payload)
            except Exception:
                data = {"raw": payload}
            yield (event_type or data.get("type", ""), data)
            event_type = None
            data_lines = []
    if data_lines:
        payload = "\n".join(data_lines)
        try:
            data = json.loads(payload)
        except Exception:
            data = {"raw": payload}
        yield (event_type or data.get("type", ""), data)


import re

DSML_OPEN = "<\uff5c\uff5cDSML\uff5c\uff5ctool_calls>"
DSML_CLOSE = "</\uff5c\uff5cDSML\uff5c\uff5ctool_calls>"
DSML_INVOKE_RE = re.compile(
    r'<\uff5c\uff5cDSML\uff5c\uff5cinvoke\s+name="([^"]*)"\s*>'
    r'(.*?)'
    r'</\uff5c\uff5cDSML\uff5c\uff5cinvoke>',
    re.DOTALL,
)
DSML_PARAM_RE = re.compile(
    r'<\uff5c\uff5cDSML\uff5c\uff5cparameter\s+name="([^"]*)"[^>]*>'
    r'([^<]*)'
    r'</\uff5c\uff5cDSML\uff5c\uff5cparameter>',
    re.DOTALL,
)

NOOP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "No-op",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def parse_dsml_content(content):
    """从文本中提取 DeepSeek DSML 工具调用标记。

    返回 (clean_text, tool_calls_list)。clean_text 为去除 DSML 后的纯文本;
    tool_calls_list 为标准 OpenAI tool_calls 格式(可能为空)。
    """
    if not content or DSML_OPEN not in content:
        return content, []
    tool_calls = []
    tc_index = 0

    def _replace_invoke(m):
        nonlocal tc_index
        fn_name = m.group(1)
        body = m.group(2)
        args = {}
        for pm in DSML_PARAM_RE.finditer(body):
            args[pm.group(1)] = pm.group(2).strip()
        import json as _json
        tool_calls.append({
            "id": f"call_dsml_{tc_index}",
            "type": "function",
            "index": tc_index,
            "function": {"name": fn_name, "arguments": _json.dumps(args, ensure_ascii=False)},
        })
        tc_index += 1
        return ""

    # 先替换整个 tool_calls 块内的 invoke
    def _replace_block(m):
        return m.group(0)  # placeholder, handled below

    cleaned = content
    for m in DSML_INVOKE_RE.finditer(content):
        _replace_invoke(m)

    # 移除整个 DSML 块
    start = cleaned.find(DSML_OPEN)
    while start >= 0:
        end = cleaned.find(DSML_CLOSE, start)
        if end < 0:
            cleaned = cleaned[:start]
            break
        cleaned = cleaned[:start] + cleaned[end + len(DSML_CLOSE):]
        start = cleaned.find(DSML_OPEN)

    # 清理多余空白
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned, tool_calls


class ModelNotFoundError(Exception):
    pass


class BodyTooLargeError(Exception):
    pass


class UpstreamError(Exception):
    def __init__(self, status, body):
        super().__init__("upstream http %s: %s" % (status, body))
        self.status = status
        self.body = body


class Provider:
    """单个 OpenAI 兼容上游。"""

    def __init__(self, name, base_url, api_key, models, family,
                 headers=None, require_tools=False, enabled=True):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.models = [str(m).strip() for m in (models or []) if str(m).strip()]
        self.family = family or name
        self.enabled = bool(enabled)
        self.require_tools = bool(require_tools)
        self.headers = {
            str(key).strip(): str(value)
            for key, value in (headers or {}).items()
            if str(key).strip() and value is not None
        }
        self.chat_url = self.base_url + "/chat/completions"
        self.responses_url = self.base_url + "/responses"
        self.models_url = self.base_url + "/models"


class Config:
    def __init__(self, data, base_dir):
        self.host = str(data.get("host", "127.0.0.1"))
        self.port = int(data.get("port", 11434))
        self.timeout = float(data.get("timeout", 300))
        self.cache_ttl = float(data.get("cache_ttl", 60))
        self.fetch_wait_timeout = float(data.get("fetch_wait_timeout", 30))
        self.max_body_bytes = int(data.get("max_body_bytes", 64 * 1024 * 1024))
        self.retry_without_tools = bool(data.get("retry_without_tools", True))
        self.strip_tools = bool(data.get("strip_tools", False))
        self.stream_mode = str(data.get("stream_mode", "auto")).lower()
        if self.stream_mode not in ("auto", "stream", "non_stream"):
            self.stream_mode = "auto"
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
            if not p.get("enabled", True):
                continue
            self.providers.append(Provider(
                name=name,
                base_url=base_url,
                api_key=p.get("api_key"),
                models=p.get("models"),
                family=p.get("family"),
                headers=p.get("headers"),
                require_tools=p.get("require_tools", False),
                enabled=p.get("enabled", True),
            ))


class Proxy:
    def __init__(self, config):
        self.config = config
        self.log_level = config.log_level
        self.lock = threading.RLock()
        self._opener_cache = None
        self.session_id = uuid.uuid4().hex
        self.model_routes = {}       # Ollama model name(lower) -> (provider name, upstream id)
        self.base_model_routes = {}  # model base(lower) -> first (provider name, upstream id)
        self.fetched_at = {}         # provider name -> 上次 /models 时间
        self.fetched_ids = {}        # provider name -> [model ids]
        self.models_entries = None
        self.models_defaults = {}
        self.fetch_wait = threading.Condition(self.lock)
        self.fetching = set()
        self.refreshing = set()
        self._init_owner_from_config()
        self._init_routes_from_models()

    # ---------------- 通用 ----------------
    def log(self, msg, level="info"):
        levels = {"quiet": 0, "info": 1, "debug": 2}
        want = levels.get(self.log_level, 1)
        if want < levels.get(level, 1):
            return
        stream = sys.stderr if level == "error" else sys.stdout
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), file=stream, flush=True)

    def _opener(self):
        # The proxy/opener policy is fixed at startup (config.use_env_proxy), so build
        # the opener once and reuse it — build_opener() has non-trivial per-call cost.
        if self._opener_cache is None:
            if self.config.use_env_proxy:
                self._opener_cache = urllib.request.build_opener()
            else:
                self._opener_cache = urllib.request.build_opener(
                    urllib.request.ProxyHandler({}))
        return self._opener_cache

    def get_provider(self, name):
        for provider in self.config.providers:
            if provider.name.lower() == str(name).lower():
                return provider
        return None

    @staticmethod
    def _model_base(name):
        return str(name).split(":", 1)[0]

    @staticmethod
    def _provider_tag_suffix(name):
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_."
        value = str(name).lower()
        return "".join(ch if ch in allowed else "-" for ch in value) or "provider"

    @classmethod
    def _qualified_model_name(cls, upstream_id, provider_name):
        base, separator, tag = str(upstream_id).partition(":")
        suffix = cls._provider_tag_suffix(provider_name)
        if separator and tag:
            if tag.lower() == "latest" or tag == suffix:
                return base + ":" + suffix
            return base + ":" + tag + "-" + suffix
        return base + ":" + suffix

    @staticmethod
    def _infer_model_family(name):
        """根据常见命名推演 family；推演不出时返回空字符串。"""
        value = str(name).lower().replace("_", "-")
        rules = (
            ("paligemma", "paligemma"),
            ("nemotron", "nemotron"),
            ("deepseek", "deepseek"),
            ("minimax", "minimax"),
            ("step-", "step"),
            ("ising-calibration", "gemma4"),
            ("gemma", "gemma"),
            ("glm", "glm"),
            ("qwen", "qwen"),
            ("muse", "muse"),
            ("llama", "llama"),
            ("gpt-oss", "gptoss"),
            ("gpt", "gpt"),
            ("claude", "claude"),
        )
        for needle, family in rules:
            if needle in value:
                return family
        return ""

    @staticmethod
    def _number(value):
        """尽量转成整数量纲；不合法时返回 None。"""
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return int(number) if number.is_integer() else number

    def _register_route(self, provider, upstream_id, ollama_name):
        provider_name = provider.name
        route = (provider_name, str(upstream_id))
        self.model_routes.setdefault(str(ollama_name).lower(), route)
        self.base_model_routes.setdefault(self._model_base(upstream_id).lower(), route)
        if ollama_name != upstream_id:
            self.model_routes.setdefault(str(upstream_id).lower(), route)

    def _register_provider_model(self, provider, upstream_id):
        plain = str(upstream_id) if ":" in str(upstream_id) else str(upstream_id) + ":latest"
        qualified = self._qualified_model_name(upstream_id, provider.name)
        self._register_route(provider, upstream_id, plain)
        self._register_route(provider, upstream_id, qualified)

    def _init_owner_from_config(self):
        for provider in self.config.providers:
            for mid in provider.models:
                self._register_provider_model(provider, mid)
        for alias, target in self.config.mapping.items():
            provider = self.get_provider(target.get("provider") or target.get("name"))
            if provider and target.get("model"):
                self._register_route(provider, target["model"], alias)

    def _init_routes_from_models(self):
        for item in self.list_models_files():
            provider = self.get_provider(item["provider"])
            if provider:
                self._register_provider_model(provider, item["upstream"])
                if isinstance(item.get("data"), dict) and item["data"] and item.get("name"):
                    self._register_route(provider, item["upstream"], item["name"])
                    self._register_route(
                        provider, item["upstream"],
                        self._qualified_model_name(item["name"], provider.name))

    def upstream_request(self, provider, url, payload=None):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if provider.api_key:
            headers["Authorization"] = "Bearer " + provider.api_key
        headers.update(provider.headers)
        session_id = getattr(self, "session_id", "")
        for key, value in provider.headers.items():
            headers[key] = str(value).replace("{session_id}", session_id).replace("${session_id}", session_id)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url, data=data, headers=headers,
            method="POST" if payload is not None else "GET")
        started = time.time()
        self.log("upstream 请求 provider=%s %s %s bytes=%d header_names=%s" % (
            provider.name, req.get_method(), url,
            len(data or b""),
            ",".join(sorted(name for name, _value in req.header_items()))), level="debug")
        try:
            response = self._opener().open(req, timeout=self.config.timeout)
            self.log("upstream 响应 provider=%s status=%s type=%s bytes=%d elapsed=%.2fs" % (
                provider.name, response.status,
                response.headers.get("Content-Type", ""),
                int(response.headers.get("Content-Length") or 0),
                time.time() - started), level="debug")
            return response
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", "replace")
            self.log("upstream HTTP错误 provider=%s status=%s type=%s body=%s elapsed=%.2fs" % (
                provider.name, exc.code, exc.headers.get("Content-Type", ""),
                err_body[:300], time.time() - started), level="debug")
            raise UpstreamError(exc.code, err_body) from exc
        except urllib.error.URLError as exc:
            self.log("upstream 连接错误 provider=%s reason=%s elapsed=%.2fs" % (
                provider.name, exc.reason, time.time() - started), level="debug")
            raise UpstreamError(0, str(exc.reason)) from exc

    def _retry_without_tools(self, provider, url, payload, exc):
        """上游 5xx 且请求带 tools 时,剥离 tools 重试一次。"""
        if not self.config.retry_without_tools or exc.status not in (500, 502, 503):
            return None
        if getattr(provider, "require_tools", False):
            retry_payload = dict(payload)
            retry_payload.setdefault("tools", NOOP_TOOLS)
            self.log("上游 %s 错误,模型要求 tools,保留工具后重试 model=%s provider=%s url=%s" % (
                exc.status, retry_payload.get("model"), provider.name, url))
            try:
                return self.upstream_request(provider, url, retry_payload)
            except UpstreamError as retry_exc:
                self.log("保留工具重试仍失败 model=%s provider=%s status=%s body=%s" % (
                    retry_payload.get("model"), provider.name, retry_exc.status, (retry_exc.body or "")[:200]), level="error")
                raise retry_exc
        if not payload.get("tools"):
            return None
        stripped = dict(payload)
        stripped.pop("tools", None)
        stripped.pop("tool_choice", None)
        self.log("上游 %s 错误,剥离 tools 后重试 model=%s provider=%s url=%s" % (
            exc.status, payload.get("model"), provider.name, url))
        try:
            return self.upstream_request(provider, url, stripped)
        except UpstreamError as retry_exc:
            self.log("剥离 tools 重试仍失败 model=%s provider=%s status=%s body=%s" % (
                payload.get("model"), provider.name, retry_exc.status, (retry_exc.body or "")[:200]), level="error")
            raise retry_exc

    # ---------------- 模型列表 ----------------
    def fetch_provider_models(self, provider, force=False):
        now = time.time()
        with self.lock:
            cached = self.fetched_ids.get(provider.name)
            cached_at = self.fetched_at.get(provider.name, 0)
            if not force and cached is not None:
                if (now - cached_at) < self.config.cache_ttl:
                    self.log("[%s] /models 使用缓存 %d 个模型" % (provider.name, len(cached)), level="debug")
                    return list(cached)
                if provider.name not in self.refreshing:
                    self.refreshing.add(provider.name)
                    self.log("[%s] /models 缓存过期,后台刷新 %d 个模型" % (provider.name, len(cached)), level="debug")
                    threading.Thread(
                        target=self._background_refresh, args=(provider,),
                        daemon=True, name="models-refresh-" + provider.name).start()
                return list(cached)
            if provider.name in self.fetching:
                deadline = time.time() + self.config.fetch_wait_timeout
                while provider.name in self.fetching and time.time() < deadline:
                    self.fetch_wait.wait(0.2)
                cached = self.fetched_ids.get(provider.name)
                if cached is not None:
                    return list(cached)
                return list(provider.models)
            self.fetching.add(provider.name)
            self.log("[%s] /models 首次获取中..." % provider.name, level="debug")
        try:
            ids = self._fetch_provider_models_raw(provider)
        except Exception as exc:
            self.log("[%s] /models 获取失败: %s" % (provider.name, exc), level="error")
            return list(provider.models)
        finally:
            with self.lock:
                self.fetching.discard(provider.name)
                self.fetch_wait.notify_all()
        return ids

    def _fetch_provider_models_raw(self, provider):
        ids = []
        if provider.models:
            ids = list(provider.models)
        else:
            resp = self.upstream_request(provider, provider.models_url)
            try:
                raw = resp.read().decode("utf-8", "replace")
            finally:
                resp.close()
            data = json.loads(raw)
            ids = [str(m.get("id", "")).strip() for m in data.get("data", [])
                   if str(m.get("id", "")).strip()]
            self.log("[%s] /models 获取到 %d 个模型" % (provider.name, len(ids)))
        with self.lock:
            self.fetched_ids[provider.name] = list(ids)
            self.fetched_at[provider.name] = time.time()
            for mid in ids:
                self._register_provider_model(provider, mid)
        return list(ids)

    def _background_refresh(self, provider):
        try:
            self._fetch_provider_models_raw(provider)
        except Exception as exc:
            self.log("[%s] /models 后台刷新失败: %s" % (provider.name, exc), level="error")
        finally:
            with self.lock:
                self.refreshing.discard(provider.name)
                self.fetch_wait.notify_all()

    def _fetch_missing_models(self, providers):
        with self.lock:
            missing = [p for p in providers if p.name not in self.fetched_ids]
        if not missing:
            return
        threads = []
        for provider in missing:
            thread = threading.Thread(
                target=self.fetch_provider_models, args=(provider,),
                daemon=True, name="models-fetch-" + provider.name)
            thread.start()
            threads.append(thread)
        deadline = time.time() + self.config.fetch_wait_timeout
        for thread in threads:
            thread.join(max(0.0, deadline - time.time()))

    def warm_models(self):
        for provider in self.config.providers:
            if provider.name in self.fetched_ids:
                continue
            threading.Thread(
                target=self.fetch_provider_models, args=(provider,),
                daemon=True, name="models-warm-" + provider.name).start()

    def resolve_model(self, name):
        """把 Ollama 模型名精确映射为 (provider, upstream_model_id)。"""
        name = (name or "").strip()
        if not name:
            return None, None

        mapping_target = self.config.mapping.get(name)
        if mapping_target is None:
            mapping_target = self.config.mapping.get(self._model_base(name))
        if isinstance(mapping_target, dict):
            provider = self.get_provider(mapping_target.get("provider") or mapping_target.get("name"))
            if provider:
                return provider, str(mapping_target.get("model") or self._model_base(name))

        with self.lock:
            route = self.model_routes.get(name.lower())
            if route is None:
                route = self.base_model_routes.get(self._model_base(name).lower())
        if route:
            provider = self.get_provider(route[0])
            if provider:
                return provider, route[1]

        for provider in self.config.providers:
            for mid in provider.models:
                if mid == name or self._model_base(mid) == self._model_base(name):
                    return provider, mid

        self._fetch_missing_models(self.config.providers)

        with self.lock:
            route = self.model_routes.get(name.lower())
            if route is None:
                route = self.base_model_routes.get(self._model_base(name).lower())
        if route:
            provider = self.get_provider(route[0])
            if provider:
                return provider, route[1]

        return None, None

    def public_model_name(self, provider, upstream_id):
        # 统一命名:<上游模型>:<provider>,不再使用 :latest 作为默认后缀。
        return self._qualified_model_name(upstream_id, provider.name)

    # ---------------- models/*.json 匹配 ----------------
    # ---------------- provider-scoped models/*.json matching ----------------
    def list_models_files(self):
        if self.models_entries is None:
            files = []
            if os.path.isdir(self.config.models_dir):
                for root, _dirs, filenames in os.walk(self.config.models_dir):
                    for filename in sorted(filenames):
                        if filename.lower().endswith(".json"):
                            files.append(os.path.join(root, filename))
            entries = []
            def add_entry(item):
                key = (item["provider"].lower(), item["upstream"].lower())
                index = next((i for i, existing in enumerate(entries)
                              if (existing["provider"].lower(),
                                  existing["upstream"].lower()) == key), None)
                if index is None:
                    entries.append(item)
                elif isinstance(item.get("data"), dict) and item["data"]:
                    # 新 provider 文件优先于仍留在目录里的旧版 per-model 模板。
                    entries[index] = item

            for path in files:
                data = self.load_models_file(path)
                if not isinstance(data, dict):
                    continue
                stem = os.path.splitext(os.path.basename(path))[0]
                defaults = data.get("defaults") or data.get("default_model") or {}
                if isinstance(defaults, dict) and defaults and data.get("provider"):
                    self.models_defaults[str(data["provider"])] = defaults
                api_type = data.get("api_type")
                providers = data.get("providers")
                if isinstance(providers, dict):
                    scoped_entries = providers.items()
                elif data.get("provider") and isinstance(data.get("models"), list):
                    # 新格式:一个 provider 一个文件,models 数组保存多个模型。
                    scoped_entries = [(data.get("provider"), data)]
                elif data.get("provider"):
                    scoped_entries = [(data.get("provider"), {
                        "model": data.get("model") or stem,
                        "tag": data.get("tag"),
                        "show": data.get("show"),
                    })]
                else:
                    continue

                for provider_name, entry in scoped_entries:
                    if not isinstance(provider_name, str) or not isinstance(entry, dict):
                        continue

                    if isinstance(entry.get("models"), list):
                        for model in entry["models"]:
                            if not isinstance(model, dict):
                                continue
                            upstream_id = str(model.get("model") or model.get("name") or stem)
                            tag_name = str(model.get("name") or self._model_base(upstream_id))
                            add_entry({
                                "provider": provider_name,
                                "upstream": upstream_id,
                                "base": self._model_base(upstream_id),
                                "name": tag_name,
                                "stem": stem,
                                "data": model,
                                "tag": {},
                                "show": {},
                                "api_type": model.get("api_type") or api_type,
                                "defaults": defaults,
                                "path": path,
                            })
                        continue

                    upstream_id = str(entry.get("model") or stem)
                    tag = entry.get("tag") or entry.get("tag_model") or {}
                    tag_name = str(tag.get("name") or tag.get("model") or upstream_id)
                    show = entry.get("show") or {}
                    add_entry({
                        "provider": provider_name,
                        "upstream": upstream_id,
                        "base": self._model_base(upstream_id),
                        "name": tag_name,
                        "stem": stem,
                        "data": {},
                        "tag": tag,
                        "show": show,
                        "api_type": api_type if api_type is not None else show.get("api_type"),
                        "defaults": defaults,
                        "path": path,
                    })
            self.models_entries = entries
        return self.models_entries

    def load_models_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def find_models_entry(self, provider, upstream_id, ollama_name=None):
        """Find a model template by the owning (provider, upstream model) pair."""
        entries = self.list_models_files()
        provider_l = str(getattr(provider, "name", provider)).lower()
        upstream_l = str(upstream_id).lower()
        base_l = self._model_base(upstream_id).lower()

        for item in entries:
            if item["provider"].lower() == provider_l and item["upstream"].lower() == upstream_l:
                return item, item["path"]
        if ollama_name:
            name_l = str(ollama_name).lower()
            for item in entries:
                if item["provider"].lower() != provider_l:
                    continue
                if name_l in (item["name"].lower(), item["upstream"].lower(),
                              item["base"].lower(), item["stem"].lower()):
                    return item, item["path"]
        for item in entries:
            if item["provider"].lower() == provider_l and item["base"].lower() == base_l:
                return item, item["path"]
        return None, None

    def get_api_type(self, provider, upstream_id):
        """返回模型 API 类型: 'chat_completions' 或 'responses'。默认 'chat_completions'。"""
        entry, _path = self.find_models_entry(provider, upstream_id)
        if entry and entry.get("api_type"):
            return str(entry["api_type"])
        return "chat_completions"

    @staticmethod
    def chat_to_responses_payload(chat_payload):
        """将 Chat Completions 请求体转换为 Responses API 请求体。"""
        resp = {"model": chat_payload.get("model", "")}
        messages = chat_payload.get("messages", [])
        inp = []
        system_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts)
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            if role == "system":
                system_text += (system_text and "\n" or "") + content
            else:
                inp.append({"role": role, "content": content})
        if system_text:
            resp["instructions"] = system_text
        resp["input"] = inp
        tools = chat_payload.get("tools")
        if tools:
            converted_tools = []
            for tool in tools:
                fn = tool.get("function", {})
                converted_tools.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            resp["tools"] = converted_tools
        if chat_payload.get("stream") is not None:
            resp["stream"] = bool(chat_payload["stream"])
        return resp

    @staticmethod
    def responses_to_chat_response(resp_data, model_name):
        """将 Responses API 非流式响应转换为 Chat Completions 格式。"""
        output_items = resp_data.get("output", [])
        text_parts = []
        tool_calls = []
        tc_index = 0
        for item in output_items:
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))
            elif item_type == "function_call":
                args = item.get("arguments", "{}")
                try:
                    json.loads(args)
                except Exception:
                    args = "{}"
                tool_calls.append({
                    "id": item.get("id", f"call_{tc_index}"),
                    "type": "function",
                    "index": tc_index,
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": args,
                    },
                })
                tc_index += 1
        usage_raw = resp_data.get("usage", {}) or {}
        usage = {
            "prompt_tokens": usage_raw.get("input_tokens", 0),
            "completion_tokens": usage_raw.get("output_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        }
        message = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = "tool_calls" if tool_calls else "stop"
        return {
            "id": resp_data.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }

    def responses_stream_to_v1_sse(self, resp, model_name, client_wants_stream=True):
        """将 Responses API SSE 流转换为 Chat Completions SSE 流。"""
        base_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        usage_data = {}
        finish_reason = "stop"

        def generate():
            nonlocal finish_reason, usage_data
            for event_type, data in iter_sse_events(resp):
                if event_type == "response.output_text.delta":
                    delta_text = data.get("delta", "")
                    if not delta_text:
                        continue
                    chunk = {
                        "id": base_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                    }
                    yield self._sse_bytes(chunk)
                elif event_type == "response.output_item.added":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        tc_chunk = {
                            "id": base_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"tool_calls": [{
                                "index": 0,
                                "id": item.get("id", ""),
                                "type": "function",
                                "function": {"name": item.get("name", ""), "arguments": ""},
                            }]}, "finish_reason": None}],
                        }
                        yield self._sse_bytes(tc_chunk)
                elif event_type == "response.output_item.done":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        args = item.get("arguments", "{}")
                        tc_chunk = {
                            "id": base_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"tool_calls": [{
                                "index": 0,
                                "function": {"arguments": args},
                            }]}, "finish_reason": None}],
                        }
                        yield self._sse_bytes(tc_chunk)
                        finish_reason = "tool_calls"
                elif event_type == "response.completed":
                    final_resp = data.get("response", {})
                    usage_raw = final_resp.get("usage", {}) or {}
                    usage_data = {
                        "prompt_tokens": usage_raw.get("input_tokens", 0),
                        "completion_tokens": usage_raw.get("output_tokens", 0),
                        "total_tokens": usage_raw.get("total_tokens", 0),
                    }
            final_event = {
                "id": base_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            if usage_data:
                final_event["usage"] = usage_data
            yield self._sse_bytes(final_event)
            yield b"data: [DONE]\n\n"

        return generate()

    def _provider_model_ids(self, provider):
        """合并远端发现的模型与 provider 静态目录；静态目录可在上游失败时兜底。"""
        ids = list(self.fetch_provider_models(provider))
        existing_bases = {self._model_base(mid).lower() for mid in ids}
        catalog_ids = [item["upstream"] for item in self.list_models_files()
                       if item["provider"].lower() == provider.name.lower()]
        for mid in catalog_ids:
            if mid not in ids and self._model_base(mid).lower() not in existing_bases:
                ids.append(mid)
                existing_bases.add(self._model_base(mid).lower())
        return ids

    # ---------------- /api/tags and /api/show responses ----------------
    def tags(self):
        entries, seen = [], set()
        self._fetch_missing_models(self.config.providers)
        for provider in self.config.providers:
            ids = self._provider_model_ids(provider)
            for mid in ids:
                ollama_name = self._public_model_name_for(provider, mid)
                key = ollama_name.lower()
                if key in seen:
                    continue
                seen.add(key)
                entries.append(self.tags_entry_for(ollama_name, mid, provider))
        self.log("tags 返回 %d 个模型(provider=%d)" % (len(entries), len(self.config.providers)))
        return {"models": entries}

    def tags_entry_for(self, ollama_name, upstream_id, provider):
        entry, path = self.find_models_entry(provider, upstream_id, ollama_name)
        if entry is not None:
            self.log("tags model=%s provider=%s source=file(%s)" % (
                ollama_name, provider.name, os.path.basename(path)), level="debug")
            return self._tags_response(entry, ollama_name)
        return self.generate_tag_entry(ollama_name, provider, upstream_id)

    def _public_model_name_for(self, provider, upstream_id):
        entry, _path = self.find_models_entry(provider, upstream_id)
        display_name = entry.get("name") if entry else ""
        return self._qualified_model_name(display_name or upstream_id, provider.name)

    def generate_tag_entry(self, ollama_name, provider, upstream_id=None):
        upstream_id = upstream_id or self._public_model_base(ollama_name)
        return {
            "name": ollama_name,
            "model": ollama_name,
            "modified_at": now_iso(),
            "size": 0,
            "digest": uuid.uuid4().hex[:12],
            "details": {
                "parent_model": "",
                "format": "",
                "family": "",
                "families": None,
                "parameter_size": "",
                "quantization_level": "",
            },
        }

    def _stable_digest(self, provider_name, model_id):
        seed = "%s:%s" % (provider_name, str(model_id))
        return uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:12]

    def _defaults_for(self, item):
        defaults = item.get("defaults") if isinstance(item.get("defaults"), dict) else {}
        merged = dict(defaults)
        value = defaults.get("capabilities")
        merged["capabilities"] = [str(cap) for cap in value] if isinstance(value, list) else []
        merged["details"] = dict(defaults.get("details") or {})
        merged["model_info"] = dict(defaults.get("model_info") or {})
        return merged

    def _provider_defaults(self, provider):
        defaults = self.models_defaults.get(str(provider.name))
        if isinstance(defaults, dict):
            return self._defaults_for({"defaults": defaults})
        for item in self.list_models_files():
            if item["provider"].lower() == str(provider.name).lower() and item.get("defaults"):
                return self._defaults_for(item)
        return {
            "capabilities": [],
            "details": {},
            "model_info": {},
        }

    def _model_values(self, item):
        """把新旧模板统一成供 tags/show 使用的扁平元数据。"""
        defaults = self._defaults_for(item)
        if isinstance(item.get("data"), dict) and item["data"]:
            values = dict(item["data"])
        else:
            tag = item.get("tag") if isinstance(item.get("tag"), dict) else {}
            show = item.get("show") if isinstance(item.get("show"), dict) else {}
            values = {
                "name": tag.get("name") or tag.get("model") or item.get("upstream"),
                "digest": tag.get("digest"),
                "size": tag.get("size"),
                "modified_at": tag.get("modified_at") or show.get("modified_at"),
                "capabilities": show.get("capabilities"),
                "details": show.get("details") or {},
                "model_info": show.get("model_info") or {},
            }

        capabilities = values.get("capabilities")
        capabilities = [str(cap) for cap in capabilities] if isinstance(capabilities, list) else []
        if not capabilities:
            capabilities = defaults["capabilities"] or ["completion", "tools"]
        values["capabilities"] = capabilities
        if not isinstance(values.get("details"), dict):
            values["details"] = {}
        if not isinstance(values.get("model_info"), dict):
            values["model_info"] = {}
        return values, defaults

    def _tags_response(self, item, public_name):
        values, defaults = self._model_values(item)
        digest = values.get("digest") or defaults.get("digest")
        if not digest:
            digest = self._stable_digest(item["provider"], item["upstream"])
        try:
            size = int(values.get("size", defaults.get("size", 0)) or 0)
        except (TypeError, ValueError):
            size = 0
        return {
            "name": public_name,
            "model": public_name,
            "modified_at": values.get("modified_at") or now_iso(),
            "size": size,
            "digest": str(digest),
            # 列表只保留轻量详情；完整参数由 /api/show 返回。
            "details": {
                "parent_model": "",
                "format": "",
                "family": "",
                "families": None,
                "parameter_size": "",
                "quantization_level": "",
            },
        }

    def _public_model_base(self, name):
        """去掉公开模型名上的 :latest / :provider 后缀，得到目录匹配名。"""
        return self._model_base(name)

    def _show_response(self, item, provider, ollama_name, upstream_id):
        values, defaults = self._model_values(item)
        raw_details = values["details"]
        family = str(raw_details.get("family")
                     or self._infer_model_family(item.get("name") or upstream_id)
                     or provider.family or "")

        detail_defaults = defaults.get("details")

        def detail_text(key):
            value = raw_details.get(key)
            if value is None or str(value) == "":
                value = detail_defaults.get(key)
            return "" if value is None else str(value)

        parameter_size = detail_text("parameter_size")
        parameter_count = self._number(parameter_size)

        info_values = values["model_info"]
        context_key = "%s.context_length" % family
        embedding_key = "%s.embedding_length" % family
        context_length = (self._number(info_values.get(context_key))
                          or self._number(defaults.get("context_length"))
                          or self._number(defaults["model_info"].get(context_key))
                          or 128000)
        embedding_length = (self._number(info_values.get(embedding_key))
                            or self._number(defaults.get("embedding_length"))
                            or self._number(defaults["model_info"].get(embedding_key))
                            or 2048)

        return {
            "capabilities": list(values["capabilities"]),
            "details": {
                "parent_model": detail_text("parent_model"),
                "format": detail_text("format"),
                "family": family,
                "families": None,
                "parameter_size": parameter_size,
                "quantization_level": detail_text("quantization_level"),
            },
            "model_info": {
                "general.architecture": family,
                "general.parameter_count": parameter_count,
                context_key: context_length,
                embedding_key: embedding_length,
            },
            "modified_at": values.get("modified_at") or now_iso(),
        }

    def show_for(self, name):
        provider, upstream = self.resolve_model(name)
        if provider is not None and upstream:
            entry, path = self.find_models_entry(provider, upstream, name)
            if entry is not None:
                self.log("show model=%s provider=%s source=file(%s)" % (
                    name, provider.name, os.path.basename(path)))
                return self._show_response(entry, provider, name, upstream)
        if provider is None:
            provider = self.config.providers[0]
        self.log("show model=%s source=auto provider=%s" % (name, provider.name))
        return self.generate_show(name, provider)

    def generate_show(self, ollama_name, provider, upstream_id=None):
        model_hint = upstream_id or self._public_model_base(ollama_name)
        defaults = self._provider_defaults(provider)
        default_details = defaults.get("details", {})
        family = (default_details.get("family")
                  or self._infer_model_family(model_hint)
                  or provider.family or "")
        return {
            "capabilities": list(defaults.get("capabilities") or ["completion", "tools"]),
            "details": {
                "parent_model": "",
                "format": "",
                "family": family,
                "families": None,
                "parameter_size": str(default_details.get("parameter_size") or ""),
                "quantization_level": str(default_details.get("quantization_level") or ""),
            },
            "model_info": {
                "general.architecture": family,
                "general.parameter_count": None,
                "%s.context_length" % family: self._number(
                    defaults.get("context_length")) or 128000,
                "%s.embedding_length" % family: self._number(
                    defaults.get("embedding_length")) or 2048,
            },
            "modified_at": now_iso(),
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
    def build_chat_payload(self, body, upstream_model, provider=None):
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
        if provider is not None and getattr(provider, "require_tools", False):
            if not params.get("tools"):
                params["tools"] = NOOP_TOOLS
        elif self.config.strip_tools:
            params.pop("tools", None)
            params.pop("tool_choice", None)
        return params

    def chat(self, body):
        t0 = time.time()
        ollama_name = body.get("model", "")
        provider, upstream = self.resolve_model(ollama_name)
        if provider is None:
            raise ModelNotFoundError("model '%s' not found, try pulling it first" % ollama_name)
        api_type = self.get_api_type(provider, upstream)
        url = provider.responses_url if api_type == "responses" else provider.chat_url
        self.log("chat model=%s provider=%s url=%s stream=false" % (ollama_name, provider.name, url))
        payload = self.build_chat_payload(body, upstream, provider)
        if api_type == "responses":
            payload["stream"] = False
            resp_payload = self.chat_to_responses_payload(payload)
            try:
                resp = self.upstream_request(provider, provider.responses_url, resp_payload)
            except UpstreamError as exc:
                if resp_payload.get("tools") and self.config.retry_without_tools and exc.status in (500, 502, 503):
                    retry_payload = dict(resp_payload)
                    retry_payload.pop("tools", None)
                    try:
                        resp = self.upstream_request(provider, provider.responses_url, retry_payload)
                    except UpstreamError:
                        raise exc
                else:
                    raise
            raw = resp.read().decode("utf-8", "replace")
            resp.close()
            resp_data = json.loads(raw)
            data = self.responses_to_chat_response(resp_data, ollama_name)
        else:
            try:
                resp = self.upstream_request(provider, provider.chat_url, payload)
            except UpstreamError as exc:
                resp = self._retry_without_tools(provider, provider.chat_url, payload, exc)
                if resp is None:
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
        payload = self.build_chat_payload(body, upstream, provider)
        api_type = self.get_api_type(provider, upstream)
        if api_type == "responses":
            payload["stream"] = True
            resp_payload = self.chat_to_responses_payload(payload)
            try:
                resp = self.upstream_request(provider, provider.responses_url, resp_payload)
            except UpstreamError as exc:
                if resp_payload.get("tools") and self.config.retry_without_tools and exc.status in (500, 502, 503):
                    retry_payload = dict(resp_payload)
                    retry_payload.pop("tools", None)
                    try:
                        resp = self.upstream_request(provider, provider.responses_url, retry_payload)
                    except UpstreamError:
                        raise exc
                else:
                    raise
            full_content, tool_calls, usage = [], {}, {}
            final_reason = "stop"
            for event_type, event_data in iter_sse_events(resp):
                if event_type == "response.output_text.delta":
                    full_content.append(event_data.get("delta", ""))
                elif event_type == "response.output_item.done":
                    item = event_data.get("item", {})
                    if item.get("type") == "function_call":
                        tc = {"function": {"name": item.get("name",""), "arguments": item.get("arguments","{}")}}
                        tool_calls[len(tool_calls)] = tc
                elif event_type == "response.completed":
                    fr = event_data.get("response", {}).get("usage", {}) or {}
                    usage = {"prompt_tokens": fr.get("input_tokens",0), "eval_count": fr.get("output_tokens",0)}
            message = {"role":"assistant","content":"".join(full_content)}
            write(ndjson({"model": ollama_name, "created_at": now_iso(), "message": message, "done": False}))
            final = {"model": ollama_name, "created_at": now_iso(), "done": True,
                     "total_duration": int((time.time()-t0)*1e9),
                     "prompt_eval_count": usage.get("prompt_tokens",0), "eval_count": usage.get("eval_count",0)}
            write(ndjson(final))
            return
        try:
            resp = self.upstream_request(provider, provider.chat_url, payload)
        except UpstreamError as exc:
            resp = self._retry_without_tools(provider, provider.chat_url, payload, exc)
            if resp is None:
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
        dsml_text, dsml_tool_calls = parse_dsml_content(content)
        existing_tcs = message.get("tool_calls") or []
        all_tool_calls = list(existing_tcs) + dsml_tool_calls
        omsg = {"role": message.get("role", "assistant"), "content": dsml_text}
        done_reason = choice.get("finish_reason") or "stop"
        if all_tool_calls:
            converted = Proxy.openai_tool_calls_to_ollama(all_tool_calls) if existing_tcs else [
                {"function": {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}}
                for tc in all_tool_calls
            ]
            omsg["tool_calls"] = converted
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
    @staticmethod
    def _get_image_mime(data):
        """嗅探 base64 图片的真实 MIME 类型(而非一律当成 PNG)。

        Ollama 的 /api/generate ``images`` 只传裸 base64,不携带格式信息;
        此前代码硬编码 ``image/png``,jpg/webp/gif 等多模态请求会因 MIME 错误被上游拒绝。
        通过解码后的魔数判断真实类型,无法识别时回退到 PNG(与旧行为一致)。
        """
        try:
            raw = base64.b64decode(data.encode("ascii") if isinstance(data, str) else data)
        except (ValueError, TypeError):
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
        """容忍 ``data:image/...;base64,`` 前缀(部分客户端会带上),剥离为纯 base64。"""
        out = []
        for img in images or []:
            value = str(img).strip()
            if "," in value and value.lstrip().lower().startswith("data:"):
                value = value.split(",", 1)[1]
            out.append(value)
        return out

    def build_generate_payload(self, body, upstream_model):
        prompt = body.get("prompt", "") or ""
        messages = []
        if body.get("system"):
            messages.append({"role": "system", "content": str(body["system"])})
        images = self._strip_data_uri(body.get("images") or [])
        if images:
            content = [{"type": "text", "text": str(prompt)}]
            for img in images:
                content.append({"type": "image_url",
                                "image_url": {"url": "data:%s;base64,%s"
                                              % (self._get_image_mime(img), img)}})
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
        api_type = self.get_api_type(provider, upstream)
        if api_type == "responses":
            payload["stream"] = False
            resp_payload = self.chat_to_responses_payload(payload)
            try:
                resp = self.upstream_request(provider, provider.responses_url, resp_payload)
            except UpstreamError:
                retry_payload = dict(resp_payload)
                retry_payload.pop("tools", None)
                if retry_payload.get("tools") and self.config.retry_without_tools:
                    resp = self.upstream_request(provider, provider.responses_url, retry_payload)
                else:
                    raise
            raw = resp.read().decode("utf-8", "replace")
            resp.close()
            resp_data = json.loads(raw)
            data = self.responses_to_chat_response(resp_data, ollama_name)
        else:
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
        api_type = self.get_api_type(provider, upstream)
        if api_type == "responses":
            payload["stream"] = True
            resp_payload = self.chat_to_responses_payload(payload)
            try:
                resp = self.upstream_request(provider, provider.responses_url, resp_payload)
            except UpstreamError as exc:
                retry_payload = dict(resp_payload)
                retry_payload.pop("tools", None)
                if retry_payload.get("tools") and self.config.retry_without_tools and exc.status in (500, 502, 503):
                    try:
                        resp = self.upstream_request(provider, provider.responses_url, retry_payload)
                    except UpstreamError:
                        raise exc
                else:
                    raise
            full_text = []
            usage_data = {}
            for event_type, event_data in iter_sse_events(resp):
                if event_type == "response.output_text.delta":
                    delta = event_data.get("delta", "")
                    full_text.append(delta)
                    write(ndjson({"model": ollama_name, "created_at": now_iso(), "response": delta, "done": False}))
                elif event_type == "response.completed":
                    fr = event_data.get("response", {}).get("usage", {}) or {}
                    usage_data = {"prompt_tokens": fr.get("input_tokens",0), "eval_count": fr.get("output_tokens",0)}
            write(ndjson({"model": ollama_name, "created_at": now_iso(), "response": "", "done": True,
                          "total_duration": int((time.time()-t0)*1e9),
                          "prompt_eval_count": usage_data.get("prompt_tokens",0), "eval_count": usage_data.get("eval_count",0)}))
            return
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
            for mid in self._provider_model_ids(provider):
                tag = self._public_model_name_for(provider, mid)
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

    def _resolved_upstream_stream(self, client_stream):
        if self.config.stream_mode == "stream":
            return True
        if self.config.stream_mode == "non_stream":
            return False
        return bool(client_stream)

    @staticmethod
    def _sse_bytes(obj):
        return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")

    def aggregate_v1_stream(self, resp, model_name):
        """把上游 SSE 流聚合为一次完整的 OpenAI chat.completion 响应。"""
        usage = {}
        content = []
        reasoning = []
        tool_calls = {}
        finish_reason = None
        response_id = ""
        created = 0
        try:
            for payload_text in iter_sse(resp):
                if not payload_text.strip():
                    continue
                try:
                    chunk = json.loads(payload_text)
                except Exception:
                    continue
                usage = chunk.get("usage") or usage
                response_id = chunk.get("id") or response_id
                created = chunk.get("created") or created
                for choice in chunk.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    value = delta.get("content")
                    if value:
                        content.append(value)
                    rvalue = delta.get("reasoning_content")
                    if rvalue:
                        reasoning.append(rvalue)
                    for tcd in delta.get("tool_calls") or []:
                        idx = tcd.get("index", 0)
                        slot = tool_calls.setdefault(idx, {"function": {"name": "", "arguments": ""}})
                        fn = tcd.get("function") or {}
                        slot["function"]["name"] += fn.get("name", "") or ""
                        slot["function"]["arguments"] += fn.get("arguments", "") or ""
        finally:
            resp.close()
        raw_text = "".join(content)
        clean_text, dsml_tool_calls = parse_dsml_content(raw_text)
        message = {"role": "assistant", "content": clean_text}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        all_tc = [tool_calls[i] for i in sorted(tool_calls)] if tool_calls else []
        for dtc in dsml_tool_calls:
            all_tc.append({"function": dtc["function"]})
        if all_tc:
            message["tool_calls"] = all_tc
            finish_reason = finish_reason if finish_reason == "stop" else "tool_calls"
        return {
            "id": response_id or "chatcmpl-proxy",
            "object": "chat.completion",
            "created": created or int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": message,
                          "finish_reason": finish_reason or "stop", "logprobs": None}],
            "usage": usage or {},
        }

    def _dsml_filtered_v1_stream(self, resp, model_name):
        """包装上游 SSE 流,实时检测并转换 DeepSeek DSML 工具调用标记。"""
        base_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        text_buffer = []
        dsml_active = False
        emitted_index = 0
        finish_reason = None
        usage_data = {}

        def generate():
            nonlocal dsml_active, finish_reason, usage_data, emitted_index
            for payload_text in iter_sse(resp):
                if not payload_text.strip():
                    continue
                try:
                    chunk = json.loads(payload_text)
                except Exception:
                    continue
                if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
                    usage_data = chunk["usage"]
                fr = None
                for choice in chunk.get("choices") or []:
                    if choice.get("finish_reason"):
                        fr = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    value = delta.get("content")
                    # Forward reasoning_content and tool_calls unchanged
                    rc = delta.get("reasoning_content")
                    tc = delta.get("tool_calls")
                    if rc:
                        yield self._sse_bytes({
                            "id": base_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"reasoning_content": rc}, "finish_reason": None}],
                        })
                    if tc:
                        yield self._sse_bytes(chunk)
                        continue
                    if value is None:
                        continue
                    text_buffer.append(value)
                    full = "".join(text_buffer)
                    if DSML_OPEN in full:
                        # Split into pre-DSML clean text and DSML portion
                        idx = full.find(DSML_OPEN)
                        clean_part = full[:idx]
                        if clean_part and not dsml_active:
                            # Emit only unemitted portion of clean_part
                            emit = clean_part[emitted_index:]
                            if emit:
                                yield self._sse_bytes({
                                    "id": base_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model_name,
                                    "choices": [{"index": 0, "delta": {"content": emit}, "finish_reason": None}],
                                })
                                emitted_index += len(emit)
                            dsml_active = True
                            # Reset buffer to just the DSML part
                            text_buffer.clear()
                            text_buffer.append(full[idx:])
                    elif dsml_active:
                        # Still accumulating DSML block
                        pass
                    elif DSML_OPEN[:8] in full[-20:]:
                        # Partial match at end - hold back last few chars
                        safe_emit_len = max(0, len(full) - 20)
                        emit = full[emitted_index:safe_emit_len + emitted_index]
                        # Actually just emit up to len(full)-len(DSML_OPEN)+1
                        holdback = min(len(DSML_OPEN), 20)
                        emit_end = max(0, len(full) - holdback)
                        emit = full[emitted_index:emit_end]
                        if emit:
                            yield self._sse_bytes({
                                "id": base_id, "object": "chat.completion.chunk",
                                "created": created, "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": emit}, "finish_reason": None}],
                            })
                            emitted_index += len(emit)
                    else:
                        # No DSML detected, emit everything
                        emit = full[emitted_index:]
                        if emit:
                            yield self._sse_bytes({
                                "id": base_id, "object": "chat.completion.chunk",
                                "created": created, "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": emit}, "finish_reason": None}],
                            })
                            emitted_index += len(emit)
                if fr:
                    finish_reason = fr

            # Stream ended - process remaining buffered text
            remaining = "".join(text_buffer)
            if dsml_active or DSML_CLOSE in remaining or DSML_OPEN in remaining:
                clean_text, dsml_tool_calls = parse_dsml_content(remaining)
            else:
                clean_text = remaining[emitted_index:] if len(remaining) > emitted_index else ""
                dsml_tool_calls = []
            if clean_text:
                emit = clean_text[emitted_index:] if len(clean_text) > emitted_index else ""
                if emit:
                    yield self._sse_bytes({
                        "id": base_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": emit}, "finish_reason": None}],
                    })
            if dsml_tool_calls:
                yield self._sse_bytes({
                    "id": base_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {"tool_calls": [
                        {"index": i, "type": "function",
                         "id": tc.get("id",""),
                         "function": tc.get("function", {})}
                        for i, tc in enumerate(dsml_tool_calls)
                    ]}, "finish_reason": None}],
                })
                finish_reason = "tool_calls"
            final_event = {
                "id": base_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
            }
            if usage_data:
                final_event["usage"] = usage_data
            yield self._sse_bytes(final_event)
            yield b"data: [DONE]\n\n"

        return generate()

    def _json_to_v1_sse(self, data, model_name):
        """把上游一次性 JSON 应答转为 SSE 事件流,供流式客户端读取。"""
        data = dict(data)
        data["model"] = model_name
        base_id = data.get("id") or "chatcmpl-proxy"
        created = data.get("created") or int(time.time())

        def generate():
            for ci, choice in enumerate(data.get("choices") or []):
                message = choice.get("message") or {}
                delta = {"role": message.get("role", "assistant")}
                content = message.get("content")
                if content:
                    delta["content"] = content
                if message.get("reasoning_content"):
                    delta["reasoning_content"] = message["reasoning_content"]
                if message.get("tool_calls"):
                    delta["tool_calls"] = message["tool_calls"]
                yield self._sse_bytes({
                    "id": base_id, "object": "chat.completion.chunk", "created": created,
                    "model": model_name, "choices": [{"index": ci, "delta": delta, "finish_reason": None}],
                })
                final_event = {
                    "id": base_id, "object": "chat.completion.chunk", "created": created,
                    "model": model_name, "choices": [{"index": ci, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}],
                }
                if data.get("usage"):
                    final_event["usage"] = data["usage"]
                final = self._sse_bytes(final_event)
                yield final
            yield b"data: [DONE]\n\n"
        return generate()

    def _upstream_chat_call(self, provider, upstream_id, payload, stream=None):
        """统一上游调用入口: 根据 api_type 路由到 chat/completions 或 responses。
        返回 (response, actual_api_type)。对 responses 模型自动做 payload 转换。"""
        api_type = self.get_api_type(provider, upstream_id)
        if stream is not None:
            payload = dict(payload)
            payload["stream"] = stream
        if api_type == "responses":
            resp_payload = self.chat_to_responses_payload(payload)
            url = provider.responses_url
        else:
            resp_payload = payload
            url = provider.chat_url
        try:
            http_resp = self.upstream_request(provider, url, resp_payload)
        except UpstreamError as exc:
            # 对 responses 模型剥离 tools 重试
            if api_type == "responses" and self.config.retry_without_tools and exc.status in (500, 502, 503) and resp_payload.get("tools"):
                retry_payload = dict(resp_payload)
                retry_payload.pop("tools", None)
                try:
                    http_resp = self.upstream_request(provider, url, retry_payload)
                except UpstreamError:
                    raise exc
            else:
                raise
        return http_resp, api_type

    def v1_chat(self, body, stream=False):
        t0 = time.time()
        ollama_name = body.get("model", "")
        provider, upstream = self.resolve_model(ollama_name)
        if provider is None:
            raise ModelNotFoundError("model '%s' not found" % ollama_name)
        api_type = self.get_api_type(provider, upstream)
        url = provider.responses_url if api_type == "responses" else provider.chat_url
        self.log("v1/chat model=%s provider=%s url=%s stream=%s" % (
            ollama_name, provider.name, url, "true" if stream else "false"))
        new_body = dict(body)
        new_body["model"] = upstream
        client_stream = bool(stream)
        upstream_stream = self._resolved_upstream_stream(client_stream)
        if getattr(provider, "require_tools", False):
            if not new_body.get("tools"):
                new_body["tools"] = NOOP_TOOLS
        elif self.config.strip_tools:
            new_body.pop("tools", None)
            new_body.pop("tool_choice", None)
            new_body.pop("parallel_tool_calls", None)

        if api_type == "responses":
            resp_payload = self.chat_to_responses_payload(new_body)
            resp_payload["stream"] = upstream_stream
            try:
                resp = self.upstream_request(provider, provider.responses_url, resp_payload)
            except UpstreamError as r_exc:
                if resp_payload.get("tools") and self.config.retry_without_tools and r_exc.status in (500, 502, 503):
                    retry_payload = dict(resp_payload)
                    retry_payload.pop("tools", None)
                    try:
                        resp = self.upstream_request(provider, provider.responses_url, retry_payload)
                    except UpstreamError:
                        raise r_exc
                else:
                    raise
            if upstream_stream and client_stream:
                return self.responses_stream_to_v1_sse(resp, ollama_name)
            elif upstream_stream:
                raw = resp.read().decode("utf-8", "replace")
                resp.close()
                data = self._aggregate_responses_stream(raw)
                data["model"] = ollama_name
            else:
                raw = resp.read().decode("utf-8", "replace")
                resp.close()
                resp_data = json.loads(raw)
                data = self.responses_to_chat_response(resp_data, ollama_name)
        else:
            new_body["stream"] = upstream_stream
            try:
                resp = self.upstream_request(provider, provider.chat_url, new_body)
            except UpstreamError as exc:
                resp = self._retry_without_tools(provider, provider.chat_url, new_body, exc)
                if resp is None:
                    self.log("v1/chat 失败 model=%s provider=%s status=%s body=%s" % (
                        ollama_name, provider.name, exc.status, (exc.body or "")[:200]), level="error")
                    raise
            if upstream_stream:
                if client_stream:
                    return self._dsml_filtered_v1_stream(resp, ollama_name)
                data = self.aggregate_v1_stream(resp, ollama_name)
            else:
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
        if client_stream and not upstream_stream:
            return self._json_to_v1_sse(data, ollama_name)
        return data

    def _aggregate_responses_stream(self, sse_text):
        """聚合 Responses API SSE 文本为完整 Chat Completions JSON。"""
        text_parts = []
        tool_calls = []
        usage_data = {}
        for line in sse_text.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            try:
                event_data = json.loads(payload)
            except Exception:
                continue
            etype = event_data.get("type", "")
            if etype == "response.output_text.delta":
                text_parts.append(event_data.get("delta", ""))
            elif etype == "response.output_item.done":
                item = event_data.get("item", {})
                if item.get("type") == "function_call":
                    tool_calls.append(item)
            elif etype == "response.completed":
                fr = event_data.get("response", {}).get("usage", {}) or {}
                usage_data = {
                    "prompt_tokens": fr.get("input_tokens", 0),
                    "completion_tokens": fr.get("output_tokens", 0),
                    "total_tokens": fr.get("total_tokens", 0),
                }
        message = {"role": "assistant", "content": "".join(text_parts) or None}
        finish_reason = "stop"
        if tool_calls:
            message["tool_calls"] = [{
                "id": tc.get("id", f"call_{i}"),
                "type": "function",
                "function": {"name": tc.get("name", ""), "arguments": tc.get("arguments", "{}")},
            } for i, tc in enumerate(tool_calls)]
            finish_reason = "tool_calls"
        return {
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage_data,
        }


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "OllamaProxy/1.0"
    protocol_version = "HTTP/1.1"
    timeout = 120

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
    def log(self, message, level="info"):
        self.server.proxy.log(message, level)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as exc:
            self.log("客户端提前断开连接 %s" % exc, level="debug")
            self.close_connection = True

    def _debug_headers(self):
        wanted = ("User-Agent", "Content-Type", "Accept", "Authorization",
                  "X-Requested-With", "Origin", "Referer")
        values = []
        for name in wanted:
            value = self.headers.get(name)
            if not value:
                continue
            if name.lower() == "authorization":
                kind = value.split(" ", 1)[0] if " " in value else "token"
                value = kind + " ***"
            values.append("%s=%s" % (name, value))
        return " ".join(values)

    def _log_request(self):
        target = urllib.parse.urlparse(self.path)
        suffix = ("?" + target.query) if target.query else ""
        self.log("web 入口 %s %s%s %s" % (
            self.command, target.path, suffix, self._debug_headers()), level="debug")

    def _log_json_body(self, body):
        if not isinstance(body, dict):
            self.log("web 请求体类型=%s" % type(body).__name__, level="debug")
            return
        summary = {
            "model": body.get("model"),
            "stream": body.get("stream"),
            "messages": len(body["messages"]) if isinstance(body.get("messages"), list) else None,
            "prompt_bytes": len(str(body.get("prompt") or "").encode("utf-8")),
            "tools": len(body["tools"]) if isinstance(body.get("tools"), list) else None,
        }
        summary = {key: value for key, value in summary.items() if value is not None}
        self.log("web 请求体 %s" % json.dumps(summary, ensure_ascii=False, sort_keys=True), level="debug")

    @staticmethod
    def _sse_payload(line):
        if b"\"usage\"" not in line:
            return None
        if not line.lower().startswith(b"data:"):
            return None
        payload = line[5:].strip()
        if not payload or payload.lower() == b"[done]":
            return None
        try:
            value = json.loads(payload.decode("utf-8"))
            return value if isinstance(value, dict) else None
        except (ValueError, UnicodeDecodeError):
            return None

    def read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        limit = self.proxy.config.max_body_bytes
        if length < 0:
            length = 0
        if limit and limit > 0 and length > limit:
            raise BodyTooLargeError("请求体超过上限 %d 字节" % limit)
        if length > 0:
            return self.rfile.read(length)
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return self._read_chunked(limit)
        # No Content-Length and not chunked -> the request has no body. Return
        # immediately instead of blocking on a limit+1-byte read that stalls to
        # the connection timeout for the common empty-body POST case.
        return b""

    def _read_chunked(self, limit):
        data = bytearray()
        while True:
            size_line = self.rfile.readline()
            if not size_line or size_line in (b"\r\n", b"\n"):
                break
            try:
                size = int(size_line.strip().split(b";", 1)[0], 16)
            except ValueError:
                break
            if size == 0:
                # consume trailer block up to the terminating blank line
                while True:
                    trailer = self.rfile.readline()
                    if trailer in (b"\r\n", b"\n", b""):
                        break
                break
            if limit and limit > 0 and len(data) + size > limit:
                raise BodyTooLargeError("请求体超过上限 %d 字节" % limit)
            data.extend(self.rfile.read(size))
            self.rfile.readline()  # trailing CRLF after each chunk
        return bytes(data)

    def json_body(self):
        raw = self.read_body()
        self.log("web 请求体字节=%d" % len(raw), level="debug")
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            preview = raw[:200].decode("utf-8", "replace").replace("\n", "\\n")
            self.log("web 非法JSON预览=%r error=%s" % (preview, exc), level="error")
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

    def send_error_json(self, message, status=500, headers=None):
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/v1/"):
            obj = {"error": {"message": message, "type": "proxy_error", "code": status}}
        else:
            obj = {"error": message}
        self.send_json(obj, status=status, headers=headers)

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
        self._log_request()
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
        self._log_request()
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self.json_body()
            self._log_json_body(body)
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
                    event_count = 0
                    sse_buffer = bytearray()
                    usage = {}
                    client_done = False
                    try:
                        for raw in resp:
                            if not raw:
                                continue
                            self.write_chunk(raw)
                            sse_buffer.extend(raw)
                            while b"\n" in sse_buffer:
                                line, remaining = sse_buffer.split(b"\n", 1)
                                sse_buffer[:] = remaining
                                line = line.strip()
                                if not line:
                                    continue
                                event_count += 1
                                payload = self._sse_payload(line)
                                if payload and isinstance(payload.get("usage"), dict):
                                    usage = payload["usage"]
                                if line.lower() == b"data: [done]":
                                    client_done = True
                                    break
                            if client_done:
                                break
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
                        self.log("web 客户端断开 events=%d error=%s" % (event_count, exc), level="debug")
                    finally:
                        try:
                            resp.close()
                        except Exception:
                            pass
                        self.end_stream()
                        elapsed = time.time() - getattr(self, "_req_start", time.time())
                        self.log("web 流式结束 model=%s events=%d client_done=%s "
                                 "prompt_tokens=%s completion_tokens=%s total_tokens=%s elapsed=%.2fs" % (
                                     body.get("model"), event_count, client_done,
                                     usage.get("prompt_tokens", "-"), usage.get("completion_tokens", "-"),
                                     usage.get("total_tokens", "-"), elapsed), level="info")
                else:
                    self.send_json(self.proxy.v1_chat(body, stream=False))
            else:
                self.send_error_json("not found: " + path, 404)
        except BodyTooLargeError as exc:
            self.send_error_json(str(exc), 413, headers={"Connection": "close"})
            self.close_connection = True
        except ModelNotFoundError as exc:
            self.send_error_json(str(exc), 404)
        except UpstreamError as exc:
            self.send_error_json("upstream error: %s" % (exc.body or exc), 502)
        except ValueError as exc:
            self.send_error_json(str(exc), 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
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
    parser.add_argument("--version", action="version",
                        version="openai-ollama-proxy %s" % __version__)
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

    proxy.warm_models()
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
