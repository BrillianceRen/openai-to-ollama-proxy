#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
antigravity-ollama-proxy
========================

基于 Google Antigravity 服务 (cloudcode-pa.googleapis.com)
同时提供 Ollama API 接口与 OpenAI 兼容接口 (/v1/chat/completions 与 /v1/models)。
仅依赖 Python 标准库 (零外部依赖)。

端点支持
--------
GET  /                      返回 "Ollama is running"
GET  /api/version           返回模拟的 Ollama 版本号
GET  /api/tags              汇总 Antigravity 模型列表, 优先匹配 models/antigravity.json
POST /api/show              返回模型详情
POST /api/chat              Ollama 聊天对话 (支持流式 NDJSON)
POST /api/generate          Ollama 单轮生成 (支持流式 NDJSON)
GET  /api/ps                返回空运行列表 (兼容 Ollama 状态探测)
GET  /v1/models             OpenAI 兼容模型列表
POST /v1/chat/completions   OpenAI 兼容聊天接口 (支持流式 SSE)

用法
----
    # 启动代理
    python antigravity-ollama-proxy.py --config config.json
    python antigravity-ollama-proxy.py --port 11434 --project-id <PROJECT_ID> --refresh-token <REFRESH_TOKEN>

    # 首次登录引导获取 refresh_token
    python antigravity-ollama-proxy.py --login
"""

import argparse
import base64
import copy
import hashlib
import json
import logging
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Windows 控制台编码保护
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_MODEL = "gemini-3.7-flash-high"
OLLAMA_VERSION = "0.5.4"
__version__ = "1.0.0"

# Antigravity 专有客户端凭据与接口常量
# 默认使用 Google Antigravity CLI 桌面发行版内嵌的公用凭据（支持环境变量动态覆盖）
_CID_CODES = [107, 106, 109, 107, 106, 106, 108, 106, 108, 106, 111, 99, 107, 119, 46, 55, 50, 41, 41, 51, 52, 104, 50, 104, 107, 54, 57, 40, 63, 104, 105, 111, 44, 46, 53, 54, 53, 48, 50, 110, 61, 110, 106, 105, 63, 42, 116, 59, 42, 42, 41, 116, 61, 53, 53, 61, 54, 63, 47, 41, 63, 40, 57, 53, 52, 46, 63, 52, 46, 116, 57, 53, 55]
_SEC_CODES = [29, 21, 25, 9, 10, 2, 119, 17, 111, 98, 28, 13, 8, 110, 98, 108, 22, 62, 22, 16, 107, 55, 22, 24, 98, 41, 2, 25, 110, 32, 108, 43, 30, 27, 60]

ANTIGRAVITY_CLIENT_ID = os.environ.get(
    "ANTIGRAVITY_CLIENT_ID",
    "".join(chr(b ^ 0x5A) for b in _CID_CODES)
)
ANTIGRAVITY_CLIENT_SECRET = os.environ.get(
    "ANTIGRAVITY_CLIENT_SECRET",
    "".join(chr(b ^ 0x5A) for b in _SEC_CODES)
)
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
ANTIGRAVITY_CLI_VERSION = "1.1.12"
ANTIGRAVITY_CLI_PLATFORM = "windows/amd64"
ANTIGRAVITY_USER_AGENT = f"antigravity/cli/{ANTIGRAVITY_CLI_VERSION} {ANTIGRAVITY_CLI_PLATFORM}"
DEFAULT_API_URL = "https://daily-cloudcode-pa.googleapis.com"
TOKEN_URL = "https://oauth2.googleapis.com/token"

# 默认模型映射 (保持原样直通，用户可在 config.json 中自行配置特定别名)
DEFAULT_MODEL_MAPPINGS = {}


def now_iso():
    """Ollama 规范 RFC3339 时间戳"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def ndjson(obj):
    """序列化为 NDJSON 字节串"""
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


class Config:
    def __init__(self, data=None, base_dir=None):
        data = data or {}
        base_dir = base_dir or SCRIPT_DIR
        self.host = str(data.get("host", "127.0.0.1"))
        self.port = int(data.get("port", 11434))
        self.timeout = int(data.get("timeout", 300))
        self.models_dir = os.path.join(base_dir, str(data.get("models_dir", "models")))
        self.use_env_proxy = bool(data.get("use_env_proxy", True))

        ag_cfg = data.get("antigravity", {}) if isinstance(data.get("antigravity"), dict) else {}
        self.project_id = str(
            ag_cfg.get("project_id") or data.get("project_id") or os.environ.get("ANTIGRAVITY_PROJECT_ID") or ""
        ).strip()
        self.refresh_token = str(
            ag_cfg.get("refresh_token") or data.get("refresh_token") or os.environ.get("ANTIGRAVITY_REFRESH_TOKEN") or ""
        ).strip()
        self.api_url = str(
            ag_cfg.get("api_url") or data.get("api_url") or os.environ.get("ANTIGRAVITY_API_URL") or DEFAULT_API_URL
        ).rstrip("/")
        self.default_model = str(
            ag_cfg.get("default_model") or data.get("default_model") or DEFAULT_MODEL
        ).strip()
        self.filter_thinking = bool(
            ag_cfg.get("filter_thinking", data.get("filter_thinking", True))
        )
        self.enable_credit = bool(
            ag_cfg.get("enable_credit", data.get("enable_credit", False))
        )

        # 合并模型映射
        self.model_mappings = copy.deepcopy(DEFAULT_MODEL_MAPPINGS)
        custom_mappings = ag_cfg.get("model_mappings") or data.get("model_mappings") or {}
        if isinstance(custom_mappings, dict):
            for k, v in custom_mappings.items():
                self.model_mappings[k.strip().lower()] = str(v).strip()

    @classmethod
    def load(cls, path):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return cls(json.load(f), os.path.dirname(os.path.abspath(path)))
            except Exception as e:
                logging.warning(f"Failed to read config from {path}: {e}")
        return cls()


class AntigravityAuth:
    """单一账户 OAuth Token 管理器, 自动静默续期"""

    def __init__(self, config: Config):
        self.config = config
        self.access_token = ""
        self.expires_at = 0.0
        self.lock = threading.Lock()

    def get_access_token(self) -> str:
        with self.lock:
            # 如果当前 token 有效期大于 5 分钟, 直接使用
            if self.access_token and time.time() < (self.expires_at - 300):
                return self.access_token

            refresh_token = self.config.refresh_token
            if not refresh_token:
                raise ValueError("Antigravity refresh_token is missing. Please set it in config.json or use --login")

            # 向上游换取新 access_token
            logging.info("Refreshing Antigravity access token via Google OAuth...")
            params = urllib.parse.urlencode({
                "client_id": ANTIGRAVITY_CLIENT_ID,
                "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }).encode("utf-8")

            req = urllib.request.Request(
                TOKEN_URL,
                data=params,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST"
            )

            ctx = ssl.create_default_context()
            try:
                with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    new_token = data.get("access_token")
                    expires_in = int(data.get("expires_in", 3600))
                    if not new_token:
                        raise ValueError(f"Token response did not contain access_token: {data}")
                    self.access_token = new_token
                    self.expires_at = time.time() + expires_in
                    logging.info(f"Access token refreshed successfully (valid for {expires_in}s)")
                    return self.access_token
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                logging.error(f"Failed to refresh access token (HTTP {e.code}): {err_body}")
                raise RuntimeError(f"OAuth refresh failed: HTTP {e.code}: {err_body}")
            except Exception as e:
                logging.error(f"Error during OAuth refresh: {e}")
                raise

    @staticmethod
    def interactive_login(config_path: str):
        """交互式引导用户完成 Google OAuth 授权并保存 refresh_token"""
        print("\n" + "=" * 60)
        print("  Antigravity OAuth 一键登录引导")
        print("=" * 60)

        port = 51121
        redirect_uri = f"http://localhost:{port}"
        captured_code = {"code": None}

        class OAuthCallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # 静默请求日志

            def do_GET(self):
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                if "code" in params:
                    captured_code["code"] = params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write("""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>授权成功</title></head>
<body style='font-family:system-ui,-apple-system,sans-serif;text-align:center;padding-top:60px;'>
  <h2 style='color:#1a73e8;'>🎉 Google Antigravity 授权成功！</h2>
  <p style='color:#555;'>已成功接收 OAuth 授权凭据，您可以关闭此网页并返回终端查看。</p>
</body></html>""".encode("utf-8"))
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"No code parameter found in callback.")

        # 启动本地回调监听
        httpd = None
        server_running = False
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port), OAuthCallbackHandler)
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            server_running = True
        except Exception as e:
            logging.warning(f"无法在端口 {port} 启动自动监听 ({e})，将使用手动粘贴模式。")

        auth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth?"
            + urllib.parse.urlencode({
                "client_id": ANTIGRAVITY_CLIENT_ID,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": " ".join(ANTIGRAVITY_SCOPES),
                "access_type": "offline",
                "prompt": "consent",
            })
        )

        print("\n步骤 1: 请在浏览器中打开以下链接完成 Google 账号登录与授权:")
        print("-" * 60)
        print(auth_url)
        print("-" * 60)

        # 自动尝试唤起默认浏览器
        try:
            import webbrowser
            webbrowser.open(auth_url)
        except Exception:
            pass

        print("\n步骤 2: 等待授权回调...")
        if server_running:
            print(f"(已在 {redirect_uri} 启动本地监听，浏览器授权后将自动截获)")

        code = None
        # 等待最多 120 秒自动截获
        if server_running:
            start_t = time.time()
            while time.time() - start_t < 120:
                if captured_code["code"]:
                    code = captured_code["code"]
                    print("\n[✓] 已成功自动捕获授权回调！")
                    break
                time.sleep(0.5)

        if not code:
            print("\n如果浏览器未自动回调，请将授权后浏览器地址栏的完整 URL（或其中的 code=...）粘贴在此处:")
            user_input = input("URL 或 Code: ").strip()
            if "code=" in user_input:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(user_input).query)
                if "code" in qs:
                    code = qs["code"][0]
            elif user_input:
                code = user_input

        if httpd:
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception:
                pass

        if not code:
            print("未获取到有效授权码，退出登录流程。")
            return

        # 获取已有配置中的 project_id 作为默认提示
        default_pid = ""
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    _c = json.load(f)
                    default_pid = _c.get("antigravity", {}).get("project_id") or ""
            except Exception:
                pass

        if default_pid:
            print(f"\n步骤 3: 确认 Google Cloud Project ID (默认: {default_pid}):")
            pid_in = input(f"Project ID [{default_pid}]: ").strip()
            project_id = pid_in if pid_in else default_pid
        else:
            print("\n步骤 3: 请输入你的 Google Cloud Project ID (项目号或ID，通常在 Google Cloud 控制台首页可见):")
            project_id = input("Project ID: ").strip()

        print("\n正在向 Google 换取 Token...")
        params = urllib.parse.urlencode({
            "client_id": ANTIGRAVITY_CLIENT_ID,
            "client_secret": ANTIGRAVITY_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }).encode("utf-8")

        req = urllib.request.Request(
            TOKEN_URL,
            data=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST"
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                refresh_token = data.get("refresh_token")
                if not refresh_token:
                    print(f"未能获得 refresh_token，返回结果: {data}")
                    return

                # 保存至 config.json
                cfg_data = {}
                if os.path.isfile(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            cfg_data = json.load(f)
                    except Exception:
                        cfg_data = {}

                if "antigravity" not in cfg_data or not isinstance(cfg_data["antigravity"], dict):
                    cfg_data["antigravity"] = {}

                cfg_data["antigravity"]["refresh_token"] = refresh_token
                if project_id:
                    cfg_data["antigravity"]["project_id"] = project_id
                cfg_data["antigravity"]["default_model"] = cfg_data["antigravity"].get("default_model", DEFAULT_MODEL)

                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_data, f, indent=2, ensure_ascii=False)

                print("\n" + "=" * 60)
                print("授权成功！凭证已成功保存至: " + config_path)
                print("现在可以直接启动: python antigravity-ollama-proxy.py")
                print("=" * 60 + "\n")
        except Exception as e:
            print(f"换取 Token 失败: {e}")


def _clean_json_schema(schema):
    """清理 Gemini 不支持的 JSON Schema 关键字（例如 $schema, additionalProperties 等）"""
    if not isinstance(schema, dict):
        return schema
    disallowed = {"$schema", "$defs", "definitions", "title", "additionalProperties", "additional_properties"}
    cleaned = {}
    for k, v in schema.items():
        if k in disallowed:
            continue
        if k == "properties" and isinstance(v, dict):
            cleaned["properties"] = {pk: _clean_json_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            if isinstance(v, dict):
                cleaned["items"] = _clean_json_schema(v)
            elif isinstance(v, list):
                cleaned["items"] = [_clean_json_schema(x) for x in v]
            else:
                cleaned["items"] = v
        else:
            cleaned[k] = v
    return cleaned


def _convert_tools_to_gemini(tools):
    """将 OpenAI/Ollama 格式的 tools 列表转换为 Gemini functionDeclarations"""
    if not tools:
        return None
    declarations = []
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        decl = {
            "name": name,
            "description": fn.get("description", ""),
            "parameters": _clean_json_schema(params)
        }
        declarations.append(decl)
    if declarations:
        return [{"functionDeclarations": declarations}]
    return None


DEFAULT_THOUGHT_SIGNATURE = (
    "EvEBCu4BARFNMg8IoyEpW4Ncw4TCydxeOZRHfZ72SXhhgdmBAXKyh7DGwX20vIwXJN8Q753cWiqedRdR4grXWZQO"
    "INnZZsR9uWuGaFYVFHcpo9yjKY1f7aX3UqF279DLJPab75HAoA2ZRDasCJUvEzEBTUPZaPydcenQhAhtRIWl8ZUP"
    "2/mU3o4nPch7wM1RGMP2k5K8bNY6utiVRsyN6K5Ih1vEZaes7VfRZYrjSvJhmHNm1Uhel4JjjKDFsr8kKcVHmMK+"
    "l58vQKYJFpFNeLpGgmmzLZmSfsK4Rxy9oKG5KS1R+U3u+CgMUnmn3Dl1hWsAfw=="
)


class AntigravityClient:
    """处理与上游 Antigravity RPC 的封装与通信"""

    def __init__(self, config: Config, auth: AntigravityAuth):
        self.config = config
        self.auth = auth
        self.lock = threading.Lock()
        self._thought_signatures: dict[str, str] = {}
        self._latest_valid_signature = DEFAULT_THOUGHT_SIGNATURE
        self._sig_cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "thought_signatures_antigravity.json")
        self._load_signatures_from_disk()

    def _load_signatures_from_disk(self):
        try:
            if os.path.exists(self._sig_cache_file):
                with open(self._sig_cache_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    if isinstance(raw, dict):
                        for k, val in raw.items():
                            if isinstance(val, str) and len(val) > 20:
                                self._thought_signatures[k] = val
                                self._latest_valid_signature = val
        except Exception:
            pass

    def _save_signatures_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self._sig_cache_file), exist_ok=True)
            items = dict(list(self._thought_signatures.items())[-500:])
            with open(self._sig_cache_file, "w", encoding="utf-8") as f:
                json.dump(items, f)
        except Exception:
            pass

    def cache_thought_signature(self, name: str, args: Any, sig: str, fn_id: str = ""):
        if not sig or not isinstance(sig, str):
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

    def get_thought_signature(self, name: str, args: Any, fn_id: str = "") -> str:
        with self.lock:
            if fn_id and f"id:{fn_id}" in self._thought_signatures:
                return self._thought_signatures[f"id:{fn_id}"]
            args_key = json.dumps(args, sort_keys=True, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
            if f"fn:{name}:{args_key}" in self._thought_signatures:
                return self._thought_signatures[f"fn:{name}:{args_key}"]
            if f"latest:{name}" in self._thought_signatures:
                return self._thought_signatures[f"latest:{name}"]
            return self._latest_valid_signature

    def resolve_model(self, requested_model: str) -> str:
        """模型名称解析（原样保留，仅剥离 :latest 标签，用户若在配置中声明了映射则遵循映射）"""
        name = (requested_model or "").strip()
        if name.lower().endswith(":latest"):
            name = name[:-7]
        if not name:
            return self.config.default_model
        return self.config.model_mappings.get(name, name)

    def _extract_first_user_text(self, contents: list) -> str:
        for c in contents:
            if isinstance(c, dict) and c.get("role") == "user":
                for p in c.get("parts", []):
                    if isinstance(p, dict) and p.get("text"):
                        return str(p["text"])
        return ""

    def wrap_antigravity_payload(
        self,
        contents: list,
        system_instruction: str = "",
        model: str = "",
        options: dict = None,
        tools: list = None
    ) -> dict:
        """
        包装为 Antigravity 专有 RPC 请求体 (对齐 Antigravity CLI 规范)
        """
        model = self.resolve_model(model)
        # Antigravity 专有 RPC 请求体中，免费层/个人开发者专有项目名为 aicode-consumers
        project_id = self.config.project_id
        if not project_id or project_id.isdigit() or project_id.startswith("gen-lang-client"):
            project_id = "aicode-consumers"

        # 生成稳健的 sessionId
        first_text = self._extract_first_user_text(contents)
        if first_text:
            digest = hashlib.sha256(first_text.encode("utf-8")).digest()
            val = int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF
            session_id = f"-{val}"
        else:
            session_id = f"-{uuid.uuid4().int % 9_000_000_000_000_000_000}"

        trajectory_id = str(uuid.uuid4())
        request_id = f"agent/{uuid.uuid4()}/{int(time.time() * 1000)}/{trajectory_id}/1"

        used_claude = "claude" in model.lower()
        labels = {
            "last_step_index": "1",
            "model_enum": model,
            "trajectory_id": trajectory_id,
            "used_claude": str(used_claude).lower(),
            "used_claude_conservative": str(used_claude).lower(),
        }

        inner_request = {
            "contents": contents,
            "sessionId": session_id,
            "labels": labels,
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "VALIDATED"
                }
            }
        }

        gemini_tools = _convert_tools_to_gemini(tools)
        if gemini_tools:
            inner_request["tools"] = gemini_tools

        if system_instruction:
            inner_request["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # 选项与生成控制
        opts = options or {}
        gen_config = {}
        if "temperature" in opts:
            gen_config["temperature"] = float(opts["temperature"])
        if "top_p" in opts:
            gen_config["topP"] = float(opts["top_p"])
        if "max_tokens" in opts or "max_output_tokens" in opts:
            gen_config["maxOutputTokens"] = int(opts.get("max_output_tokens") or opts.get("max_tokens"))
        if gen_config:
            inner_request["generationConfig"] = gen_config

        payload = {
            "project": project_id,
            "requestId": request_id,
            "request": inner_request,
            "model": model,
            "userAgent": "antigravity",
            "requestType": "agent",
        }
        if self.config.enable_credit:
            payload["enabledCreditTypes"] = ["GOOGLE_ONE_AI"]
        return payload

    def stream_generate(self, payload: dict):
        """
        向上游发起流式请求并生成 (text, thought) 事件生成器
        自动支持候选端点 (daily-cloudcode-pa / cloudcode-pa) 故障与 429 配额自动故障转移
        """
        access_token = self.auth.get_access_token()

        candidate_urls = [self.config.api_url]
        for u in ("https://daily-cloudcode-pa.googleapis.com", "https://cloudcode-pa.googleapis.com"):
            if u not in candidate_urls:
                candidate_urls.append(u)

        ctx = ssl.create_default_context()
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        resp = None
        last_error = None

        for base_url in candidate_urls:
            url = f"{base_url}/v1internal:streamGenerateContent?alt=sse"
            headers = {
                "User-Agent": ANTIGRAVITY_USER_AGENT,
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "requestType": "agent",
                "requestId": f"req-{uuid.uuid4()}",
            }

            logging.debug(f"[UPSTREAM] Dispatching RPC to: {url}")
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                req_inner = payload.get("request", {})
                logging.debug(f"[UPSTREAM] SessionID: {req_inner.get('sessionId')}")
                logging.debug(f"[UPSTREAM] Contents count: {len(req_inner.get('contents', []))}")
                logging.debug(f"[FULL UPSTREAM RPC PAYLOAD]:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")

            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            try:
                resp = urllib.request.urlopen(req, timeout=self.config.timeout, context=ctx)
                logging.debug(f"[UPSTREAM] Connected successfully to {base_url}. Upstream HTTP Status: {resp.status}")
                break
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {e.code}: {err_msg}")
                logging.warning(f"[UPSTREAM] {base_url} returned HTTP {e.code} for model '{payload.get('model')}': {err_msg[:120]}")
                if e.code in (429, 503) and base_url != candidate_urls[-1]:
                    logging.info(f"[UPSTREAM] Attempting next candidate endpoint due to HTTP {e.code}...")
                    continue
                logging.error(f"[UPSTREAM ERROR] HTTP {e.code} for model '{payload.get('model')}': {err_msg}")
                raise last_error from e
            except Exception as e:
                last_error = e
                logging.warning(f"[UPSTREAM] Connection to {base_url} failed: {e}")
                if base_url != candidate_urls[-1]:
                    continue
                logging.error(f"[UPSTREAM ERROR] All candidate connections failed: {e}")
                raise

        def event_generator():
            try:
                buffer = ""
                chunk_index = 0
                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            item = json.loads(data_str)
                        except Exception:
                            continue

                        # Antigravity 响应可能嵌套在 response 键中
                        resp_obj = item.get("response") if isinstance(item.get("response"), dict) else item
                        candidates = resp_obj.get("candidates", [])
                        for cand in candidates:
                            parts = cand.get("content", {}).get("parts", [])
                            for part in parts:
                                if not isinstance(part, dict):
                                    continue
                                fc = part.get("functionCall")
                                text_val = part.get("text", "")
                                sig = part.get("thoughtSignature") or part.get("thought_signature")
                                is_thought = bool(part.get("thought")) or (sig and not fc)

                                if fc:
                                    chunk_index += 1
                                    fn_name = fc.get("name", "")
                                    fn_args = fc.get("args", {})
                                    fn_id = fc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                                    if sig:
                                        self.cache_thought_signature(fn_name, fn_args, sig, fn_id)
                                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                                        args_prev = json.dumps(fn_args, ensure_ascii=False)[:80]
                                        sig_info = f" | sig={sig[:16]}..." if sig else ""
                                        logging.debug(f"[UPSTREAM CHUNK] #{chunk_index} [FUNCTION_CALL] {fn_name}({args_prev}){sig_info}")
                                    yield ("", "", {"name": fn_name, "args": fn_args, "id": fn_id, "thought_signature": sig})
                                elif text_val:
                                    chunk_index += 1
                                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                                        preview = text_val.replace("\n", "\\n")[:60]
                                        tag = "THOUGHT" if is_thought else "TEXT"
                                        logging.debug(f"[UPSTREAM CHUNK] #{chunk_index} [{tag}] {preview}")
                                    if is_thought:
                                        yield ("", text_val, None)
                                    else:
                                        yield (text_val, "", None)
            finally:
                resp.close()

        return event_generator()


def _unpack_chunk(item):
    """安全解包生成器返回项，自适应兼容 (text, thought) 与 (text, thought, tool_call)"""
    if isinstance(item, (list, tuple)):
        text = item[0] if len(item) > 0 else ""
        thought = item[1] if len(item) > 1 else ""
        tool_call = item[2] if len(item) > 2 else None
        return text, thought, tool_call
    return str(item), "", None


class Handler(BaseHTTPRequestHandler):
    """同时兼容 Ollama 与 OpenAI 端点的请求分发处理器"""

    protocol_version = "HTTP/1.1"

    @property
    def config(self) -> Config:
        return self.server.config

    @property
    def client(self) -> AntigravityClient:
        return self.server.client

    def log_message(self, format, *args):
        # 将原始 HTTP 日志设为 DEBUG，避免与结构化 [REQUEST]/[RESPONSE] 日志重复
        logging.debug("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))

    def send_json(self, status_code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        if logging.getLogger().isEnabledFor(logging.DEBUG) and self.path != "/api/ps":
            logging.debug(f"[FULL CLIENT RESPONSE JSON] {status_code} for {self.command} {self.path}:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def read_body_json(self) -> dict:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len <= 0:
            return {}
        body = self.rfile.read(content_len)
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
        except Exception as e:
            logging.error(f"[ERROR] Failed to parse JSON request body: {e}")
            return {}
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[FULL CLIENT REQUEST BODY] {self.command} {self.path}:\n{json.dumps(data, ensure_ascii=False, indent=2)}")
        return data

    def do_GET(self):
        try:
            url_path = urllib.parse.urlparse(self.path).path

            # Ollama 状态探测（高频心跳，debug输出）
            if url_path == "/api/ps":
                logging.debug(f"[REQUEST] GET /api/ps (polling) from {self.client_address[0]}")
                self.send_json(200, {"models": []})
                return

            # Ollama 探测根端点
            if url_path == "/":
                logging.info(f"[REQUEST] GET / (health check) from {self.client_address[0]}")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"Ollama is running\n")
                return

            # Ollama 版本
            if url_path == "/api/version":
                logging.info(f"[REQUEST] GET /api/version from {self.client_address[0]}")
                self.send_json(200, {"version": OLLAMA_VERSION})
                return

            # Ollama 模型列表
            if url_path == "/api/tags":
                logging.info(f"[REQUEST] GET /api/tags from {self.client_address[0]}")
                self.handle_api_tags()
                return

            # OpenAI 模型列表
            if url_path == "/v1/models":
                logging.info(f"[REQUEST] GET /v1/models from {self.client_address[0]}")
                self.handle_v1_models()
                return

            logging.warning(f"[REQUEST] GET {url_path} 404 Not Found from {self.client_address[0]}")
            self.send_json(404, {"error": f"Endpoint not found: {url_path}"})
        except Exception as e:
            logging.exception(f"[ERROR] Unhandled exception in GET {self.path}: {e}")
            try:
                self.send_json(500, {"error": f"Internal server error: {str(e)}"})
            except Exception:
                pass

    def do_POST(self):
        try:
            url_path = urllib.parse.urlparse(self.path).path

            if url_path == "/api/show":
                self.handle_api_show()
                return

            # Ollama 聊天与生成
            if url_path == "/api/chat":
                self.handle_api_chat()
                return

            if url_path == "/api/generate":
                self.handle_api_generate()
                return

            # OpenAI 兼容聊天接口
            if url_path == "/v1/chat/completions":
                self.handle_v1_chat_completions()
                return

            logging.warning(f"[REQUEST] POST {url_path} 404 Not Found from {self.client_address[0]}")
            self.send_json(404, {"error": f"Endpoint not found: {url_path}"})
        except Exception as e:
            logging.exception(f"[ERROR] Unhandled exception in POST {self.path}: {e}")
            try:
                self.send_json(500, {"error": f"Internal server error: {str(e)}"})
            except Exception:
                pass

    # ======================== 模型列表与详情 ========================

    def _load_antigravity_models(self) -> list:
        models_file = os.path.join(self.config.models_dir, "antigravity.json")
        if os.path.isfile(models_file):
            try:
                with open(models_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("models", [])
            except Exception as e:
                logging.warning(f"Failed to read models file {models_file}: {e}")

        # 默认回退模型列表
        now = now_iso()
        return [
            {
                "name": "claude-sonnet-4-5",
                "model": "claude-sonnet-4-5",
                "size": 200000000000,
                "digest": "agclaudesonnet",
                "modified_at": now,
                "details": {"family": "claude", "parameter_size": "200B"}
            },
            {
                "name": "gemini-2.5-pro",
                "model": "gemini-2.5-pro",
                "size": 200000000000,
                "digest": "aggemini25pro",
                "modified_at": now,
                "details": {"family": "gemini", "parameter_size": "200B"}
            },
            {
                "name": "gemini-2.5-flash",
                "model": "gemini-2.5-flash",
                "size": 70000000000,
                "digest": "aggemini25flash",
                "modified_at": now,
                "details": {"family": "gemini", "parameter_size": "70B"}
            },
            {
                "name": "antigravity-claude",
                "model": "claude-sonnet-4-5",
                "size": 200000000000,
                "digest": "agaliasclaude",
                "modified_at": now,
                "details": {"family": "claude", "parameter_size": "200B"}
            }
        ]

    def handle_api_tags(self):
        raw_models = self._load_antigravity_models()
        tags_models = []
        for m in raw_models:
            tags_models.append({
                "name": m.get("name", ""),
                "model": m.get("model", m.get("name", "")),
                "modified_at": m.get("modified_at") or now_iso(),
                "size": m.get("size", 0),
                "digest": m.get("digest", ""),
                "details": {
                    "parent_model": "",
                    "format": "",
                    "family": "",
                    "families": None,
                    "parameter_size": "",
                    "quantization_level": ""
                }
            })
        self.send_json(200, {"models": tags_models})

    def handle_v1_models(self):
        models = self._load_antigravity_models()
        openai_models = []
        for m in models:
            m_id = m.get("name") or m.get("model")
            openai_models.append({
                "id": m_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "antigravity",
                "permission": [],
                "root": m_id,
                "parent": None
            })
        self.send_json(200, {"object": "list", "data": openai_models})

    def handle_api_show(self):
        req = self.read_body_json()
        target = (req.get("model") or req.get("name") or "").strip()
        if target.endswith(":latest"):
            target = target[:-7]
        logging.info(f"[REQUEST] POST /api/show | model='{target}' | client={self.client_address[0]}")
        models = self._load_antigravity_models()
        found = None
        for m in models:
            m_name = m.get("name", "")
            m_model = m.get("model", "")
            if m_name == target or m_model == target or m_name.split(":")[0] == target:
                found = m
                break
        if not found and models:
            resolved = self.client.resolve_model(target)
            for m in models:
                if m.get("name") == resolved or m.get("model") == resolved:
                    found = m
                    break
        if not found and models:
            found = models[0]

        if not found:
            self.send_json(404, {"error": f"model '{target}' not found"})
            return

        model_name = found.get("name", target)
        arch = found.get("model_info", {}).get("general.architecture", "")
        param_count = found.get("model_info", {}).get("general.parameter_count") or found.get("size", 0)

        # 严格遵循官方 Ollama /api/show 响应规范
        resp = {
            "capabilities": found.get("capabilities", ["completion", "tools"]),
            "details": {
                "parent_model": model_name,
                "format": found.get("details", {}).get("format", ""),
                "family": arch or found.get("details", {}).get("family", ""),
                "families": None,
                "parameter_size": str(param_count) if param_count else "",
                "quantization_level": found.get("details", {}).get("quantization_level", "")
            },
            "model_info": found.get("model_info", {}),
            "modified_at": found.get("modified_at") or now_iso()
        }
        self.send_json(200, resp)

    # ======================== 协议消息转换 ========================

    def _convert_messages_to_contents(self, messages: list):
        """将通用/Ollama/OpenAI 的 messages 转换为 Google Gemini 格式的 contents 与 systemInstruction"""
        system_instructions = []
        contents = []
        tool_id_to_name = {}

        for msg in messages or []:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str) and content:
                    system_instructions.append(content)
                continue

            parts = []

            # 1. 工具执行结果返回 (OpenAI role: tool / function)
            if role in ("tool", "function"):
                name = msg.get("name") or tool_id_to_name.get(msg.get("tool_call_id", ""), "tool_response")
                resp_content = content
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        resp_content = parsed if isinstance(parsed, dict) else {"result": parsed}
                    except Exception:
                        resp_content = {"result": content}
                elif not isinstance(content, dict):
                    resp_content = {"result": content}
                parts.append({
                    "functionResponse": {
                        "name": name,
                        "response": resp_content
                    }
                })
                contents.append({"role": "user", "parts": parts})
                continue

            gemini_role = "model" if role in ("assistant", "model") else "user"

            # 2. Assistant 历史生成的工具调用 (tool_calls)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                fn_name = fn.get("name") or tc.get("name") or ""
                fn_id = tc.get("id") or ""
                if fn_id and fn_name:
                    tool_id_to_name[fn_id] = fn_name
                args = fn.get("arguments", {}) if "arguments" in fn else tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                if not isinstance(args, dict):
                    args = {"value": args}
                sig = ""
                if hasattr(self, "client") and self.client and hasattr(self.client, "get_thought_signature"):
                    sig = self.client.get_thought_signature(fn_name, args, fn_id)
                elif hasattr(self, "server") and hasattr(self.server, "client") and hasattr(self.server.client, "get_thought_signature"):
                    sig = self.server.client.get_thought_signature(fn_name, args, fn_id)
                if not sig:
                    sig = DEFAULT_THOUGHT_SIGNATURE

                fc_part = {
                    "functionCall": {
                        "name": fn_name,
                        "args": args
                    },
                    "thoughtSignature": sig
                }
                parts.append(fc_part)

            # 3. 文本内容与多模态图片
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            url = (item.get("image_url") or {}).get("url", "")
                            if url.startswith("data:"):
                                mime = url.split(";")[0].replace("data:", "")
                                b64 = url.split(",", 1)[1] if "," in url else ""
                                parts.append({"inlineData": {"mimeType": mime, "data": b64}})
                    elif isinstance(item, str):
                        parts.append({"text": item})

            # Ollama 风格传入的 images
            for img in msg.get("images") or []:
                raw_b64 = str(img).strip()
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": raw_b64}})

            if not parts:
                parts.append({"text": ""})

            contents.append({"role": gemini_role, "parts": parts})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        sys_text = "\n\n".join(system_instructions)
        return contents, sys_text

    # ======================== Ollama 端点逻辑 ========================

    def handle_api_chat(self):
        t0 = time.time()
        req = self.read_body_json()
        model_name = req.get("model", "")
        messages = req.get("messages", [])
        is_stream = req.get("stream", True)
        options = req.get("options", {})
        tools = req.get("tools")

        resolved_model = self.client.resolve_model(model_name)
        logging.info(
            f"[REQUEST] POST /api/chat | model='{model_name}' (upstream='{resolved_model}') | "
            f"stream={is_stream} | msgs={len(messages)} | tools={len(tools) if tools else 0} | client={self.client_address[0]}"
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for m in messages:
                if m.get("role") == "user":
                    preview = str(m.get("content", ""))[:120].replace("\n", "\\n")
                    logging.debug(f"[DEBUG] User prompt preview: {preview}")
                    break
            logging.debug(f"[DEBUG] Client options: {options}")

        contents, sys_text = self._convert_messages_to_contents(messages)
        payload = self.client.wrap_antigravity_payload(
            contents=contents,
            system_instruction=sys_text,
            model=model_name,
            options=options,
            tools=tools
        )

        try:
            stream_gen = self.client.stream_generate(payload)
        except Exception as e:
            logging.error(f"[ERROR] POST /api/chat failed | model='{model_name}': {e}")
            self.send_json(500, {"error": f"Antigravity upstream call failed: {str(e)}"})
            return

        filter_thinking = self.config.filter_thinking
        total_chunks = 0
        total_chars = 0

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()

            thinking_started = False
            done_reason = "stop"
            accum_text = []
            accum_thought = []
            accum_tools = []
            try:
                for item in stream_gen:
                    text_chunk, thought_chunk, tool_call = _unpack_chunk(item)
                    total_chunks += 1
                    total_chars += len(text_chunk) + len(thought_chunk)
                    # 工具调用处理
                    if tool_call:
                        done_reason = "tool_calls"
                        accum_tools.append(tool_call)
                        chunk_data = ndjson({
                            "model": model_name,
                            "created_at": now_iso(),
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [{
                                    "function": {
                                        "name": tool_call.get("name", ""),
                                        "arguments": tool_call.get("args", {})
                                    }
                                }]
                            },
                            "done": False
                        })
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk_data), chunk_data))
                        self.wfile.flush()
                    # 思考链处理
                    elif thought_chunk:
                        accum_thought.append(thought_chunk)
                        if not filter_thinking:
                            chunk_msg = thought_chunk
                            if not thinking_started:
                                chunk_msg = "<think>\n" + chunk_msg
                                thinking_started = True
                            chunk_data = ndjson({
                                "model": model_name,
                                "created_at": now_iso(),
                                "message": {"role": "assistant", "content": chunk_msg},
                                "done": False
                            })
                            self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk_data), chunk_data))
                            self.wfile.flush()
                    # 正文回答处理
                    elif text_chunk:
                        accum_text.append(text_chunk)
                        chunk_msg = text_chunk
                        if thinking_started:
                            chunk_msg = "\n</think>\n" + chunk_msg
                            thinking_started = False

                        chunk_data = ndjson({
                            "model": model_name,
                            "created_at": now_iso(),
                            "message": {"role": "assistant", "content": chunk_msg},
                            "done": False
                        })
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk_data), chunk_data))
                        self.wfile.flush()

                # 结束包
                if thinking_started:
                    closing = ndjson({
                        "model": model_name,
                        "created_at": now_iso(),
                        "message": {"role": "assistant", "content": "\n</think>\n"},
                        "done": False
                    })
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(closing), closing))

                final_data = ndjson({
                    "model": model_name,
                    "created_at": now_iso(),
                    "done": True,
                    "done_reason": done_reason,
                    "total_duration": int((time.time() - t0) * 1e9),
                    "prompt_eval_count": len(messages),
                    "eval_count": total_chunks
                })
                self.wfile.write(b"%X\r\n%s\r\n" % (len(final_data), final_data))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                elapsed = time.time() - t0
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        f"[FULL CLIENT STREAMED RESPONSE] POST /api/chat | model='{model_name}' | done_reason='{done_reason}':\n"
                        + json.dumps({
                            "text": "".join(accum_text),
                            "thoughts": "".join(accum_thought),
                            "tool_calls": accum_tools,
                            "done_reason": done_reason
                        }, ensure_ascii=False, indent=2)
                    )
                logging.info(
                    f"[RESPONSE] POST /api/chat 200 OK | model='{model_name}' | "
                    f"chunks={total_chunks} | chars={total_chars} | elapsed={elapsed:.2f}s"
                )
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                logging.debug(f"Client {self.client_address[0]} closed connection early.")
                return
            except Exception as stream_err:
                logging.exception(f"[ERROR] Stream error in POST /api/chat: {stream_err}")
                try:
                    err_chunk = ndjson({"error": str(stream_err), "done": True})
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(err_chunk), err_chunk))
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
                return
        else:
            # 非流式聚合
            full_text = []
            full_thought = []
            tool_calls = []
            for item in stream_gen:
                text_chunk, thought_chunk, tool_call = _unpack_chunk(item)
                if text_chunk:
                    full_text.append(text_chunk)
                if thought_chunk:
                    full_thought.append(thought_chunk)
                if tool_call:
                    tool_calls.append({
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": tool_call.get("args", {})
                        }
                    })

            out_content = "".join(full_text)
            if full_thought and not filter_thinking:
                out_content = f"<think>\n{''.join(full_thought)}\n</think>\n" + out_content

            msg_obj = {"role": "assistant", "content": out_content}
            done_reason = "stop"
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
                done_reason = "tool_calls"

            self.send_json(200, {
                "model": model_name,
                "created_at": now_iso(),
                "message": msg_obj,
                "done": True,
                "done_reason": done_reason
            })
            elapsed = time.time() - t0
            logging.info(
                f"[RESPONSE] POST /api/chat 200 OK (non-stream) | model='{model_name}' | "
                f"chars={len(out_content)} | elapsed={elapsed:.2f}s"
            )

    def handle_api_generate(self):
        t0 = time.time()
        req = self.read_body_json()
        model_name = req.get("model", "")
        prompt = req.get("prompt", "")
        system = req.get("system", "")
        is_stream = req.get("stream", True)
        images = req.get("images", [])

        resolved_model = self.client.resolve_model(model_name)
        logging.info(
            f"[REQUEST] POST /api/generate | model='{model_name}' (upstream='{resolved_model}') | "
            f"stream={is_stream} | prompt_len={len(prompt)} | client={self.client_address[0]}"
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            preview = prompt[:120].replace("\n", "\\n")
            logging.debug(f"[DEBUG] Prompt preview: {preview}")

        msg = {"role": "user", "content": prompt, "images": images}
        messages = [{"role": "system", "content": system}] if system else []
        messages.append(msg)

        contents, sys_text = self._convert_messages_to_contents(messages)
        payload = self.client.wrap_antigravity_payload(
            contents=contents,
            system_instruction=sys_text,
            model=model_name,
            options=req.get("options", {})
        )

        try:
            stream_gen = self.client.stream_generate(payload)
        except Exception as e:
            logging.error(f"[ERROR] POST /api/generate failed | model='{model_name}': {e}")
            self.send_json(500, {"error": f"Antigravity upstream call failed: {str(e)}"})
            return

        total_chunks = 0
        total_chars = 0

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()

            for text_chunk, *_ in stream_gen:
                if text_chunk:
                    total_chunks += 1
                    total_chars += len(text_chunk)
                    chunk_data = ndjson({
                        "model": model_name,
                        "created_at": now_iso(),
                        "response": text_chunk,
                        "done": False
                    })
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk_data), chunk_data))
                    self.wfile.flush()

            final_data = ndjson({
                "model": model_name,
                "created_at": now_iso(),
                "response": "",
                "done": True,
                "done_reason": "stop"
            })
            self.wfile.write(b"%X\r\n%s\r\n" % (len(final_data), final_data))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            elapsed = time.time() - t0
            logging.info(
                f"[RESPONSE] POST /api/generate 200 OK | model='{model_name}' | "
                f"chunks={total_chunks} | chars={total_chars} | elapsed={elapsed:.2f}s"
            )
        else:
            full_text = [t for t, *_ in stream_gen if t]
            out_str = "".join(full_text)
            self.send_json(200, {
                "model": model_name,
                "created_at": now_iso(),
                "response": out_str,
                "done": True,
                "done_reason": "stop"
            })
            elapsed = time.time() - t0
            logging.info(
                f"[RESPONSE] POST /api/generate 200 OK (non-stream) | model='{model_name}' | "
                f"chars={len(out_str)} | elapsed={elapsed:.2f}s"
            )

    # ======================== OpenAI 兼容端点逻辑 ========================

    def handle_v1_chat_completions(self):
        t0 = time.time()
        req = self.read_body_json()
        model_name = req.get("model", "")
        messages = req.get("messages", [])
        is_stream = bool(req.get("stream", False))
        tools = req.get("tools")

        resolved_model = self.client.resolve_model(model_name)
        logging.info(
            f"[REQUEST] POST /v1/chat/completions | model='{model_name}' (upstream='{resolved_model}') | "
            f"stream={is_stream} | msgs={len(messages)} | tools={len(tools) if tools else 0} | client={self.client_address[0]}"
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for m in messages:
                if m.get("role") == "user":
                    preview = str(m.get("content", ""))[:120].replace("\n", "\\n")
                    logging.debug(f"[DEBUG] User prompt preview: {preview}")
                    break

        options = {}
        for k in ("temperature", "top_p", "max_tokens"):
            if k in req:
                options[k] = req[k]

        contents, sys_text = self._convert_messages_to_contents(messages)
        payload = self.client.wrap_antigravity_payload(
            contents=contents,
            system_instruction=sys_text,
            model=model_name,
            options=options,
            tools=tools
        )

        try:
            stream_gen = self.client.stream_generate(payload)
        except Exception as e:
            logging.error(f"[ERROR] POST /v1/chat/completions failed | model='{model_name}': {e}")
            self.send_json(500, {
                "error": {
                    "message": f"Antigravity upstream call failed: {str(e)}",
                    "type": "upstream_error",
                    "code": 500
                }
            })
            return

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())
        total_chunks = 0
        total_chars = 0

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()

            tool_calls_accum = []
            finish_reason = "stop"
            accum_text = []
            accum_thought = []
            try:
                for item in stream_gen:
                    text_chunk, thought_chunk, tool_call = _unpack_chunk(item)
                    delta = {}
                    if text_chunk:
                        delta["content"] = text_chunk
                        accum_text.append(text_chunk)
                        total_chars += len(text_chunk)
                    if thought_chunk:
                        delta["reasoning_content"] = thought_chunk
                        accum_thought.append(thought_chunk)
                        total_chars += len(thought_chunk)
                    if tool_call:
                        fn_name = tool_call.get("name", "")
                        fn_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        args_obj = tool_call.get("args", {})
                        args_str = json.dumps(args_obj, ensure_ascii=False) if isinstance(args_obj, (dict, list)) else str(args_obj)
                        delta["tool_calls"] = [{
                            "index": len(tool_calls_accum),
                            "id": fn_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": args_str
                            }
                        }]
                        tool_calls_accum.append(tool_call)
                        finish_reason = "tool_calls"

                    if delta:
                        total_chunks += 1
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None
                                }
                            ]
                        }
                        sse_line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(sse_line), sse_line))
                        self.wfile.flush()

                # 最终 finish 块
                fin_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }
                    ]
                }
                fin_line = f"data: {json.dumps(fin_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                self.wfile.write(b"%X\r\n%s\r\n" % (len(fin_line), fin_line))

                done_line = b"data: [DONE]\n\n"
                self.wfile.write(b"%X\r\n%s\r\n" % (len(done_line), done_line))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                elapsed = time.time() - t0
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        f"[FULL CLIENT STREAMED RESPONSE] POST /v1/chat/completions | model='{model_name}' | finish_reason='{finish_reason}':\n"
                        + json.dumps({
                            "text": "".join(accum_text),
                            "reasoning_content": "".join(accum_thought),
                            "tool_calls": tool_calls_accum,
                            "finish_reason": finish_reason
                        }, ensure_ascii=False, indent=2)
                    )
                logging.info(
                    f"[RESPONSE] POST /v1/chat/completions 200 OK | model='{model_name}' | "
                    f"chunks={total_chunks} | chars={total_chars} | elapsed={elapsed:.2f}s"
                )
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                logging.debug(f"Client {self.client_address[0]} closed connection early.")
                return
            except Exception as stream_err:
                logging.exception(f"[ERROR] Stream error in POST /v1/chat/completions: {stream_err}")
                return
        else:
            full_text = []
            full_thought = []
            tool_calls = []
            for item in stream_gen:
                text_chunk, thought_chunk, tool_call = _unpack_chunk(item)
                if text_chunk:
                    full_text.append(text_chunk)
                if thought_chunk:
                    full_thought.append(thought_chunk)
                if tool_call:
                    fn_name = tool_call.get("name", "")
                    fn_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    args_obj = tool_call.get("args", {})
                    args_str = json.dumps(args_obj, ensure_ascii=False) if isinstance(args_obj, (dict, list)) else str(args_obj)
                    tool_calls.append({
                        "id": fn_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_str
                        }
                    })

            out_text = "".join(full_text)
            msg_out = {
                "role": "assistant",
                "content": out_text if (out_text or not tool_calls) else None
            }
            if tool_calls:
                msg_out["tool_calls"] = tool_calls
            if full_thought:
                msg_out["reasoning_content"] = "".join(full_thought)

            finish_reason = "tool_calls" if tool_calls else "stop"
            self.send_json(200, {
                "id": req_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": msg_out,
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": len(messages),
                    "completion_tokens": len(out_text),
                    "total_tokens": len(messages) + len(out_text)
                }
            })
            elapsed = time.time() - t0
            logging.info(
                f"[RESPONSE] POST /v1/chat/completions 200 OK (non-stream) | model='{model_name}' | "
                f"chars={len(out_text)} | elapsed={elapsed:.2f}s"
            )


class ProxyServer(ThreadingHTTPServer):
    def __init__(self, config: Config):
        self.config = config
        self.auth = AntigravityAuth(config)
        self.client = AntigravityClient(config, self.auth)
        super().__init__((config.host, config.port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Antigravity to Ollama & OpenAI Dual-Protocol Proxy")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.json")
    parser.add_argument("--host", help="Server host (e.g. 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Server port (default 11434)")
    parser.add_argument("--project-id", help="Google Cloud Project ID")
    parser.add_argument("--refresh-token", help="Antigravity OAuth Refresh Token")
    parser.add_argument("--api-url", help="Antigravity upstream API URL")
    parser.add_argument("--default-model", help="Default model name")
    parser.add_argument("--filter-thinking", action="store_true", default=None, help="Filter out thinking/reasoning chunks in Ollama responses")
    parser.add_argument("--no-filter-thinking", action="store_false", dest="filter_thinking", help="Include thinking wrapped in <think> tags")
    parser.add_argument("--login", action="store_true", help="Interactive OAuth login helper")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    args = parser.parse_args()

    # 日志等级与格式配置
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True
    )
    if args.verbose:
        logging.info("Verbose (DEBUG) logging is ENABLED. Full request previews, upstream RPC dispatch, and streaming chunks will be printed.")

    # 登录模式
    if args.login:
        AntigravityAuth.interactive_login(args.config)
        return

    # 加载配置
    config = Config.load(args.config)
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.project_id:
        config.project_id = args.project_id
    if args.refresh_token:
        config.refresh_token = args.refresh_token
    if args.api_url:
        config.api_url = args.api_url
    if args.default_model:
        config.default_model = args.default_model
    if args.filter_thinking is not None:
        config.filter_thinking = args.filter_thinking

    logging.info(f"Starting Antigravity Dual-Protocol Proxy v{__version__} on {config.host}:{config.port}")
    logging.info(f"Target Antigravity Upstream: {config.api_url}")
    logging.info(f"Default Model: {config.default_model}")
    logging.info(f"Filter Thinking: {config.filter_thinking}")

    if not config.refresh_token:
        logging.warning("No refresh_token configured. Requests will fail unless provided.")
        logging.warning("You can run with --login to obtain a refresh_token.")

    server = ProxyServer(config)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
