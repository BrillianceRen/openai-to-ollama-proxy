# -*- coding: utf-8 -*-
"""Unit tests for antigravity-ollama-proxy."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antigravity_ollama_proxy as proxy_mod
from antigravity_ollama_proxy import (
    Config,
    AntigravityAuth,
    AntigravityClient,
    Handler,
    ProxyServer,
    now_iso,
    ndjson,
)


def test_antigravity_now_iso():
    ts = now_iso()
    assert ts.endswith("Z")
    assert "T" in ts


def test_antigravity_ndjson():
    data = {"test": 123}
    encoded = ndjson(data)
    assert encoded.endswith(b"\n")
    assert json.loads(encoded.decode("utf-8")) == data


def test_antigravity_config_defaults(monkeypatch):
    monkeypatch.delenv("ANTIGRAVITY_PROJECT_ID", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_API_URL", raising=False)
    monkeypatch.delenv("ANTIGRAVITY_REFRESH_TOKEN", raising=False)
    cfg = Config({})
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 11434
    assert cfg.filter_thinking is True
    assert cfg.default_model == "gemini-3.7-flash-high"
    assert cfg.api_url == "https://daily-cloudcode-pa.googleapis.com"
    assert cfg.project_id == ""


def test_antigravity_model_normalization():
    cfg = Config({
        "antigravity": {
            "default_model": "claude-sonnet-4-6",
            "model_mappings": {
                "my-custom-model": "claude-opus-4-6-thinking"
            }
        }
    })
    auth = AntigravityAuth(cfg)
    client = AntigravityClient(cfg, auth)

    assert client.resolve_model("gemini-3.7-flash-high") == "gemini-3.7-flash-high"
    assert client.resolve_model("gemini-3.7-flash-high:latest") == "gemini-3.7-flash-high"
    assert client.resolve_model("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert client.resolve_model("my-custom-model") == "claude-opus-4-6-thinking"
    assert client.resolve_model("unknown-model-xyz") == "unknown-model-xyz"
    assert client.resolve_model("") == "claude-sonnet-4-6"


def test_antigravity_wrap_payload():
    cfg = Config({
        "antigravity": {
            "project_id": "test-project-12345",
            "default_model": "claude-sonnet-4-6"
        }
    })
    auth = AntigravityAuth(cfg)
    client = AntigravityClient(cfg, auth)

    contents = [
        {"role": "user", "parts": [{"text": "Hello, write a python script"}]}
    ]

    payload = client.wrap_antigravity_payload(
        contents=contents,
        system_instruction="You are a helpful assistant.",
        model="claude-sonnet-4-6",
        options={"temperature": 0.5, "max_tokens": 512}
    )

    assert payload["project"] == "test-project-12345"
    assert payload["userAgent"] == "antigravity"
    if cfg.enable_credit:
        assert payload.get("enabledCreditTypes") == ["GOOGLE_ONE_AI"]
    else:
        assert "enabledCreditTypes" not in payload
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["requestId"].startswith("agent/")

    inner = payload["request"]
    assert inner["sessionId"].startswith("-")
    assert inner["systemInstruction"]["parts"][0]["text"] == "You are a helpful assistant."
    assert inner["toolConfig"]["functionCallingConfig"]["mode"] == "VALIDATED"
    assert inner["generationConfig"]["temperature"] == 0.5
    assert inner["generationConfig"]["maxOutputTokens"] == 512

    labels = inner["labels"]
    assert labels["last_step_index"] == "1"
    assert labels["model_enum"] == "claude-sonnet-4-6"
    assert labels["used_claude"] == "true"


def test_convert_messages_to_contents():
    # We can instantiate a dummy Handler without starting a network server
    class DummyServer:
        def __init__(self):
            self.config = Config({})
            self.client = AntigravityClient(self.config, AntigravityAuth(self.config))

    handler = Handler.__new__(Handler)
    handler.server = DummyServer()

    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]

    contents, sys_text = handler._convert_messages_to_contents(messages)
    assert sys_text == "System prompt."
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hello!"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "Hi there!"
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["text"] == "How are you?"


def test_convert_messages_multimodal():
    class DummyServer:
        def __init__(self):
            self.config = Config({})
            self.client = AntigravityClient(self.config, AntigravityAuth(self.config))

    handler = Handler.__new__(Handler)
    handler.server = DummyServer()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
            ]
        }
    ]

    contents, _ = handler._convert_messages_to_contents(messages)
    assert len(contents) == 1
    parts = contents[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "Describe this image:"
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert parts[1]["inlineData"]["data"] == "iVBORw0KGgo="


import threading
import urllib.request


@pytest.fixture
def running_server(monkeypatch):
    """Fixture that boots a real ProxyServer with mocked upstream generator."""
    cfg = Config({
        "host": "127.0.0.1",
        "port": 0,  # Ephemeral port
        "antigravity": {
            "project_id": "mock-project",
            "refresh_token": "mock-token",
            "filter_thinking": True
        }
    })
    server = ProxyServer(cfg)
    assigned_port = server.server_address[1]

    # Mock stream_generate to yield mock text and thought
    def mock_stream_generate(payload):
        yield ("", "Let me think about this...")
        yield ("Here is ", "")
        yield ("the answer.", "")

    monkeypatch.setattr(server.client, "stream_generate", mock_stream_generate)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    base_url = f"http://127.0.0.1:{assigned_port}"
    yield base_url, server

    server.shutdown()
    server.server_close()


def test_e2e_get_endpoints(running_server):
    base_url, _ = running_server

    # GET /
    with urllib.request.urlopen(f"{base_url}/") as resp:
        assert resp.status == 200
        assert resp.read() == b"Ollama is running\n"

    # GET /api/version
    with urllib.request.urlopen(f"{base_url}/api/version") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["version"] == "0.5.4"

    # GET /api/ps
    with urllib.request.urlopen(f"{base_url}/api/ps") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["models"] == []

    # GET /api/tags
    with urllib.request.urlopen(f"{base_url}/api/tags") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert "models" in data
        assert len(data["models"]) > 0

    # GET /v1/models
    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["object"] == "list"
        assert len(data["data"]) > 0


def test_e2e_ollama_chat_streaming(running_server):
    base_url, _ = running_server

    req_body = json.dumps({
        "model": "antigravity-claude",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        lines = [line.strip() for line in resp.read().decode().splitlines() if line.strip()]
        chunks = [json.loads(line) for line in lines]
        assert len(chunks) >= 2
        # Filter thinking is True, so thoughts shouldn't appear in assistant message content
        content_parts = [c.get("message", {}).get("content", "") for c in chunks if not c.get("done")]
        assert "".join(content_parts) == "Here is the answer."
        assert chunks[-1]["done"] is True


def test_e2e_openai_chat_streaming(running_server):
    base_url, _ = running_server

    req_body = json.dumps({
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        raw = resp.read().decode()
        assert "data: [DONE]" in raw
        # Parse data lines
        sse_lines = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:") and not line.endswith("[DONE]")]
        chunks = [json.loads(line) for line in sse_lines if line]
        assert len(chunks) >= 2
        assert any(c["choices"][0]["delta"].get("reasoning_content") for c in chunks)
        texts = [c["choices"][0]["delta"].get("content", "") for c in chunks]
        assert "".join(texts) == "Here is the answer."


def test_e2e_non_streaming(running_server):
    base_url, _ = running_server

    # 1. Ollama non-streaming
    ollama_req = json.dumps({
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False
    }).encode("utf-8")

    req1 = urllib.request.Request(
        f"{base_url}/api/chat",
        data=ollama_req,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req1) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["done"] is True
        assert data["message"]["content"] == "Here is the answer."

    # 2. OpenAI non-streaming
    openai_req = json.dumps({
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False
    }).encode("utf-8")

    req2 = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=openai_req,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req2) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Here is the answer."
        assert data["choices"][0]["message"]["reasoning_content"] == "Let me think about this..."


def test_tool_conversion_and_schema_cleaning():
    from antigravity_ollama_proxy import _clean_json_schema, _convert_tools_to_gemini

    raw_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CommandArgs",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "command": {"type": "string", "description": "CLI command"}
        },
        "required": ["command"]
    }
    cleaned = _clean_json_schema(raw_schema)
    assert "$schema" not in cleaned
    assert "additionalProperties" not in cleaned
    assert "title" not in cleaned
    assert cleaned["properties"]["command"]["type"] == "string"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute command",
                "parameters": raw_schema
            }
        }
    ]
    gemini_tools = _convert_tools_to_gemini(tools)
    assert gemini_tools is not None
    assert len(gemini_tools[0]["functionDeclarations"]) == 1
    decl = gemini_tools[0]["functionDeclarations"][0]
    assert decl["name"] == "run_command"
    assert decl["description"] == "Execute command"
    assert "$schema" not in decl["parameters"]


def test_messages_conversion_with_tools():
    class DummyServer:
        def __init__(self):
            self.config = Config({})
    h = Handler.__new__(Handler)
    h.server = DummyServer()

    messages = [
        {"role": "user", "content": "Run git status"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "run_command", "arguments": "{\"command\": \"git status\"}"}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "name": "run_command",
            "content": "On branch main\nnothing to commit"
        }
    ]
    contents, sys_text = h._convert_messages_to_contents(messages)
    assert len(contents) == 3
    # 1. user
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Run git status"
    # 2. model tool call
    assert contents[1]["role"] == "model"
    assert "functionCall" in contents[1]["parts"][0]
    assert contents[1]["parts"][0]["functionCall"]["name"] == "run_command"
    assert contents[1]["parts"][0]["functionCall"]["args"] == {"command": "git status"}
    # 3. user functionResponse
    assert contents[2]["role"] == "user"
    assert "functionResponse" in contents[2]["parts"][0]
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "run_command"
    assert contents[2]["parts"][0]["functionResponse"]["response"] == {"result": "On branch main\nnothing to commit"}


def test_e2e_tool_calls_streaming(running_server, monkeypatch):
    base_url, server = running_server

    def mock_tools_generate(payload):
        yield ("", "", {"name": "run_command", "args": {"command": "git status"}, "id": "call_git1"})

    monkeypatch.setattr(server.client, "stream_generate", mock_tools_generate)

    # 1. OpenAI v1 stream
    req_body = json.dumps({
        "model": "gemini-3.7-flash-high",
        "messages": [{"role": "user", "content": "check git"}],
        "tools": [{"type": "function", "function": {"name": "run_command", "parameters": {}}}],
        "stream": True
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        raw = resp.read().decode()
        sse_lines = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:") and not line.endswith("[DONE]")]
        chunks = [json.loads(line) for line in sse_lines if line]
        # Check tool_calls delta
        assert any("tool_calls" in c["choices"][0]["delta"] for c in chunks)
        # Check finish_reason
        assert any(c["choices"][0].get("finish_reason") == "tool_calls" for c in chunks)

    # 2. Ollama stream
    ollama_req = json.dumps({
        "model": "gemini-3.7-flash-high",
        "messages": [{"role": "user", "content": "check git"}],
        "tools": [{"type": "function", "function": {"name": "run_command", "parameters": {}}}],
        "stream": True
    }).encode("utf-8")
    req2 = urllib.request.Request(
        f"{base_url}/api/chat",
        data=ollama_req,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req2) as resp:
        assert resp.status == 200
        lines = [json.loads(l) for l in resp.read().decode().splitlines() if l.strip()]
        assert any("tool_calls" in l.get("message", {}) for l in lines)
        assert lines[-1]["done_reason"] == "tool_calls"


