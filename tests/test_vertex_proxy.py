# -*- coding: utf-8 -*-
"""Unit tests for vertex-ollama-proxy."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vertex_ollama_proxy as proxy_mod
from vertex_ollama_proxy import (
    Config,
    VertexClient,
    Handler,
    ProxyServer,
    now_iso,
    ndjson,
    iter_sse_events,
)


def test_vertex_now_iso():
    ts = now_iso()
    assert ts.endswith("Z")
    assert "T" in ts


def test_vertex_model_normalization():
    cfg = Config({"default_model": "gemini-2.5-flash"})
    client = VertexClient(cfg)

    assert client.normalize_model_name("gemini-2.5-flash") == "gemini-2.5-flash"
    assert client.normalize_model_name("gemini-2.5-flash:latest") == "gemini-2.5-flash"
    assert client.normalize_model_name("models/gemini-2.5-flash") == "gemini-2.5-flash"
    assert client.normalize_model_name("flash") == "gemini-2.5-flash"
    assert client.normalize_model_name("pro") == "gemini-2.5-pro"
    assert client.normalize_model_name("") == "gemini-2.5-flash"


def test_vertex_payload_multiturn():
    cfg = Config()
    client = VertexClient(cfg)

    messages = [
        {"role": "system", "content": "You are a concise AI."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How's the weather?"}
    ]

    payload = client.build_payload(
        messages=messages,
        options={"temperature": 0.3, "max_tokens": 150}
    )

    assert payload["systemInstruction"]["parts"][0]["text"] == "You are a concise AI."
    assert payload["generationConfig"]["temperature"] == 0.3
    assert payload["generationConfig"]["maxOutputTokens"] == 150

    contents = payload["contents"]
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hello!"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "Hi!"
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["text"] == "How's the weather?"


def test_vertex_payload_tools():
    cfg = Config()
    client = VertexClient(cfg)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Calculate expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"]
                }
            }
        }
    ]

    payload = client.build_payload(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=tools
    )

    assert len(payload["tools"]) == 1
    decl = payload["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "calc"
    assert decl["parameters"]["required"] == ["expr"]


def test_vertex_base_url():
    cfg = Config({
        "vertex_project": "123456789",
        "vertex_location": "europe-west1"
    })
    client = VertexClient(cfg)
    assert client.base_url() == "https://europe-west1-aiplatform.googleapis.com/v1beta1/projects/123456789/locations/europe-west1"


def test_vertex_auth_headers_api_key():
    cfg = Config({"api_key": "my-test-api-key"})
    client = VertexClient(cfg)
    headers = client._get_auth_headers()
    assert headers == {"x-goog-api-key": "my-test-api-key"}


def test_vertex_auth_headers_bearer_token():
    cfg = Config({"bearer_token": "ya29.test-bearer-token"})
    client = VertexClient(cfg)
    headers = client._get_auth_headers()
    assert headers == {"Authorization": "Bearer ya29.test-bearer-token"}
