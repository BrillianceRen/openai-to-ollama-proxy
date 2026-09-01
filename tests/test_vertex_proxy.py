# -*- coding: utf-8 -*-
"""Unit tests for vertex-ollama-proxy (powered by google-genai SDK)."""
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
    assert client.normalize_model_name("gemini-3.7-flash") == "gemini-2.5-flash"


def test_vertex_payload_multiturn():
    cfg = Config({"vertex_project": "12345"})
    client = VertexClient(cfg)

    messages = [
        {"role": "system", "content": "You are a concise AI."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How's the weather?"}
    ]

    contents, gen_config = client.convert_messages_and_config(
        messages=messages,
        options={"temperature": 0.3, "max_tokens": 150}
    )

    assert gen_config.system_instruction == "You are a concise AI."
    assert gen_config.temperature == 0.3
    assert gen_config.max_output_tokens == 150

    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello!"
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Hi!"
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "How's the weather?"


def test_vertex_payload_tools():
    cfg = Config({"vertex_project": "12345"})
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

    contents, gen_config = client.convert_messages_and_config(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=tools
    )

    assert gen_config.tools is not None
    assert len(gen_config.tools) == 1
    decl = gen_config.tools[0].function_declarations[0]
    assert decl.name == "calc"
    assert getattr(decl.parameters, "required", None) == ["expr"]
