# -*- coding: utf-8 -*-
"""Unit and integration tests for gemini-ollama-proxy."""
import json
import os
import sys
import time
import urllib.request
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemini_ollama_proxy as proxy_mod
from gemini_ollama_proxy import (
    Config,
    GeminiClient,
    Handler,
    ProxyServer,
    now_iso,
    ndjson,
    iter_sse_events,
)


def test_now_iso_format():
    ts = now_iso()
    assert ts.endswith("Z")
    assert "T" in ts


def test_ndjson_encoding():
    data = {"test": "hello 世界"}
    encoded = ndjson(data)
    assert encoded.endswith(b"\n")
    assert json.loads(encoded.decode("utf-8")) == data


def test_model_name_normalization():
    cfg = Config({"default_model": "gemini-3.5-flash"})
    client = GeminiClient(cfg)

    assert client.normalize_model_name("gemini-3.5-flash") == "gemini-3.5-flash"
    assert client.normalize_model_name("gemini-3.5-flash:latest") == "gemini-3.5-flash"
    assert client.normalize_model_name("models/gemini-3.5-flash") == "gemini-3.5-flash"
    assert client.normalize_model_name("flash") == "gemini-3.5-flash"
    assert client.normalize_model_name("pro") == "gemini-3.1-pro-preview"
    assert client.normalize_model_name("gemma") == "gemma-4-26b-a4b-it"
    assert client.normalize_model_name("") == "gemini-3.5-flash"


def test_payload_builder_multiturn():
    cfg = Config()
    client = GeminiClient(cfg)

    messages = [
        {"role": "system", "content": "You are a helpful bot."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]

    payload = client.build_interactions_payload(
        messages=messages,
        model_name="gemini-3.5-flash",
        options={"temperature": 0.5, "num_predict": 100},
        stream=True
    )

    assert payload["model"] == "gemini-3.5-flash"
    assert payload["system_instruction"] == "You are a helpful bot."
    assert payload["stream"] is True
    assert payload["generation_config"]["temperature"] == 0.5
    assert payload["generation_config"]["max_output_tokens"] == 100

    steps = payload["input"]
    assert len(steps) == 3
    assert steps[0]["type"] == "user_input"
    assert steps[0]["content"][0]["text"] == "Hello!"
    assert steps[1]["type"] == "model_output"
    assert steps[1]["content"][0]["text"] == "Hi there!"
    assert steps[2]["type"] == "user_input"
    assert steps[2]["content"][0]["text"] == "How are you?"


def test_payload_builder_tools():
    cfg = Config()
    client = GeminiClient(cfg)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_stock",
                "description": "Stock price lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"]
                }
            }
        }
    ]

    payload = client.build_interactions_payload(
        messages=[{"role": "user", "content": "What is GOOG price?"}],
        model_name="gemini-3.5-flash",
        tools=tools
    )

    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "lookup_stock"
    assert payload["tools"][0]["parameters"]["required"] == ["ticker"]


def test_image_mime_detection():
    # PNG signature: \x89PNG\r\n\x1a\n
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    assert GeminiClient._get_image_mime(png_b64) == "image/png"
