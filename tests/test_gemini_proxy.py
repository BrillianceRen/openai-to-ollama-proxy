# -*- coding: utf-8 -*-
"""Unit tests for gemini-ollama-proxy (powered by google-genai SDK)."""
import json
import os
import sys
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
    cfg = Config({"default_model": "gemini-3.6-flash"})
    client = GeminiClient(cfg)

    assert client.normalize_model_name("gemini-3.6-flash") == "gemini-3.6-flash"
    assert client.normalize_model_name("gemini-3.6-flash:latest") == "gemini-3.6-flash"
    assert client.normalize_model_name("models/gemini-3.6-flash") == "gemini-3.6-flash"
    assert client.normalize_model_name("flash") == "gemini-3.6-flash"
    assert client.normalize_model_name("pro") == "gemini-3.1-pro-preview"
    assert client.normalize_model_name("") == "gemini-3.6-flash"


def test_payload_builder_multiturn():
    cfg = Config({"api_key": "test-key"})
    client = GeminiClient(cfg)

    messages = [
        {"role": "system", "content": "You are a helpful bot."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]

    contents, gen_config = client.convert_messages_and_config(
        messages=messages,
        options={"temperature": 0.5, "num_predict": 100}
    )

    assert gen_config.system_instruction == "You are a helpful bot."
    assert gen_config.temperature == 0.5
    assert gen_config.max_output_tokens == 100

    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello!"
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Hi there!"
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "How are you?"


def test_payload_builder_tools():
    cfg = Config({"api_key": "test-key"})
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

    contents, gen_config = client.convert_messages_and_config(
        messages=[{"role": "user", "content": "What is GOOG price?"}],
        tools=tools
    )

    assert gen_config.tools is not None
    assert len(gen_config.tools) == 1
    decls = gen_config.tools[0].function_declarations
    assert len(decls) == 1
    assert decls[0].name == "lookup_stock"
    assert getattr(decls[0].parameters, "required", None) == ["ticker"]


def test_image_mime_detection():
    # PNG signature: \x89PNG\r\n\x1a\n
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert GeminiClient._get_image_mime(png_bytes) == "image/png"
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    assert GeminiClient._get_image_mime(jpeg_bytes) == "image/jpeg"
