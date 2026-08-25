# -*- coding: utf-8 -*-
"""Offline tests for openai_ollama_proxy — no network, fake upstream responses.

Run:  python -m pytest tests/ -v
"""
import base64
import json
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai_ollama_proxy as proxy_mod  # noqa: E402
from openai_ollama_proxy import (  # noqa: E402
    BodyTooLargeError,
    Config,
    Handler,
    ModelNotFoundError,
    Provider,
    Proxy,
    UpstreamError,
    __version__,
    iter_sse,
    ndjson,
    now_iso,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- helpers
def build_proxy(tmp_path, providers=None, mapping=None):
    models_dir = os.path.join(str(tmp_path), "models")
    os.makedirs(models_dir, exist_ok=True)
    providers = providers or [
        {"name": "deepseek", "base_url": "https://api.deepseek.com/v1",
         "api_key": "sk-test", "family": "deepseek", "models": ["deepseek-v4-flash"]},
    ]
    cfg = Config({"providers": providers, "models_dir": models_dir,
                  "mapping": mapping or {}}, str(tmp_path))
    proxy = Proxy(cfg)
    # Seed model lists so tags()/v1_models() short-circuit without spawning threads.
    for prov in cfg.providers:
        proxy.fetched_ids[prov.name] = list(prov.models)
        proxy.fetched_at[prov.name] = time.time()
    return proxy


class FakeResponse:
    """Mimics urllib response: .read() for JSON bodies, iteration for SSE streams.

    Iteration emulates HTTPResponse.__iter__, which is readline-based: each raw
    yields one line *including* its trailing newline. SSE ``data`` payloads and the
    blank separator lines between them therefore arrive as separate iterated items.
    """

    def __init__(self, body=b"", sse=None):
        self._body = body
        self._sse = sse or []
        self.closed = False

    def read(self, *args):
        return self._body

    def close(self):
        self.closed = True

    def __iter__(self):
        blob = b"".join(self._sse)
        for line in blob.split(b"\n"):
            yield line + b"\n"


# --------------------------------------------------------------------------- module level
def test_version_constant():
    assert isinstance(__version__, str)
    assert __version__


def test_now_iso_rfc3339():
    val = now_iso()
    assert val.endswith("Z")
    assert val[4] == "-" and val[7] == "-" and val[10] == "T"


def test_ndjson_encodes_utf8():
    raw = ndjson({"message": {"content": "héllo"}})
    assert raw.endswith(b"\n")
    assert "héllo".encode("utf-8") in raw


def test_import_exposes_api():
    assert hasattr(proxy_mod, "ProxyServer")
    assert hasattr(proxy_mod, "Handler")
    assert hasattr(proxy_mod, "OLLAMA_VERSION")


# --------------------------------------------------------------------------- iter_sse
def test_iter_sse_single_event():
    resp = FakeResponse(sse=[b'data: {"x": 1}\n\n'])
    got = list(iter_sse(resp))
    assert got == ['{"x": 1}']


def test_iter_sse_multiline_spans():
    # events delimited by blank lines; a single event may span log lines
    resp = FakeResponse(sse=[
        b'data: line1\n',
        b'data: line2\n\n',
        b'data: line3\n\n',
    ])
    got = list(iter_sse(resp))
    assert got == ["line1\nline2", "line3"]


def test_iter_sse_stops_on_done():
    resp = FakeResponse(sse=[
        b'data: {"a": 1}\n\n',
        b'data: [DONE]\n\n',
        b'data: {"b": 2}\n\n',
    ])
    got = list(iter_sse(resp))
    assert got == ['{"a": 1}']
    assert "b" not in "".join(got)


def test_iter_sse_done_after_buffered_payload_flushes():
    resp = FakeResponse(sse=[b'data: hello world\n', b'data: [done]\n\n'])
    got = list(iter_sse(resp))
    assert got == ["hello world"]


def test_iter_sse_ignores_heartbeat_comments():
    resp = FakeResponse(sse=[b': keepalive\n\n', b'data: {"ok": true}\n\n'])
    got = list(iter_sse(resp))
    assert got == ['{"ok": true}']


# --------------------------------------------------------------------------- model naming
def test_model_base_splits_colon():
    assert Proxy._model_base("deepseek-v4-flash:latest") == "deepseek-v4-flash"
    assert Proxy._model_base("deepseek-v4-flash") == "deepseek-v4-flash"


def test_provider_tag_suffix_sanitizes():
    assert Proxy._provider_tag_suffix("Deep Seek!") == "deep-seek-"
    assert Proxy._provider_tag_suffix("glm") == "glm"
    # empty/whitespace -> default
    assert Proxy._provider_tag_suffix("") == "provider"


def test_qualified_model_name_appends_provider():
    q = Proxy._qualified_model_name("glm-5.2", "bigmodel")
    assert q == "glm-5.2:bigmodel"
    q2 = Proxy._qualified_model_name("glm-5.2:latest", "bigmodel")
    assert q2 == "glm-5.2:bigmodel"


# --------------------------------------------------------------------------- Config / Provider
def test_config_parses_defaults(tmp_path):
    cfg = Config({"providers": [{"name": "a", "base_url": "http://x/"}]}, str(tmp_path))
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 11434
    assert cfg.timeout == 300
    assert cfg.cache_ttl == 60
    assert cfg.max_body_bytes == 64 * 1024 * 1024
    assert cfg.default_num_ctx == 4096
    assert cfg.use_env_proxy is True


def test_config_requires_providers(tmp_path):
    with pytest.raises(ValueError):
        Config({"providers": []}, str(tmp_path))


def test_config_provider_requires_name_and_base(tmp_path):
    with pytest.raises(ValueError):
        Config({"providers": [{"name": "x"}]}, str(tmp_path))
    with pytest.raises(ValueError):
        Config({"providers": [{"base_url": "http://x"}]}, str(tmp_path))


def test_config_invalid_log_level_defaults_to_info(tmp_path):
    cfg = Config({"log_level": "bogus",
                  "providers": [{"name": "a", "base_url": "http://x/"}]}, str(tmp_path))
    assert cfg.log_level == "info"


def test_provider_strips_base_url_slash():
    prov = Provider("deepseek", "https://api.deepseek.com/v1/", "key",
                    ["deepseek-v4-flash"], "deepseek")
    assert prov.base_url == "https://api.deepseek.com/v1"
    assert prov.chat_url == "https://api.deepseek.com/v1/chat/completions"
    assert prov.models_url == "https://api.deepseek.com/v1/models"


def test_provider_filters_blank_models():
    prov = Provider("d", "http://x", None, ["", "a", "  ", "b:latest"], "d")
    assert prov.models == ["a", "b:latest"]


# --------------------------------------------------------------------------- tool call conversion
def test_ollama_tool_calls_to_openai():
    out = Proxy.ollama_tool_calls_to_openai([
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
    ])
    assert out[0]["id"].startswith("call_")
    assert out[0]["function"]["name"] == "get_weather"
    # dict args serialized to JSON string per OpenAI convention
    assert json.loads(out[0]["function"]["arguments"]) == {"city": "Paris"}


def test_ollama_tool_calls_to_openai_string_args_passthrough():
    out = Proxy.ollama_tool_calls_to_openai(
        [{"function": {"name": "f", "arguments": '{"x": 1}'}}])
    assert out[0]["function"]["arguments"] == '{"x": 1}'


def test_openai_tool_calls_to_ollama_roundtrip():
    oa = [{"id": "call_0", "type": "function",
           "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    ol = Proxy.openai_tool_calls_to_ollama(oa)
    assert ol[0]["function"]["name"] == "get_weather"
    assert ol[0]["function"]["arguments"] == {"city": "Paris"}


def test_openai_tool_calls_to_ollama_bad_args_wrapped():
    out = Proxy.openai_tool_calls_to_ollama(
        [{"function": {"name": "f", "arguments": "{not json"}}])
    assert isinstance(out[0]["function"]["arguments"], dict)
    assert "raw" in out[0]["function"]["arguments"]


def test_convert_messages_stringifies_nonlist_content():
    msgs = [{"role": "user", "content": 123}, {"role": "assistant", "content": None}]
    converted = Proxy.convert_messages(msgs)
    assert converted[0]["content"] == "123"
    assert converted[1]["content"] == ""


# --------------------------------------------------------------------------- build_generate_payload / images
def _b64(raw_bytes):
    return base64.b64encode(raw_bytes).decode("ascii")


def test_get_image_mime_png():
    assert Proxy._get_image_mime(_b64(b"\x89PNG\r\n\x1a\n" + b"raw")) == "image/png"


def test_get_image_mime_jpeg():
    assert Proxy._get_image_mime(_b64(b"\xff\xd8\xff\xe0" + b"raw")) == "image/jpeg"


def test_get_image_mime_gif():
    assert Proxy._get_image_mime(_b64(b"GIF89a" + b"raw")) == "image/gif"


def test_get_image_mime_webp():
    assert Proxy._get_image_mime(_b64(b"RIFF\x00\x00\x00\x00WEBPraw")) == "image/webp"


def test_get_image_mime_bmp():
    assert Proxy._get_image_mime(_b64(b"BMraw")) == "image/bmp"


def test_get_image_mime_unknown_falls_back_png():
    assert Proxy._get_image_mime(_b64(b"\x00\x01\x02\x03")) == "image/png"


def test_get_image_mime_invalid_b64():
    assert Proxy._get_image_mime("!!!not-base64!!!") == "image/png"


def test_strip_data_uri():
    out = Proxy._strip_data_uri(["data:image/png;base64,AAAA", "BBBB"])
    assert out == ["AAAA", "BBBB"]
    # non-data strings untouched
    assert Proxy._strip_data_uri(["plain"]) == ["plain"]


def test_build_generate_payload_sets_real_mime(tmp_path):
    proxy = build_proxy(tmp_path)
    body = {"model": "deepseek-v4-flash", "prompt": "look",
            "images": [_b64(b"\xff\xd8\xff\xe0jpegdata")]}
    payload = proxy.build_generate_payload(body, "deepseek-v4-flash")
    img_url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert img_url.startswith("data:image/jpeg;base64,")
    assert payload["messages"][0]["content"][0]["type"] == "text"


def test_build_generate_payload_without_images(tmp_path):
    proxy = build_proxy(tmp_path)
    body = {"model": "deepseek-v4-flash", "prompt": "hi", "system": "be terse"}
    payload = proxy.build_generate_payload(body, "deepseek-v4-flash")
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


# --------------------------------------------------------------------------- apply_options / build_chat_payload
def test_apply_options_maps_keys(tmp_path):
    proxy = build_proxy(tmp_path)
    params = {"x": 1}
    proxy.apply_options(params, {"temperature": 0.7, "num_predict": 128, "top_p": 0.9})
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 128
    assert params["top_p"] == 0.9


def test_apply_options_stop_listifies_string(tmp_path):
    proxy = build_proxy(tmp_path)
    params = {}
    proxy.apply_options(params, {"stop": "END"})
    assert params["stop"] == ["END"]


def test_apply_options_ignores_nonpositive_num_predict(tmp_path):
    proxy = build_proxy(tmp_path)
    params = {}
    proxy.apply_options(params, {"num_predict": 0})
    assert "max_tokens" not in params


def test_build_chat_payload_required_and_format(tmp_path):
    proxy = build_proxy(tmp_path)
    body = {"model": "deepseek-v4-flash", "stream": False, "format": "json",
            "messages": [{"role": "user", "content": "hi"}]}
    payload = proxy.build_chat_payload(body, "deepseek-v4-flash")
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["stream"] is False
    assert payload["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------------------- resolve_model
def test_resolve_model_exact(tmp_path):
    proxy = build_proxy(tmp_path, providers=[
        {"name": "deepseek", "base_url": "http://x/", "family": "deepseek",
         "models": ["deepseek-v4-flash"]}])
    provider, mid = proxy.resolve_model("deepseek-v4-flash")
    assert provider is not None and mid == "deepseek-v4-flash"


def test_resolve_model_unknown_returns_none(tmp_path):
    proxy = build_proxy(tmp_path, providers=[
        {"name": "deepseek", "base_url": "http://x/", "family": "deepseek",
         "models": ["deepseek-v4-flash"]}])
    provider, mid = proxy.resolve_model("no-such-model")
    assert provider is None and mid is None


def test_resolve_model_empty_name(tmp_path):
    proxy = build_proxy(tmp_path)
    assert proxy.resolve_model("") == (None, None)
    assert proxy.resolve_model(None) == (None, None)


def test_resolve_model_mapping_alias(tmp_path):
    proxy = build_proxy(tmp_path, mapping={
        "my-alias": {"provider": "deepseek", "model": "deepseek-v4-flash"}})
    provider, mid = proxy.resolve_model("my-alias")
    assert mid == "deepseek-v4-flash"


# --------------------------------------------------------------------------- tags / show
def test_tags_lists_models(tmp_path):
    proxy = build_proxy(tmp_path)
    result = proxy.tags()
    names = [m["name"] for m in result["models"]]
    assert "deepseek-v4-flash:deepseek" in names
    assert all(m["details"]["family"] for m in result["models"])


def test_tags_qualifies_shared_upstream_ids(tmp_path):
    # Two providers expose the same upstream model id — each can't own the bare
    # Each provider exposes the model under its own provider-qualified name.
    proxy = build_proxy(tmp_path, providers=[
        {"name": "a", "base_url": "http://x/", "models": ["model-x"]},
        {"name": "b", "base_url": "http://y/", "models": ["model-x"]}])
    result = proxy.tags()
    names = sorted(m["name"] for m in result["models"])
    assert names == ["model-x:a", "model-x:b"]


def test_show_for_auto_generates(tmp_path):
    proxy = build_proxy(tmp_path)
    show = proxy.show_for("deepseek-v4-flash")
    assert "capabilities" in show
    assert show["model_info"]["general.architecture"] == "deepseek"


def test_show_for_unknown_uses_first_provider(tmp_path):
    proxy = build_proxy(tmp_path)
    show = proxy.show_for("totally-unknown")
    assert show["capabilities"] == ["completion", "tools"]


# --------------------------------------------------------------------------- chat (non-stream)
def test_chat_converts_openai_response(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    up_resp = {
        "id": "cmpl-1", "choices": [{"message": {"role": "assistant",
        "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}
    fake = FakeResponse(body=json.dumps(up_resp).encode("utf-8"))
    monkeypatch.setattr(proxy, "upstream_request", lambda *a, **k: fake)
    out = proxy.chat({"model": "deepseek-v4-flash",
                      "messages": [{"role": "user", "content": "hi"}]})
    assert out["message"]["content"] == "hello"
    assert out["done"] is True and out["done_reason"] == "stop"
    assert out["eval_count"] == 2
    assert out["model"] == "deepseek-v4-flash"


def test_chat_model_not_found(tmp_path):
    proxy = build_proxy(tmp_path)
    with pytest.raises(ModelNotFoundError):
        proxy.chat({"model": "missing", "messages": []})


def test_chat_converts_tool_calls(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    up_resp = {"choices": [{"message": {
        "role": "assistant",
        "tool_calls": [{"id": "call_0", "type": "function",
                        "function": {"name": "get_weather",
                                     "arguments": '{"city": "Paris"}'}}]},
        "finish_reason": "tool_calls"}]}
    fake = FakeResponse(body=json.dumps(up_resp).encode("utf-8"))
    monkeypatch.setattr(proxy, "upstream_request", lambda *a, **k: fake)
    out = proxy.chat({"model": "deepseek-v4-flash",
                      "messages": [{"role": "user", "content": "weather?"}]})
    assert out["done_reason"] == "tool_calls"
    assert out["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


# --------------------------------------------------------------------------- chat_stream
def _sse_chunk(obj):
    return ("data: %s\n\n" % json.dumps(obj, ensure_ascii=False)).encode("utf-8")


def test_chat_stream_accumulates_content(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    chunks = [
        _sse_chunk({"choices": [{"delta": {"role": "assistant",
                                            "content": "Hel"}}]}),
        _sse_chunk({"choices": [{"delta": {"content": "lo"}}]}),
        _sse_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse_chunk({"choices": [], "usage": {"prompt_tokens": 2,
                                             "completion_tokens": 3,
                                             "total_tokens": 5}}),
        b"data: [DONE]\n\n",
    ]
    fake = FakeResponse(sse=chunks)
    monkeypatch.setattr(proxy, "upstream_request", lambda *a, **k: fake)
    emitted = []
    proxy.chat_stream({"model": "deepseek-v4-flash", "messages": []}, emitted.append)
    contents = [json.loads(x)["message"]["content"] for x in emitted
                if (json.loads(x).get("message") or {}).get("content")]
    assert contents == ["Hel", "lo"]
    done = [json.loads(x) for x in emitted if json.loads(x).get("done")]
    assert done[-1]["done_reason"] == "stop"
    assert done[-1]["eval_count"] == 3


def test_chat_stream_tool_calls_accumulated(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    chunks = [
        _sse_chunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "get_", "arguments": "{\"city\":"}}]}}]}),
        _sse_chunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "weather", "arguments": " \"Paris\"}"}}]}}]}),
        _sse_chunk({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        b"data: [DONE]\n\n",
    ]
    fake = FakeResponse(sse=chunks)
    monkeypatch.setattr(proxy, "upstream_request", lambda *a, **k: fake)
    emitted = []
    proxy.chat_stream({"model": "deepseek-v4-flash", "messages": []}, emitted.append)
    done = [json.loads(x) for x in emitted if json.loads(x).get("done")][-1]
    call = done["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"] == {"city": "Paris"}
    assert done["done_reason"] == "tool_calls"


def test_chat_stream_error_chunk(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    fake = FakeResponse(sse=[_sse_chunk({"error": "rate limited"}), b"data: [DONE]\n\n"])
    monkeypatch.setattr(proxy, "upstream_request", lambda *a, **k: fake)
    emitted = []
    proxy.chat_stream({"model": "deepseek-v4-flash", "messages": []}, emitted.append)
    assert json.loads(emitted[0])["error"] == "rate limited"
    assert json.loads(emitted[0])["done"] is False


# --------------------------------------------------------------------------- generate / generate_stream
def test_generate_returns_response(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    up_resp = {"choices": [{"message": {"role": "assistant", "content": "the answer"},
                            "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}
    fake = FakeResponse(body=json.dumps(up_resp).encode("utf-8"))
    monkeypatch.setattr(proxy, "upstream_request", lambda *a, **k: fake)
    out = proxy.generate({"model": "deepseek-v4-flash", "prompt": "q?"})
    assert out["response"] == "the answer"
    assert out["done_reason"] == "stop"
    assert out["eval_count"] == 5


def test_generate_model_not_found(tmp_path):
    proxy = build_proxy(tmp_path)
    with pytest.raises(ModelNotFoundError):
        proxy.generate({"model": "missing", "prompt": "hi"})


def test_generate_stream_emits_chunks(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    chunks = [
        _sse_chunk({"choices": [{"delta": {"content": "part1"}}]}),
        _sse_chunk({"choices": [{"delta": {"content": "part2"}}]}),
        _sse_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse_chunk({"choices": [], "usage": {"prompt_tokens": 1,
                                             "completion_tokens": 2,
                                             "total_tokens": 3}}),
        b"data: [DONE]\n\n",
    ]
    fake = FakeResponse(sse=chunks)
    monkeypatch.setattr(proxy, "upstream_request", lambda *a, **k: fake)
    emitted = []
    proxy.generate_stream({"model": "deepseek-v4-flash", "prompt": "go"}, emitted.append)
    responses = [json.loads(x)["response"] for x in emitted
                 if json.loads(x).get("response")]
    assert responses == ["part1", "part2"]


# --------------------------------------------------------------------------- v1
def test_v1_models_shape(tmp_path):
    proxy = build_proxy(tmp_path)
    result = proxy.v1_models()
    assert result["object"] == "list"
    assert result["data"][0]["object"] == "model"


def test_v1_chat_passes_through_and_renames_model(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    up_resp = {"id": "x", "model": "deepseek-v4-flash",
               "choices": [{"message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}]}
    fake = FakeResponse(body=json.dumps(up_resp).encode("utf-8"))
    monkeypatch.setattr(proxy, "upstream_request", lambda *a, **k: fake)
    out = proxy.v1_chat({"model": "deepseek-v4-flash",
                         "messages": [{"role": "user", "content": "hi"}]},
                        stream=False)
    assert out["model"] == "deepseek-v4-flash"
    # upstream body must carry the *upstream* model id forwarded
    captured = {}
    def capturing(provider, url, payload=None):
        captured["payload"] = payload
        return fake
    proxy.upstream_request = capturing
    proxy.v1_chat({"model": "deepseek-v4-flash", "messages": []}, stream=False)
    assert captured["payload"]["model"] == "deepseek-v4-flash"


# --------------------------------------------------------------------------- upstream error mapping
def test_upstream_error_maps_http(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    def boom(provider, url, payload=None):
        raise UpstreamError(429, '{"error":"quota"}')
    monkeypatch.setattr(proxy, "upstream_request", boom)
    with pytest.raises(UpstreamError) as exc_info:
        proxy.chat({"model": "deepseek-v4-flash", "messages": []})
    assert exc_info.value.status == 429


def test_upstream_error_maps_url(tmp_path, monkeypatch):
    proxy = build_proxy(tmp_path)
    def boom(provider, url, payload=None):
        raise UpstreamError(0, "connection refused")
    monkeypatch.setattr(proxy, "upstream_request", boom)
    with pytest.raises(UpstreamError) as exc_info:
        proxy.generate({"model": "deepseek-v4-flash", "prompt": "x"})
    assert exc_info.value.status == 0


# --------------------------------------------------------------------------- pure helper edge cases
def test_convert_messages_preserves_tool_calls(tmp_path):
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}]
    converted = Proxy.convert_messages(msgs)
    assert converted[0]["tool_calls"][0]["function"]["name"] == "f"


def test_generate_tag_entry_shape(tmp_path):
    proxy = build_proxy(tmp_path)
    entry = proxy.generate_tag_entry("glm-5.2:latest", proxy.config.providers[0])
    assert entry["name"] == "glm-5.2:latest"
    assert entry["details"]["family"]


# --------------------------------------------------------------------------- body reading
class _BodyProbe:
    # expose the Handler body-reading methods so read_body/json_body can
    # internally dispatch via self.<method> just like on a real handler.
    read_body = Handler.read_body
    _read_chunked = Handler._read_chunked
    json_body = Handler.json_body
    log = lambda self, *a, **k: None

    def __init__(self, headers, raw_body, proxy):
        import io
        self.headers = headers
        self.rfile = io.BytesIO(raw_body)
        self.proxy = proxy


def _probe(headers, raw_body, tmp_path, max_body=None):
    proxy = build_proxy(tmp_path)
    if max_body is not None:
        proxy.config.max_body_bytes = max_body
    return Handler.read_body(_BodyProbe(headers, raw_body, proxy))


def test_read_body_content_length(tmp_path):
    raw = b'{"a": 1}'
    assert _probe({"Content-Length": str(len(raw))}, raw, tmp_path) == raw


def test_read_body_empty_returns_immediately(tmp_path):
    # Content-Length: 0 -> empty body must NOT try to read limit+1 bytes
    assert _probe({"Content-Length": "0"}, b"", tmp_path) == b""


def test_read_body_no_length_empty(tmp_path):
    assert _probe({}, b"", tmp_path) == b""


def test_read_body_chunked(tmp_path):
    raw = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
    result = _probe({"Transfer-Encoding": "chunked"}, raw, tmp_path)
    assert result == b"hello world"


def test_read_body_chunked_oversized(tmp_path):
    # 8-byte chunk exceeds a 4-byte cap
    raw = b"8\r\nabcdefgh\r\n0\r\n\r\n"
    with pytest.raises(BodyTooLargeError):
        _probe({"Transfer-Encoding": "chunked"}, raw, tmp_path, max_body=4)


def test_read_body_content_length_oversized(tmp_path):
    with pytest.raises(BodyTooLargeError):
        _probe({"Content-Length": "100"}, b"x" * 100, tmp_path, max_body=8)


def test_json_body_invalid_json(tmp_path):
    import io
    proxy = build_proxy(tmp_path)
    probe = _BodyProbe({"Content-Length": "4"}, b"not!", proxy)
    with pytest.raises(ValueError):
        Handler.json_body(probe)


def test_send_error_json_v1_shape(tmp_path):
    proxy = build_proxy(tmp_path)
    fake = type("H", (), {})()
    fake.path = "/v1/chat/completions"
    sent = {}
    fake.send_json = lambda obj, status=200, headers=None: sent.update(obj=obj, status=status)
    Handler.send_error_json(fake, "boom", 404)
    assert sent["status"] == 404
    assert sent["obj"] == {"error": {"message": "boom", "type": "proxy_error", "code": 404}}


def test_send_error_json_ollama_shape(tmp_path):
    proxy = build_proxy(tmp_path)
    fake = type("H", (), {})()
    fake.path = "/api/chat"
    sent = {}
    fake.send_json = lambda obj, status=200, headers=None: sent.update(obj=obj, status=status)
    Handler.send_error_json(fake, "boom", 404)
    assert sent["obj"] == {"error": "boom"}


def test_handler_suppresses_client_connection_abort(tmp_path):
    proxy = build_proxy(tmp_path)

    class AbortFile:
        def readline(self, *args):
            raise ConnectionAbortedError("client closed keep-alive")

    class AbortHandler(Handler):
        pass

    handler = AbortHandler.__new__(AbortHandler)
    handler.close_connection = False
    handler.log = lambda *args, **kwargs: None
    handler.rfile = AbortFile()
    Handler.handle_one_request(handler)
    assert handler.close_connection is True


# --------------------------------------------------------------------------- CLI
def _cli(*args):
    py = os.path.join(os.path.dirname(sys.executable), "python")
    return subprocess.run([py, os.path.join(REPO_ROOT, "openai_ollama_proxy.py"),
                           *args], capture_output=True, text=True)


def test_cli_version_flags():
    r = _cli("--version")
    assert r.returncode == 0
    assert __version__ in r.stdout


def test_cli_missing_config_exits_1():
    r = _cli("--config", "/nonexistent/config.json")
    assert r.returncode == 1
    assert "配置" in r.stderr or "config" in r.stderr


def test_cli_help_shows_version_and_host():
    r = _cli("--help")
    assert r.returncode == 0
    for flag in ("--version", "--host", "--port", "--verbose", "--config"):
        assert flag in r.stdout
