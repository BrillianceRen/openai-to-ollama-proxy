"""Validate x-preview-f-free model: tools/stream/headers combinations."""

import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "https://opencode.ai/zen/v1/chat/completions"
API_KEY = os.environ.get("OPENCODE_ZEN_API_KEY", "")
MODEL = "x-preview-f-free"
SESSION_ID = "test-session-001"

SPECIAL_HEADERS = {
    "User-Agent": "opencode/1.18.21 (win32 10.0.26200; x64)",
    "originator": "opencode",
    "session-id": SESSION_ID,
    "x-session-affinity": SESSION_ID,
    "X-Session-Id": SESSION_ID,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]

MESSAGES = [{"role": "user", "content": "What is the weather in Beijing? Answer briefly."}]


def run_case(use_tools, use_stream, use_headers):
    body = {"model": MODEL, "messages": MESSAGES, "stream": use_stream}
    if use_tools:
        body["tools"] = TOOLS

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    if use_headers:
        headers.update(SPECIAL_HEADERS)

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE_URL, data=data, headers=headers, method="POST")

    label = f"tools={use_tools!s:5} stream={use_stream!s:5} headers={use_headers!s:5}"
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        status = resp.status
        content_type = resp.headers.get("Content-Type", "")
        first_chunk_preview = ""
        tool_calls_found = False

        if use_stream:
            # Read SSE lines
            lines_read = 0
            collected_text = []
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content") or ""
                    tc = delta.get("tool_calls")
                    if text:
                        collected_text.append(text)
                    if tc:
                        tool_calls_found = True
                lines_read += 1
                if lines_read == 1:
                    first_chunk_preview = payload[:120]
            elapsed = time.time() - start
            full_text = "".join(collected_text)[:100]
            result = "OK"
            detail = f"text={full_text!r} tool_calls={tool_calls_found}"
        else:
            raw = resp.read().decode("utf-8")
            elapsed = time.time() - start
            data_json = json.loads(raw)
            choices = data_json.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            text = (message.get("content") or "")[:100]
            tc = message.get("tool_calls")
            tool_calls_found = bool(tc)
            finish = choices[0].get("finish_reason") if choices else None
            usage = data_json.get("usage", {})
            result = "OK"
            detail = f"text={text!r} tool_calls={tool_calls_found} finish={finish} usage={json.dumps(usage)}"
        print(f"[{result}] {label} ({elapsed:.2f}s) {detail}")
        return True
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        err_body = e.read().decode("utf-8", errors="replace")[:300]
        print(f"[FAIL] {label} ({elapsed:.2f}s) HTTP {e.code}: {err_body}")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"[FAIL] {label} ({elapsed:.2f}s) {type(e).__name__}: {e}")
        return False


def main():
    results = {}
    cases = [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ]
    passed = 0
    failed = 0
    for use_tools, use_stream, use_headers in cases:
        ok = run_case(use_tools, use_stream, use_headers)
        key = f"tools={use_tools},stream={use_stream},headers={use_headers}"
        results[key] = ok
        if ok:
            passed += 1
        else:
            failed += 1
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Total: {passed + failed}  Passed: {passed}  Failed: {failed}")
    for key, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {key}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
