# openai-ollama-proxy

<div align="center">

[简体中文](README.md) · **English**

</div>

This project turns OpenAI-compatible APIs (DeepSeek, Zhipu BigModel, Kimi, etc.) into an **Ollama API**, so the GitHub Copilot built into **Visual Studio 2022 / 2026** can use custom AI models through "Add Ollama Provider". It is **not designed for VS Code**. The runtime depends only on the Python standard library.

**If Ollama is not accepted, try [foundry-proxy](https://github.com/BrillianceRen/foundry-proxy).**

## Features

| Endpoint | Description |
| --- | --- |
| `GET /api/tags` | Aggregates model lists from all providers; prefers `models/*.json`, auto-generates tag entries otherwise |
| `POST /api/show` | Prefers `models/*.json` templates, auto-generates a response otherwise |
| `POST /api/chat` | Converts to OpenAI `/v1/chat/completions` and forwards (streaming supported) |
| `POST /api/generate` | Converts to OpenAI `/v1/chat/completions` and forwards (streaming supported, image MIME auto-detected) |
| `GET /api/ps` | Returns an empty model list, for Ollama client status polling |
| `GET /api/version` | Returns a fake Ollama version |
| `GET /v1/models` | OpenAI-compatible model list |
| `POST /v1/chat/completions` | Pass-through for OpenAI-compatible requests (streaming supported) |
| `POST /v1/responses` | OpenAI Responses API pass-through (streaming supported) |

Model naming: every public model is named `<upstream-model>:<provider>`, e.g. `glm-5.2` -> `glm-5.2:bigmodel`,
`deepseek-v4-flash` -> `deepseek-v4-flash:deepseek` / `deepseek-v4-flash:opencode-zen`.
Names no longer end with `:latest`; `:latest` routes are still accepted for compatibility but not exposed.

- Any number of providers are supported (DeepSeek, Zhipu BigModel, Kimi, OpenCode Zen, etc.). Model lists are fetched dynamically and cached; stale caches are returned immediately while a background refresh runs, so list requests are never blocked by upstream `/models`.
- `models/*.json` stores official Ollama response templates per provider; the same model name can have different parameters, context length, and capabilities under different providers.

## Quick Start

1. Install Python 3.8+
2. Edit `config.json` and fill in each provider's `api_key` (copy `config.example.json` first)
3. Start:

```powershell
python openai_ollama_proxy.py --config config.json
```

4. Verify:

```powershell
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/v1/models
```

## Configuration

```jsonc
{
  "host": "127.0.0.1",        // bind address
  "port": 11434,              // listen port (same as Ollama; change if it conflicts)
  "timeout": 300,             // upstream request timeout (seconds)
  "cache_ttl": 60,            // /models list cache TTL (seconds)
  "fetch_wait_timeout": 30,   // max wait for the first model list fetch (seconds, default 30)
  "max_body_bytes": 67108864, // request body size limit (bytes, default 64 MB)
  "retry_without_tools": true, // retry once without tools on upstream 5xx for tool requests
  "strip_tools": false,       // temporarily disable tools entirely, always strip before sending
  "stream_mode": "auto",       // /v1 stream control: auto / stream / non_stream
  "default_num_ctx": 4096,    // default context length for generated responses
  "models_dir": "models",     // models/*.json directory (relative to the config file)
  "use_env_proxy": true,      // use system environment proxy
  "log_level": "info",        // log level: quiet / info / debug
  "providers": [              // any number of providers
    {
      "name": "deepseek",                     // unique name
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "sk-xxx",
      "family": "deepseek",                   // family used for generated responses (optional)
      "headers": {},                          // extra upstream headers (optional)
      "models": []                            // empty = discover dynamically via upstream /v1/models
    }
  ],
  "mapping": {}               // explicit model name mapping, see below
}
```

- When `providers[].models` is empty, `/api/tags` and `/v1/models` fetch the list dynamically from upstream `/v1/models` (cached for `cache_ttl` seconds). You can also hard-code model ids to pin the list.
- After `cache_ttl` expires, `/api/tags` immediately returns the stale cache and refreshes in the background; all providers are warmed up in parallel at startup, and `fetch_wait_timeout` caps the wait for the first fetch.
- When an upstream returns 5xx for a request with tools, the proxy retries once after stripping `tools` / `tool_choice`, which works around temporary upstream tool-endpoint failures (`retry_without_tools`, enabled by default).
- If the tools endpoint stays unavailable, set `strip_tools: true` to disable tools entirely: `tools` / `tool_choice` are stripped before `/api/chat` and `/v1/chat/completions` are forwarded.
- `/v1/chat/completions` supports `stream_mode`: `auto` (follow the client) / `stream` (always stream upstream) / `non_stream` (always non-stream upstream). SSE and JSON are converted automatically when the mode is forced, so clients do not need to change parameters.
- `providers[].headers` adds custom headers for a provider, e.g. a required `User-Agent`. Headers are merged after Authorization and can override the default Content-Type, Accept, and User-Agent.
- OpenCode Zen requires the following headers to pass edge validation:
  `"Authorization": "Bearer <API KEY>"`.
- The `x-preview-f-free` model requires the `tools` field to be present in the request; requests without tools consistently return 503 (Endpoint is unavailable) from upstream. The proxy can inject a noop tool to work around this.
- Model API type is configured via `show.api_type` in `models/*.json`: `"chat_completions"` (default) or `"responses"`. The proxy automatically converts Chat Completions format to/from the Responses API so clients don't need to change anything.
- `tag.name` and `tag.model` in `models/*.json` no longer need explicit provider suffixes; the proxy auto-generates `<model_id>:<provider>` names.
- If the Ollama-side name differs from the upstream id, use `mapping`:

```json
"mapping": {
  "glm-5.2:bigmodel": { "provider": "bigmodel", "model": "glm-5.2" },
  "deepseek-chat:deepseek": { "provider": "deepseek", "model": "deepseek-chat" }
}
```

## Visual Studio 2022 / 2026 Copilot Setup

The GitHub Copilot built into Visual Studio does not accept a raw OpenAI-compatible URL, but it does support "Add Ollama Provider". With the proxy running (default `http://127.0.0.1:11434`):

1. Open the **Copilot Chat** panel and click the model dropdown, then choose **Manage Models**.
2. Click **Add model / Add provider** and select **Ollama**.
3. Enter the proxy address: `http://127.0.0.1:11434`.
4. Click **Add / Connect**; Visual Studio calls `/api/tags` to load the model list (templates from `models/*.json` first, auto-generated otherwise).
5. Tick the models you need (e.g. `glm-5.2:bigmodel`, `deepseek-chat:deepseek`) and save.
6. The models now appear in the Copilot model dropdown for chat.

After setup, Visual Studio fetches model info via `/api/show` and chats via `/api/chat`; the proxy converts Ollama requests to OpenAI-compatible requests and forwards them to the provider configured in `config.json`.

> The proxy exposes the standard Ollama API, so other Ollama clients can theoretically connect too, but this project targets Visual Studio 2022 / 2026 Copilot and provides no VS Code configuration instructions.

## models/ Directory

`models/*.json` stores Ollama `/api/tags` + `/api/show` response templates per provider. The template body follows the official Ollama response fields; the outer `providers` object is used only for association and routing.

```json
{
  "version": 1,
  "providers": {
    "opencode-zen": {
      "model": "deepseek-v4-flash",
      "tag": { "...": "GET /api/tags entry" },
      "show": { "...": "POST /api/show response" }
    },
    "deepseek": {
      "model": "deepseek-v4-flash",
      "tag": { "...": "may differ from the above" },
      "show": { "...": "may differ from the above" }
    }
  }
}
```

- The keys of `providers` must equal the `config.json.providers[].name` values.
- Each entry's `model` is the upstream model id sent to that provider; when omitted, the file name is used.
- One model file can contain multiple providers, each with different parameters, context, and capabilities.

Every provider exposes models as `<upstream-model>:<provider>`, e.g. `deepseek-v4-flash:deepseek` and `deepseek-v4-flash:opencode-zen`. These names appear in both `/api/tags` and `/v1/models` and route back to the exact provider on request.

## Install as a Command (pip / console script)

Besides running `python openai_ollama_proxy.py`, the project also ships a standard package (pure stdlib, no third-party runtime dependencies). Installing gives you the `openai-ollama-proxy` command:

```bash
pip install -e .            # install from source
openai-ollama-proxy --version            # openai-ollama-proxy 1.1.0
openai-ollama-proxy --config config.json # same as python openai_ollama_proxy.py
```

CLI arguments:

| Argument | Description |
| --- | --- |
| `--config <path>` | Config file path (default `config.json`) |
| `--host <addr>` | Override the bind address |
| `--port <port>` | Override the listen port |
| `--verbose` | Enable `debug` logging |
| `--version` | Print version and exit |

## Tests

The test suite is **fully offline**: fake upstream responses verify conversion and routing logic with zero network access.

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Coverage: `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/v1/*` end-to-end conversion (including streaming SSE and tool-call delta aggregation), message/tool-call round-trips, model qualification/dedup, config parsing, image MIME sniffing, request-body reading (Content-Length / chunked / oversize), and CLI behavior.

> On Windows with a non-UTF-8 locale (e.g. Chinese), run tests with `PYTHONUTF8=1` so the CLI subprocess tests decode output correctly.

## Logging

`info` is the default level. Every request logs model, provider, upstream URL, token usage, and elapsed time:

```text
[12:00:01] chat model=glm-5.2:bigmodel provider=bigmodel url=https://open.bigmodel.cn/api/paas/v4/chat/completions stream=false
[12:00:03] chat 完成 model=glm-5.2:bigmodel provider=bigmodel prompt_tokens=12 completion_tokens=87 elapsed=2.1s
[12:00:04] 127.0.0.1 "POST /api/chat HTTP/1.1" 200 - 3.1s
```

- `log_level` config: `quiet` (off) / `info` (default) / `debug` (more detail)
- `--verbose` at startup is equivalent to `debug`

## Run as a Service (Auto-start)

The `scripts/` directory provides Linux / Windows auto-start scripts: start on boot and restart after crashes.

### Linux (systemd)

```bash
# Install and start (requires root)
sudo bash scripts/install_linux_service.sh

# Status and live logs
systemctl status openai-ollama-proxy
journalctl -u openai-ollama-proxy -f

# Restart / stop / start
sudo systemctl restart openai-ollama-proxy
sudo systemctl stop openai-ollama-proxy
sudo systemctl start openai-ollama-proxy

# Uninstall
sudo bash scripts/uninstall_linux_service.sh
```

- The script copies the project to `/opt/openai-ollama-proxy`, creates a systemd unit, and enables auto-start
- Defaults to running as `root`; pass `SERVICE_USER=ollama sudo bash scripts/install_linux_service.sh` to use another existing user
- Startup arguments and restart policy (restart on failure, 3-second interval) can be tuned in the generated unit file

### Windows (Task Scheduler, no extra dependencies)

Run PowerShell as **Administrator** and execute:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows_service.ps1
```

- Creates the scheduled task `openai-ollama-proxy`, starts on boot as SYSTEM, restarts after crashes (up to 3 times, 1-minute interval)
- Logs are written to `proxy-service.log` in the project root
- Manual management:

```powershell
Start-ScheduledTask -TaskName openai-ollama-proxy   # start
Stop-ScheduledTask  -TaskName openai-ollama-proxy   # stop
Get-ScheduledTask   -TaskName openai-ollama-proxy   # status
```

- Uninstall:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\uninstall_windows_service.ps1
```

> Alternative: to run as a real Windows service, use [NSSM](https://nssm.cc):
> `nssm install openai-ollama-proxy <python.exe path> <project path>\openai_ollama_proxy.py --config <config path>`,
> then configure logging and auto-restart in `nssm edit openai-ollama-proxy`.

## Notes

- Default port is `11434`; if a real Ollama is already running on the machine, change `port` in `config.json`.
- Text chat conversion only; the `/api/generate` `images` parameter is forwarded as OpenAI `image_url` format. Image MIME is auto-detected from decoded magic bytes (`image/png` / `image/jpeg` / `image/gif` / `image/webp` / `image/bmp`) and `data:image/...;base64,` prefixes are tolerated; whether the image is understood depends on the upstream model.
- Streaming uses chunked encoding and is compatible with Ollama `application/x-ndjson` and OpenAI `text/event-stream`.
