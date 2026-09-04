# ollama-bridge

<div align="center">

[简体中文](README.md) · **English**

</div>

This project turns **OpenAI-compatible APIs** (DeepSeek, Zhipu BigModel, Kimi, NVIDIA, etc.), **Google Gemini Interactions API**, **Google Cloud Vertex AI Agent Platform API**, and **Google Antigravity Service** into **Ollama and OpenAI-compatible APIs**, so the GitHub Copilot built into **Visual Studio 2022 / 2026** can use custom AI models through "Add Ollama Provider". It is **not designed for VS Code**.

The runtime depends only on the Python standard library with **zero third-party dependencies**.

**If Ollama is not accepted, try [foundry-proxy](https://github.com/BrillianceRen/foundry-proxy).**

---

## Proxy Suite

| Script | Upstream API / Platform | Representative Models | Default Port |
| :--- | :--- | :--- | :--- |
| **`openai_ollama_proxy.py`** | OpenAI-compatible APIs (DeepSeek, Zhipu, Kimi, NVIDIA, OpenCode, etc.) | `deepseek-v4-flash`, `glm-5.2`, `kimi`, `nvidia/*` | `11434` |
| **`gemini-ollama-proxy.py`** | Google AI Developer API (`v1beta/interactions`, `v1beta/models`) | `gemini-3.5-flash`, `gemini-3.7-flash`, `gemma-4-26b-a4b-it` | `11434` |
| **`vertex-ollama-proxy.py`** | Google Cloud Vertex AI Agent Platform (`generateContent`) | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.5-flash-lite` | `11434` |
| **`antigravity-ollama-proxy.py`** | Google Antigravity Service (`cloudcode-pa.googleapis.com`) | `claude-sonnet-4-6`, `claude-opus-4-6`, `gemini-3.7-flash-high` | `11434` |

### Antigravity Core Capabilities

* **Full Multi-turn Tool Calling**: Seamless conversion and roundtrip state tracking between OpenAI / Ollama tool calls and Gemini `functionDeclarations` / Claude `tool_use`.
* **Thought Signature Caching**: Automatically extracts and persists cryptographically verified thought signatures for Gemini 2.5 / 3.7 thinking models, avoiding upstream `Function call is missing a thought_signature` errors.
* **Anthropic Claude Protocol Compatibility (Sonnet / Opus models)**:
  * **Strict Parts Ordering**: Enforces text and reasoning blocks before `functionCall` blocks in model turns to prevent Google's Anthropic converter from generating dangling assistant message prefill.
  * **Empty Text Sanitization**: Skips empty assistant turns and removes empty text parts to prevent `text.text: Field required` errors.
  * **Tool ID Pairing**: Guarantees `tool_use.id` and `functionResponse.id` synchronization, eliminating `tool_use.id: Field required`.
  * **Turn Alternation & User Boundary**: Merges adjacent same-role messages and guarantees conversation begins and ends with `user` messages, avoiding `This model does not support assistant message prefill`.
* **Protobuf Schema Cleaning**: Converts JSON Schema Draft-07 / 2020-12 `type` arrays (e.g., `["string", "null"]`) to single enum types with `nullable: true`, unpacks `anyOf` / `oneOf` unions, ensuring 100% compatibility with VS Copilot's complex tool registry.


---

## Features & Endpoints

| Endpoint | Description |
| :--- | :--- |
| `GET /` | Returns `"Ollama is running"` |
| `GET /api/version` | Returns a simulated Ollama version (e.g. `0.5.4`) |
| `GET /api/tags` | Aggregates model lists; prefers `models/*.json`, auto-generates tag entries otherwise |
| `POST /api/show` | Prefers `models/*.json` templates, auto-generates a response otherwise |
| `POST /api/chat` | Converts to upstream API and forwards (streaming NDJSON & non-streaming, Function Calling supported) |
| `POST /api/generate` | Converts to upstream API and forwards (streaming NDJSON supported, image MIME auto-detected) |
| `GET /api/ps` | Returns an empty model list, for Ollama client status polling |
| `GET /v1/models` | OpenAI-compatible model list |
| `POST /v1/chat/completions` | Pass-through for OpenAI-compatible requests (streaming SSE & non-streaming) |

---

## Quick Start

### 1. Requirements
- Python 3.8+ (no external dependencies needed)
- Copy `config.example.json` to `config.json` and fill in your API keys (`config.json` is in `.gitignore` and won't be committed).

### 2. Launch Proxy

#### (1) OpenAI-compatible Proxy
```powershell
python openai_ollama_proxy.py --config config.json
```

#### (2) Google Gemini Interactions Proxy
```powershell
python gemini-ollama-proxy.py --port 11434
```

#### (3) Google Cloud Vertex AI Proxy
```powershell
python vertex-ollama-proxy.py --port 11434
```

#### (4) Google Antigravity Proxy
```powershell
# Interactive OAuth login helper on first run
python antigravity-ollama-proxy.py --login

# Start proxy server (simultaneously exposes Ollama & OpenAI endpoints)
python antigravity-ollama-proxy.py --port 11434
```

### 3. Verify
```powershell
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/v1/models
```

---

## Configuration (`config.json`)

```jsonc
{
  "host": "127.0.0.1",        // bind address
  "port": 11434,              // listen port (same as Ollama; change if it conflicts)
  "timeout": 300,             // upstream request timeout (seconds)
  "cache_ttl": 60,            // model list cache TTL (seconds)
  "models_dir": "models",     // models/*.json directory
  "use_env_proxy": true,      // use system environment proxy
  "log_level": "info",        // log level: quiet / info / debug

  // Google Gemini API configuration
  "gemini": {
    "api_key": "AQ.YOUR_GEMINI_API_KEY",
    "api_url": "https://generativelanguage.googleapis.com",
    "default_model": "gemini-3.5-flash"
  },

  // Google Cloud Vertex AI Agent Platform configuration
  "vertex": {
    "api_key": "AQ.YOUR_VERTEX_API_KEY",
    "vertex_project": "YOUR_GCP_PROJECT_ID_OR_NUMBER",
    "vertex_location": "us-central1",
    "default_model": "gemini-2.5-flash"
  },

  // Google Antigravity Service configuration
  "antigravity": {
    "project_id": "aicode-consumers",
    "refresh_token": "1//04YOUR_REFRESH_TOKEN",
    "api_url": "https://daily-cloudcode-pa.googleapis.com",
    "default_model": "gemini-3.7-flash-high",
    "filter_thinking": true
  },

  // OpenAI-compatible Providers
  "providers": [
    {
      "name": "deepseek",
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "sk-YOUR_DEEPSEEK_KEY",
      "family": "deepseek",
      "enabled": true,
      "models": []
    },
    {
      "name": "bigmodel",
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "api_key": "sk-YOUR_BIGMODEL_KEY",
      "family": "glm",
      "enabled": false,
      "models": []
    },
    {
      "name": "nvidia",
      "base_url": "https://integrate.api.nvidia.com/v1",
      "api_key": "nvapi-YOUR_NVIDIA_KEY",
      "family": "nvidia",
      "enabled": false,
      "models": []
    }
  ],
  "mapping": {}
}
```

---

## Security & Secrets Management

- **Zero Hardcoded Secrets**: All API keys, project numbers, and tokens are read dynamically from external config files or environment variables.
- **Git Protection**: `config.json`, `*.local.json`, `.env*`, `secrets.json` are strictly ignored by `.gitignore`.
- **Environment Variables**:
  - Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - Vertex AI: `VERTEX_API_KEY`, `VERTEX_PROJECT`, `VERTEX_LOCATION`

---

## Visual Studio 2022 / 2026 Copilot Setup

Visual Studio built-in GitHub Copilot allows adding custom models via "Add Ollama Provider":

1. Open **Copilot Chat**, click the model dropdown, and select **Manage Models**;
2. Click **Add Model / Add Provider**, and select **Ollama**;
3. Set the server URL to the proxy address: `http://127.0.0.1:11434`;
4. Click **Connect**, Visual Studio will fetch the models via `/api/tags`;
5. Select the desired models and save;
6. Now you can select these models in Copilot Chat for code generation.

---

## models/ Metadata Directory

`models/<provider>.json` stores metadata for building Ollama `/api/tags` and `/api/show` responses:
- `models/antigravity.json`: Antigravity Claude 3.7 / 3.5 Sonnet / Opus / Gemini 3.7 Flash series
- `models/gemini.json`: Gemini 3.5 / 3.7 / Gemma 4 series
- `models/vertex.json`: Vertex AI Gemini 2.5 Flash / Pro / Flash-Lite series
- `models/deepseek.json`, `models/bigmodel.json`, `models/nvidia.json`, etc.

---

## Command Line Installation (pip / console script)

```bash
pip install -e .

# Launch commands
openai-ollama-proxy --config config.json
gemini-ollama-proxy --port 11434
vertex-ollama-proxy --port 11434
antigravity-ollama-proxy --port 11434
```

---

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```
*(Offline unit tests verifying payload conversion, SSE streaming, multi-turn, and tool calls)*
