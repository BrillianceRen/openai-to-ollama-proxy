# ollama-bridge

<div align="center">

**简体中文** · [English](README.en.md)

</div>

本项目的目的是把 **OpenAI 兼容 API**（DeepSeek / 智谱 BigModel / Kimi / NVIDIA 等）、**Google Gemini Interactions API**、**Google Cloud Vertex AI Agent Platform API** 以及 **Google Antigravity 服务** 转换并暴露为 **Ollama 与 OpenAI 兼容 API**，从而让 **Visual Studio 2022 / 2026 内置的 GitHub Copilot** 通过「添加 Ollama Provider」自由使用自定义 AI 模型。**本项目不是为 VS Code 设计的**。

全套工具仅依赖 Python 标准库，**零第三方依赖**。

**若 Ollama 无效，可尝试 Foundry [foundry-proxy](https://github.com/BrillianceRen/foundry-proxy)。**

---

## 代理矩阵

| 代理脚本 | 对应平台 / API | 典型支持模型 | 默认端口 |
| :--- | :--- | :--- | :--- |
| **`openai_ollama_proxy.py`** | OpenAI 兼容接口（DeepSeek, 智谱, Kimi, NVIDIA, OpenCode 等） | `deepseek-v4-flash`, `glm-5.2`, `kimi`, `nvidia/*` 等 | `11434` |
| **`gemini-ollama-proxy.py`** | Google AI Developer API (`v1beta/interactions`, `v1beta/models`) | `gemini-3.5-flash`, `gemini-3.7-flash`, `gemma-4-26b-a4b-it` 等 | `11434` |
| **`vertex-ollama-proxy.py`** | Google Cloud Vertex AI Agent Platform (`generateContent`) | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.5-flash-lite` 等 | `11434` |
| **`antigravity-ollama-proxy.py`** | Google Antigravity 服务 (`cloudcode-pa.googleapis.com`) | `claude-sonnet-4-5`, `claude-opus-4-5`, `gemini-2.5-pro` 等 | `11434` |

---

## 端点支持

| 端点 | 说明 |
| :--- | :--- |
| `GET /` | 返回 `"Ollama is running"` |
| `GET /api/version` | 返回模拟的 Ollama 版本号（如 `0.5.4`） |
| `GET /api/tags` | 汇总各模型列表；优先匹配 `models/*.json`，未命中自动生成 tag 条目 |
| `POST /api/show` | 优先返回 `models/*.json` 的 show 内容，未命中自动生成应答 |
| `POST /api/chat` | 转换为上游 API 转发（支持流式 NDJSON 与非流式，支持 Function / Tool Calling） |
| `POST /api/generate` | 转换为上游 API 转发（支持流式 NDJSON，图片自动识别 MIME 格式） |
| `GET /api/ps` | 返回空模型列表，兼容 Ollama 客户端状态轮询 |
| `GET /v1/models` | OpenAI 兼容模型列表 |
| `POST /v1/chat/completions` | OpenAI 兼容请求透传（支持流式 SSE 与非流式） |

---

## 快速开始

### 1. 安装与准备
- Python 3.8+（无需安装额外 pip 包）
- 复制 `config.example.json` 为 `config.json` 并填入相应的 API Key 与配置（`config.json` 已加入 `.gitignore`，不会被提交到 Git）。

### 2. 启动代理

#### (1) 启动通用 OpenAI 兼容代理
```powershell
python openai_ollama_proxy.py --config config.json
```

#### (2) 启动 Google Gemini Interactions 代理
```powershell
python gemini-ollama-proxy.py --port 11434
```

#### (3) 启动 Google Cloud Vertex AI 代理
```powershell
python vertex-ollama-proxy.py --port 11434
```

#### (4) 启动 Google Antigravity 代理
```powershell
# 首次使用可执行交互式授权获取 refresh_token 并写入 config.json
python antigravity-ollama-proxy.py --login

# 启动服务 (同时开放 Ollama 与 OpenAI 兼容端点)
python antigravity-ollama-proxy.py --port 11434
```

### 3. 验证连通性
```powershell
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/v1/models
```

---

## 配置文件说明 (`config.json`)

```jsonc
{
  "host": "127.0.0.1",        // 监听地址
  "port": 11434,              // 监听端口 (与 Ollama 相同，冲突可改)
  "timeout": 300,             // 上游请求超时 (秒)
  "cache_ttl": 60,            // 模型列表缓存时间 (秒)
  "models_dir": "models",     // models/*.json 所在目录
  "use_env_proxy": true,      // 是否走系统环境代理
  "log_level": "info",        // 日志级别: quiet / info / debug

  // Google Gemini API 专属配置
  "gemini": {
    "api_key": "AQ.请填入你的GeminiApiKey",
    "api_url": "https://generativelanguage.googleapis.com",
    "default_model": "gemini-3.5-flash"
  },

  // Google Cloud Vertex AI Agent Platform 专属配置
  "vertex": {
    "api_key": "AQ.请填入你的VertexApiKey",
    "vertex_project": "你的GoogleCloud项目号或ID",
    "vertex_location": "us-central1",
    "default_model": "gemini-2.5-flash"
  },

  // Google Antigravity 服务专属配置
  "antigravity": {
    "project_id": "aicode-consumers",
    "refresh_token": "1//04请填入你的RefreshToken",
    "api_url": "https://daily-cloudcode-pa.googleapis.com",
    "default_model": "gemini-3.7-flash-high",
    "filter_thinking": true
  },

  // 通用 OpenAI 兼容 Provider 列表
  "providers": [
    {
      "name": "deepseek",
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "sk-请填入你的DeepSeekKey",
      "family": "deepseek",
      "enabled": true,
      "models": []
    },
    {
      "name": "bigmodel",
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "api_key": "sk-请填入你的智谱Key",
      "family": "glm",
      "enabled": false,
      "models": []
    },
    {
      "name": "nvidia",
      "base_url": "https://integrate.api.nvidia.com/v1",
      "api_key": "nvapi-请填入你的NvidiaKey",
      "family": "nvidia",
      "enabled": false,
      "models": []
    }
  ],
  "mapping": {}
}
```

---

## 安全与敏感信息管理

- **源码零硬编码**：所有 API Key、Project ID 均通过外部配置文件或环境变量传入。
- **Git 忽略保护**：`config.json`、`*.local.json`、`.env*`、`secrets.json` 等已列入 `.gitignore`，禁止入库。
- **环境变量支持**：
  - Gemini: `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`
  - Vertex AI: `VERTEX_API_KEY`, `VERTEX_PROJECT`, `VERTEX_LOCATION`

---

## Visual Studio 2022 / 2026 内置 Copilot 配置

Visual Studio 内置的 GitHub Copilot 不支持直接填写 OpenAI 兼容地址，但支持「添加 Ollama Provider」。启动本代理后（默认监听 `http://127.0.0.1:11434`），按以下步骤配置：

1. 打开 **Copilot Chat** 对话面板，点击模型下拉框，选择 **「管理模型」**(Manage Models)；
2. 点击 **添加模型 / 添加提供程序**，提供商 (Provider) 选择 **Ollama**；
3. 服务地址填写本代理地址：`http://127.0.0.1:11434`；
4. 点击 **添加 / 连接**，Visual Studio 会请求 `/api/tags` 拉取模型列表；
5. 勾选需要的模型并保存；
6. 之后在 Copilot 的模型下拉框中即可选择这些模型进行对话与代码生成。

---

## models/ 模板目录

`models/<provider>.json` 按 provider 保存模型元数据，代理用它构建 Ollama `/api/tags` 和 `/api/show` 应答：
- `models/gemini.json`: Gemini 3.5 / 3.7 / Gemma 4 系列
- `models/vertex.json`: Vertex AI Gemini 2.5 Flash / Pro / Flash-Lite 系列
- `models/deepseek.json`, `models/bigmodel.json`, `models/nvidia.json` 等

---

## 安装为命令行工具 (pip / console script)

```bash
pip install -e .

# 启动命令
openai-ollama-proxy --config config.json
gemini-ollama-proxy --port 11434
vertex-ollama-proxy --port 11434
```

---

## 测试

测试套件**完全离线**，以假上游响应验证转换与路由逻辑，不发起外部真实网络请求：

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## 开机自启常驻服务

`scripts/` 目录提供 Linux (systemd) 与 Windows (任务计划程序) 的自启脚本，支持崩溃自动拉起。
