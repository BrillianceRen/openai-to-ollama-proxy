# openai-ollama-proxy

<div align="center">

**简体中文** · [English](README.en.md)

</div>

本项目的目的是把 OpenAI 兼容 API(DeepSeek / 智谱 BigModel / Kimi 等)转换/暴露为
**Ollama API**,从而让 **Visual Studio 2022 / 2026 内置的 GitHub Copilot** 通过
「添加 Ollama Provider」使用自定义 AI 模型。**本项目不是为 VS Code 设计的**。
仅依赖 Python 标准库,零第三方依赖。

**若ollama无效，可尝试Foundry [foundry-proxy](https://github.com/BrillianceRen/foundry-proxy)。**

## 功能

| 端点 | 说明 |
| --- | --- |
| `GET /api/tags` | 汇总各 provider 模型列表;优先匹配 `models/*.json`,未命中自动生成 tag 条目 |
| `POST /api/show` | 优先返回 `models/*.json` 的 show 内容,未命中自动生成应答 |
| `POST /api/chat` | 转换为 OpenAI `/v1/chat/completions` 转发(支持流式) |
| `POST /api/generate` | 转换为 OpenAI `/v1/chat/completions` 转发(支持流式,图片自动识别 MIME) |
| `GET /api/ps` | 返回空模型列表,兼容 Ollama 客户端状态轮询 |
| `GET /api/version` | 返回模拟的 Ollama 版本号 |
| `GET /v1/models` | OpenAI 兼容模型列表 |
| `POST /v1/chat/completions` | OpenAI 兼容请求透传(支持流式) |
| `POST /v1/responses` | OpenAI Responses API 透传(支持流式) |

模型名规则:所有模型统一命名为 `<上游模型>:<provider>`,例如 `glm-5.2` -> `glm-5.2:bigmodel`、
`deepseek-v4-flash` -> `deepseek-v4-flash:deepseek` / `deepseek-v4-flash:opencode-zen`。
不再以 `:latest` 结尾;为兼容旧请求,仍接受 `:latest` 路由,但不再出现在列表。

- 支持任意多组 provider(DeepSeek / 智谱 BigModel / Kimi / OpenCode Zen 等),模型列表
  动态拉取并缓存;过期后立即返回旧缓存并后台刷新,列表请求不会被上游 `/models` 阻塞。
- `models/*.json` 按 provider 保存 Ollama 官方应答模板;同名模型在不同 provider 下
  的参数、上下文和能力可以不同。

## 快速开始

1. 安装 Python 3.8+
2. 编辑 `config.json`,填入各 provider 的 `api_key`(可复制 `config.example.json` 再改)
3. 启动:

```powershell
python openai_ollama_proxy.py --config config.json
```

4. 验证:

```powershell
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/v1/models
```

## 配置文件说明

```jsonc
{
  "host": "127.0.0.1",        // 监听地址
  "port": 11434,              // 监听端口(与 Ollama 相同,冲突可改)
  "timeout": 300,             // 上游请求超时(秒)
  "cache_ttl": 60,            // /models 列表缓存时间(秒)
  "fetch_wait_timeout": 30,   // 首次拉取模型列表的最长等待(秒,默认 30)
  "max_body_bytes": 67108864, // 请求体大小上限(字节,默认 64 MB)
  "retry_without_tools": true, // 上游 5xx 且带 tools 时,剥离 tools 重试一次
  "strip_tools": false,       // 暂时彻底禁用 tools,始终剥离后再发送
  "stream_mode": "auto",       // /v1 流式控制: auto / stream / non_stream
  "default_num_ctx": 4096,    // 自动生成应答时的默认上下文长度
  "models_dir": "models",     // models/*.json 所在目录(相对配置文件)
  "use_env_proxy": true,      // 是否走系统环境代理
  "log_level": "info",        // 日志级别: quiet / info / debug
  "providers": [              // 可配置任意多组
    {
      "name": "deepseek",                     // 唯一名称
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "sk-xxx",
      "family": "deepseek",                   // 自动生成应答时使用的 family(可选)
      "headers": {},                          // 额外上游请求头(可选)
      "models": []                            // 留空 = 从上游 /v1/models 动态发现
    }
  ],
  "mapping": {}               // 显式模型名映射,见下
}
```

- `providers[].models` 留空数组时,`/api/tags` 与 `/v1/models` 会调用上游 `/v1/models`
  动态拉取列表(带 `cache_ttl` 缓存);也可以手写模型 id 列表固定展示。
- `cache_ttl` 过期后,`/api/tags` 会立即返回旧缓存并在后台刷新,不阻塞请求;
  启动时自动并行预热所有 provider,`fetch_wait_timeout` 控制首次拉取的最长等待。
- 上游对带 tools 的请求返回 5xx 时,代理会自动剥离 `tools` / `tool_choice` 重试一次,
  用于临时绕过上游 tools 端点故障(`retry_without_tools`,默认开启)。
- 若模型上游 tools 端点长期不可用,可设 `strip_tools: true` 彻底禁用 tools:`/api/chat` 与
  `/v1/chat/completions` 发送前都会剥离 `tools` / `tool_choice`。
- `/v1/chat/completions` 可用 `stream_mode` 控制上游请求:`auto`(跟随客户端)/ `stream`(强制流式)/
  `non_stream`(强制非流式);强制切换时会自动在 SSE 与 JSON 之间转换,客户端无需改参数。
- `providers[].headers` 可为某个 provider 添加自定义请求头,例如被上游要求特定
  `User-Agent` 时配置 `{"User-Agent": "..."}`。这些头会在 Authorization 之后合并,
  可覆盖默认 Content-Type、Accept 和 User-Agent。
- OpenCode Zen 需要以下请求头才能通过上游边缘校验:
        `"Authorization": "Bearer <API KEY>"`。
- `x-preview-f-free` 模型要求请求中必须包含 `tools` 字段;不带 tools 的请求
  上游会稳定返回 503(Endpoint is unavailable)。代理层可自动注入 noop tool 规避。
- 模型 API 类型通过 `models/*.json` 中 `show.api_type` 配置:
  `"chat_completions"`(默认)或 `"responses"`。代理自动将 Chat Completions
  格式转换为 Responses API 并转换响应格式,客户端无需感知差异。
- `models/*.json` 中 `tag.name` / `tag.model` 无需显式写出 provider 后缀,
  代理会根据 `<model_id>:<provider>` 规则自动生成。
- 若上游 id 与 Ollama 侧名字不一致,可用 `mapping` 显式指定:

```json
"mapping": {
  "glm-5.2:bigmodel": { "provider": "bigmodel", "model": "glm-5.2" },
  "deepseek-chat:deepseek": { "provider": "deepseek", "model": "deepseek-chat" }
}
```

## Visual Studio 2022 / 2026 内置 Copilot 配置

Visual Studio 内置的 GitHub Copilot 不支持直接填写 OpenAI 兼容地址,但支持
「添加 Ollama Provider」。启动本代理后(默认监听 `http://127.0.0.1:11434`),
按以下步骤配置:

1. 打开 **Copilot Chat** 对话面板,点击模型下拉框,选择 **「管理模型」**(Manage Models);
2. 点击 **添加模型 / 添加提供程序**,提供商(Provider)选择 **Ollama**;
3. 服务地址填写本代理地址:`http://127.0.0.1:11434`;
4. 点击 **添加 / 连接**,Visual Studio 会请求 `/api/tags` 拉取模型列表
   (优先使用 `models/*.json`,未命中自动生成);
5. 勾选需要的模型(如 `glm-5.2:bigmodel`、`deepseek-chat:deepseek`)并保存;
6. 之后在 Copilot 的模型下拉框中即可选择这些模型进行对话。

配置完成后,Visual Studio 通过 `/api/show` 获取模型信息、通过 `/api/chat`
发起对话;本代理收到 Ollama 请求后自动转换为 OpenAI 兼容请求,并转发到
`config.json` 中对应的 provider。

> 本代理暴露的是标准 Ollama API,其他 Ollama 客户端理论上也可连接使用,但本项目
> 主要面向 Visual Studio 2022 / 2026 内置 Copilot,不针对 VS Code 提供配置说明。

## models/ 目录

`models/*.json` 按 provider 保存 Ollama `/api/tags` + `/api/show` 应答模板。
模板内容仍遵循 Ollama 官方响应字段；外层 `providers` 只用于关联和路由。

```json
{
  "version": 1,
  "providers": {
    "opencode-zen": {
      "model": "deepseek-v4-flash",
      "tag": { "...": "GET /api/tags 条目" },
      "show": { "...": "POST /api/show 响应" }
    },
    "deepseek": {
      "model": "deepseek-v4-flash",
      "tag": { "...": "可与上面不同" },
      "show": { "...": "可与上面不同" }
    }
  }
}
```

- `providers` 的键必须等于 `config.json.providers[].name`。
- entry 的 `model` 是发送给该 provider 的上游模型 ID；省略时使用文件名。
- 同一个模型文件可以包含多个 provider，每个 provider 的参数、上下文和能力可以不同。

当多个 provider 暴露同一个上游模型 ID 时，代理会保留第一个 provider 的默认名
每个 provider 的模型都会使用 `<上游模型>:<provider>` 形式,例如
`deepseek-v4-flash:deepseek` 与 `deepseek-v4-flash:opencode-zen`。这些名称会同时出现在 `/api/tags`
和 `/v1/models` 中，并在请求时精确路由回对应 provider。

## 安装为命令(pip / console script)

除了直接运行 `python openai_ollama_proxy.py`,项目也提供标准打包(纯标准库,无第三方运行时
依赖)。安装后可获得 `openai-ollama-proxy` 命令,便于开机自启脚本与日常使用:

```bash
pip install -e .            # 从源码目录安装
openai-ollama-proxy --version            # openai-ollama-proxy 1.1.0
openai-ollama-proxy --config config.json # 同 python openai_ollama_proxy.py
```

CLI 参数:

| 参数 | 说明 |
| --- | --- |
| `--config <path>` | 配置文件路径(默认 `config.json`) |
| `--host <addr>` | 覆盖监听地址 |
| `--port <port>` | 覆盖监听端口 |
| `--verbose` | 输出 `debug` 级日志 |
| `--version` | 打印版本号并退出 |

## 测试

测试套件**完全离线**,以假上游响应验证转换与路由逻辑,不发起任何网络请求:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

覆盖范围:`/api/tags`、`/api/show`、`/api/chat`、`/api/generate`、`/v1/*` 的
端到端转换(含流式 SSE 与 tool-call 增量拼接)、消息与工具调用双向转换、
模型名校验/去重、配置解析、图片 MIME 嗅探、请求体读取(Content-Length /
chunked / 超限)以及 CLI 参数。

## 日志

默认输出 `info` 级日志,每次请求会打印 model、provider、上游 URL、token 用量与耗时,
例如:

```text
[12:00:01] chat model=glm-5.2:bigmodel provider=bigmodel url=https://open.bigmodel.cn/api/paas/v4/chat/completions stream=false
[12:00:03] chat 完成 model=glm-5.2:bigmodel provider=bigmodel prompt_tokens=12 completion_tokens=87 elapsed=2.1s
[12:00:04] 127.0.0.1 "POST /api/chat HTTP/1.1" 200 - 3.1s
```

- `log_level` 配置项: `quiet`(关闭)/ `info`(默认) / `debug`(更多细节)
- 启动时加 `--verbose` 等同于 `debug` 级别

## 常驻服务(开机自启)

`scripts/` 目录提供 Linux / Windows 的开机自启脚本:开机自动运行代理、崩溃自动重启。

### Linux(systemd)

```bash
# 安装并启动(需 root)
sudo bash scripts/install_linux_service.sh

# 查看状态与实时日志
systemctl status openai-ollama-proxy
journalctl -u openai-ollama-proxy -f

# 重启 / 停止 / 启动
sudo systemctl restart openai-ollama-proxy
sudo systemctl stop openai-ollama-proxy
sudo systemctl start openai-ollama-proxy

# 卸载
sudo bash scripts/uninstall_linux_service.sh
```

- 脚本会把项目复制到 `/opt/openai-ollama-proxy`,生成 systemd 单元并开机自启
- 默认以 `root` 运行;可用环境变量指定用户:`SERVICE_USER=ollama sudo bash scripts/install_linux_service.sh`(用户需已存在)
- 启动参数、重启策略(失败重启,间隔 3 秒)可在生成的单元文件中调整

### Windows(任务计划程序,无需额外依赖)

以**管理员**身份打开 PowerShell,执行:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows_service.ps1
```

- 创建计划任务 `openai-ollama-proxy`,以 SYSTEM 账户开机自启,崩溃后自动重启(最多 3 次,间隔 1 分钟)
- 运行日志写入项目根目录 `proxy-service.log`
- 手动管理:

```powershell
Start-ScheduledTask -TaskName openai-ollama-proxy   # 启动
Stop-ScheduledTask  -TaskName openai-ollama-proxy   # 停止
Get-ScheduledTask   -TaskName openai-ollama-proxy   # 查看状态
```

- 卸载:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\uninstall_windows_service.ps1
```

> 备选方案:若希望使用真正的 Windows 服务,可配合 [NSSM](https://nssm.cc) 使用:
> `nssm install openai-ollama-proxy <python.exe路径> <项目路径>\openai_ollama_proxy.py --config <配置路径>`,
> 并在 `nssm edit openai-ollama-proxy` 中配置日志输出与自动重启。
## 注意事项

- 默认端口 `11434`,若本机已运行真实 Ollama 会端口冲突,请修改 `config.json` 的 `port`。
- 仅做文本对话转换;`/api/generate` 的 `images` 参数会转换为 OpenAI `image_url`
  格式透传。图片 MIME 由解码后的魔数自动识别(`image/png` / `image/jpeg` /
  `image/gif` / `image/webp` / `image/bmp`),并容忍 `data:image/...;base64,`
  前缀;能否识别图片最终取决于上游模型。
- 流式输出使用 chunked 编码,兼容 Ollama 的 `application/x-ndjson` 与 OpenAI 的
  `text/event-stream`。
