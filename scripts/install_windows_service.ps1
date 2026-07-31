# openai-ollama-proxy Windows 开机自启安装脚本(任务计划程序, 无需额外依赖)
# 用法(以管理员身份运行 PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows_service.ps1
param(
    [string]$TaskName = "openai-ollama-proxy",
    [string]$ConfigPath = ""
)
$ErrorActionPreference = "Stop"

# 管理员权限检查
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "请以管理员身份运行本脚本(右键 PowerShell -> 以管理员身份运行)"
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
if (-not $ConfigPath) { $ConfigPath = Join-Path $rootDir "config.json" }
$proxyPy = Join-Path $rootDir "openai_ollama_proxy.py"

if (-not (Test-Path $proxyPy)) { Write-Error "找不到 openai_ollama_proxy.py: $proxyPy"; exit 1 }
if (-not (Test-Path $ConfigPath)) { Write-Error "找不到配置文件: $ConfigPath"; exit 1 }

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { Write-Error "未找到 python, 请先安装并加入 PATH"; exit 1 }

$logFile = Join-Path $rootDir "proxy-service.log"

# 通过 cmd /c 包装, 把 stdout/stderr 重定向到日志文件
$inner = "`"$proxyPy`" --config `"$ConfigPath`" >> `"$logFile`" 2>&1"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$inner`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "openai-ollama-proxy: OpenAI 兼容 API -> Ollama API 代理" -Force

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 2
Write-Host "已安装并启动计划任务: $TaskName"
Write-Host "配置文件: $ConfigPath"
Write-Host "日志文件: $logFile"
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State