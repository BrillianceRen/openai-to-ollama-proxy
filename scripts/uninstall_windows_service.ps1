# openai-ollama-proxy Windows 服务卸载脚本
# 用法(以管理员身份运行 PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\uninstall_windows_service.ps1
param(
    [string]$TaskName = "openai-ollama-proxy"
)
$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "请以管理员身份运行本脚本"
    exit 1
}

Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "已卸载计划任务: $TaskName"