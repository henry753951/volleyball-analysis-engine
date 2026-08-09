[CmdletBinding()]
param(
    [string]$CentralUrl = "ws://localhost:4000/api/v1/ai/providers/ws",
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$WorkerId = "analysis-worker-local",
    [int]$Concurrency = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:VOLLYAI_SERVER_WS_URL = $CentralUrl
$env:VOLLYAI_TOKEN = $Token
$env:VOLLYAI_INSTANCE_ID = $WorkerId
$env:VOLLYAI_MAX_CONCURRENCY = [string]$Concurrency
Push-Location $projectRoot
try {
    uv run volleyball-analysis worker
}
finally {
    Pop-Location
}
