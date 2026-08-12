[CmdletBinding()]
param(
    [string]$CentralUrl = "ws://localhost:4000/api/v1/ai/providers/ws",
    [Parameter(Mandatory = $true)][string]$Token,
    [Alias("WorkerId")]
    [string]$InstanceKey = "analysis-worker-local",
    [int]$Concurrency = 1,
    [int]$DetectorStride = 1,
    [int]$ReIdEvery = 1,
    [string]$CourtModel = "v1",
    [int]$CourtImageSize = 640
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:VOLLYAI_SERVER_WS_URL = $CentralUrl
$env:VOLLYAI_TOKEN = $Token
$env:VOLLYAI_INSTANCE_ID = $InstanceKey
$env:VOLLYAI_MAX_CONCURRENCY = [string]$Concurrency
$env:VOLLYAI_DETECTOR_STRIDE = [string]$DetectorStride
$env:VOLLYAI_REID_EVERY = [string]$ReIdEvery
$env:VOLLYAI_COURT_MODEL = $CourtModel
$env:VOLLYAI_COURT_IMGSZ = [string]$CourtImageSize
Push-Location $projectRoot
try {
    uv run --no-sync volleyball-analysis worker
}
finally {
    Pop-Location
}
