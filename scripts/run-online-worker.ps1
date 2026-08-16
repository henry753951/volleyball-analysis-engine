[CmdletBinding()]
param(
    [string]$CentralUrl = "ws://localhost:4000/api/v2/ai/providers/ws",
    [Parameter(Mandatory = $true)][string]$Token,
    [Alias("WorkerId")]
    [string]$InstanceKey = "analysis-worker-local",
    [ValidateRange(1, 64)]
    [int]$Concurrency = 1,
    [int]$ReIdEvery = 1,
    [string]$CourtModel = "v3",
    [int]$CourtImageSize = 512,
    [switch]$EnableReidVlm
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:VOLLYAI_SERVER_WS_URL = $CentralUrl
$env:VOLLYAI_TOKEN = $Token
$env:VOLLYAI_INSTANCE_ID = $InstanceKey
$env:VOLLYAI_MAX_CONCURRENCY = [string]$Concurrency
$env:VOLLYAI_REID_EVERY = [string]$ReIdEvery
$env:VOLLYAI_COURT_MODEL = $CourtModel
$env:VOLLYAI_COURT_IMGSZ = [string]$CourtImageSize
$env:VOLLYAI_COURT_BATCH_SIZE = "16"
$env:VOLLYAI_COURT_DECODER = "auto"
$env:VOLLYAI_REID_FEATURE_ENABLED = "true"
$env:VOLLYAI_REID_ASSOCIATION_ENABLED = "true"
$env:VOLLYAI_IDENTITY_PREVIEW_ENABLED = "true"
$env:VOLLYAI_REID_VLM_ENABLED = if ($EnableReidVlm) { "true" } else { "false" }
Push-Location $projectRoot
try {
    if ($EnableReidVlm) {
        uv run --no-sync volleyball-analysis worker --enable-reid-vlm
    }
    else {
        uv run --no-sync volleyball-analysis worker --disable-reid-vlm
    }
}
finally {
    Pop-Location
}
