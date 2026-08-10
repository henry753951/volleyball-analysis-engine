[CmdletBinding()]
param(
    [string]$CentralHttpUrl = "http://localhost:10000",
    [string]$CentralWsUrl = "ws://localhost:10000/api/v1/ai/providers/ws",
    [string]$InstanceKey = "analysis-worker-rtx5070-court36",
    [string]$TokenName = "local-rtx5070-court36",
    [string]$CourtCheckpoint = ".artifacts/models/court-keypoints-video91-canonical-v4.pt",
    [int]$CourtStride = 1,
    [int]$CourtImageSize = 1280,
    [int]$Concurrency = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$checkpoint = (Resolve-Path (Join-Path $projectRoot $CourtCheckpoint)).Path
$workerExecutable = Join-Path $projectRoot ".venv\Scripts\volleyball-analysis-worker.exe"
if (-not (Test-Path -LiteralPath $workerExecutable)) {
    throw "Worker executable is missing: $workerExecutable"
}

$headers = @{
    "x-dev-role" = "ADMIN"
    "x-dev-user-id" = "00000000-0000-4000-8000-000000000001"
    "x-dev-display-name" = "Dev Operator"
}
$tokenLabel = "$TokenName-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$created = Invoke-RestMethod `
    -Method Post `
    -Uri "$($CentralHttpUrl.TrimEnd('/'))/api/v1/operations/ai-worker-tokens" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body (@{ name = $tokenLabel } | ConvertTo-Json -Compress)

$env:VOLLYAI_SERVER_WS_URL = $CentralWsUrl
$env:VOLLYAI_TOKEN = $created.token
$env:VOLLYAI_INSTANCE_ID = $InstanceKey
$env:VOLLYAI_MAX_CONCURRENCY = [string]$Concurrency
$env:VOLLYAI_COURT_CHECKPOINT = $checkpoint
$env:VOLLYAI_COURT_STRIDE = [string]$CourtStride
$env:VOLLYAI_COURT_IMGSZ = [string]$CourtImageSize
$env:VOLLYAI_WORKSPACE = Join-Path $projectRoot ".artifacts\workspaces"

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $projectRoot ".artifacts\online-worker-$stamp.stdout.log"
$stderr = Join-Path $projectRoot ".artifacts\online-worker-$stamp.stderr.log"
$process = Start-Process `
    -FilePath $workerExecutable `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

[PSCustomObject]@{
    process_id = $process.Id
    instance_key = $InstanceKey
    token_id = $created.access_token.id
    token_name = $created.access_token.name
    token_prefix = $created.access_token.token_prefix
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json -Compress
