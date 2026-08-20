[CmdletBinding(DefaultParameterSetName = "ExistingToken")]
param(
    [Parameter(Mandatory = $true)][string]$ServerUrl,
    [Parameter(ParameterSetName = "ExistingToken")][string]$Token = $env:VOLLYAI_TOKEN,
    [Parameter(ParameterSetName = "LocalToken")][switch]$CreateLocalToken,
    [string]$CentralHttpUrl = "https://volleyai.hsulab.net",
    [string]$TokenName = "uv-analysis-worker",
    [string]$InstanceId = "$env:COMPUTERNAME-uv-analysis-worker",
    [string]$AssetsRoot,
    [ValidateSet("auto", "cpu", "cuda:0")][string]$Device = "auto",
    [switch]$WithReid,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($AssetsRoot)) { $AssetsRoot = Join-Path $projectRoot ".models" }
$configPath = Join-Path $projectRoot ".env.worker"
$assetsPath = [IO.Path]::GetFullPath($AssetsRoot)

function Resolve-WorkerWebSocketUrl([string]$Value) {
    $normalized = $Value.Trim().TrimEnd("/")
    if ($normalized.StartsWith("https://")) { $normalized = "wss://" + $normalized.Substring(8) }
    elseif ($normalized.StartsWith("http://")) { $normalized = "ws://" + $normalized.Substring(7) }
    $normalized = $normalized.Replace("/api/v1/ai/providers/ws", "/api/v2/ai/providers/ws")
    if (-not $normalized.EndsWith("/api/v2/ai/providers/ws")) {
        $normalized += "/api/v2/ai/providers/ws"
    }
    if (-not ($normalized.StartsWith("ws://") -or $normalized.StartsWith("wss://"))) {
        throw "ServerUrl must be an HTTP(S) base URL or WS(S) URL."
    }
    return $normalized
}

if ($CreateLocalToken) {
    $headers = @{
        "x-dev-role" = "ADMIN"
        "x-dev-user-id" = "00000000-0000-4000-8000-000000000001"
        "x-dev-display-name" = "Dev Operator"
    }
    $created = Invoke-RestMethod `
        -Method Post `
        -Uri "$($CentralHttpUrl.TrimEnd('/'))/api/v1/operations/ai-worker-tokens" `
        -Headers $headers `
        -ContentType "application/json" `
        -Body (@{ name = "$TokenName-$(Get-Date -Format 'yyyyMMdd-HHmmss')" } | ConvertTo-Json -Compress)
    $Token = $created.token
}
elseif ([string]::IsNullOrWhiteSpace($Token)) {
    $secureToken = Read-Host "Worker Access Token" -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    try { $Token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
}
if ($Token.Length -lt 16 -or $Token.Contains("`n") -or $Token.Contains("`r")) {
    throw "Worker token must be a single line containing at least 16 characters."
}
if ((Test-Path -LiteralPath $configPath) -and -not $Force) {
    throw "$configPath already exists. Pass -Force to replace it."
}

$sdkRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "src")).Path
$smpRoot = (Resolve-Path -LiteralPath (Join-Path $assetsPath "selective-mask-propagation")).Path
$checkpoint = (Resolve-Path -LiteralPath (Join-Path $assetsPath "volleyball_multitask\best.pth")).Path
$required = @(
    (Join-Path $sdkRoot "volleyball_sdk\__init__.py"),
    $checkpoint,
    (Join-Path $smpRoot "selective_mask_propagation\deep_eiou\tracker.py"),
    (Join-Path $smpRoot "selective_mask_propagation\osnet\checkpoints\sports_model.pth.tar-60")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required asset is missing: $path" }
}

if ($Device -eq "auto") {
    $python = Join-Path $projectRoot ".venv\Scripts\python.exe"
    $detected = & $python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')"
    if ($LASTEXITCODE -ne 0) { throw "Unable to detect the PyTorch device" }
    $Device = $detected.Trim()
}

$values = [ordered]@{
    VOLLYAI_SERVER_WS_URL = Resolve-WorkerWebSocketUrl $ServerUrl
    VOLLYAI_TOKEN = $Token
    VOLLYAI_INSTANCE_ID = $InstanceId
    VOLLYAI_DEVICE = $Device
    VOLLYAI_WORKSPACE = (Join-Path $projectRoot ".artifacts\workspaces")
    VOLLYAI_MULTITASK_SDK_ROOT = $sdkRoot
    VOLLYAI_MULTITASK_CHECKPOINT = $checkpoint
    VOLLYAI_SMP_ROOT = $smpRoot
    VOLLYAI_OSNET_CHECKPOINT = (Join-Path $smpRoot "selective_mask_propagation\osnet\checkpoints\sports_model.pth.tar-60")
    VOLLYAI_LOCAL_TRACKER = "deep_eiou"
    VOLLYAI_LOCAL_SAM3_ENABLED = "false"
    VOLLYAI_REID_FEATURE_ENABLED = $WithReid.ToString().ToLowerInvariant()
    VOLLYAI_REID_ASSOCIATION_ENABLED = "true"
    VOLLYAI_IDENTITY_PREVIEW_ENABLED = "true"
    VOLLYAI_PREWARM_MODELS = "true"
}
if ($WithReid) {
    $kprRoot = (Resolve-Path -LiteralPath (Join-Path $assetsPath "kpr")).Path
    $values["VOLLYAI_DINOV2_ROOT"] = (Resolve-Path -LiteralPath (Join-Path $assetsPath "dinov2")).Path
    $values["VOLLYAI_DINOV2_CHECKPOINT"] = (Resolve-Path -LiteralPath (Join-Path $assetsPath "checkpoints\dinov2_vits14_reg4_pretrain.pth")).Path
    $values["VOLLYAI_KPR_PYTHON"] = (Resolve-Path -LiteralPath (Join-Path $kprRoot ".venv\Scripts\python.exe")).Path
    $values["VOLLYAI_KPR_ROOT"] = $kprRoot
    $values["VOLLYAI_KPR_CHECKPOINT"] = (Resolve-Path -LiteralPath (Join-Path $kprRoot "pretrained_models\kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar")).Path
    $values["VOLLYAI_KPR_BRIDGE"] = (Resolve-Path -LiteralPath (Join-Path $projectRoot "scripts\extract_kpr_pair_features.py")).Path
}

$lines = foreach ($entry in $values.GetEnumerator()) {
    "$($entry.Key)=$($entry.Value.ToString().Replace('\', '/'))"
}
[IO.File]::WriteAllLines($configPath, $lines, [Text.UTF8Encoding]::new($false))
Write-Host "Worker configuration written to $configPath"
Write-Host "Start with: .\scripts\start-uv-worker.ps1"
