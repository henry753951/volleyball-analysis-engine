[CmdletBinding()]
param(
    [ValidateSet("docker", "uv")][string]$Mode = "docker",
    [string]$ServerUrl = "https://volleyai.hsulab.net/",
    [string]$CentralHttpUrl = "https://volleyai.hsulab.net/",
    [string]$Token = $env:VOLLYAI_TOKEN,
    [switch]$CreateLocalToken,
    [string]$InstancePrefix = "analysis-worker",
    [string]$AssetsRoot,
    [string]$MultitaskSdkRoot = $env:VOLLYAI_MULTITASK_SDK_ROOT,
    [string]$MultitaskSdkUrl = $env:VOLLYAI_MULTITASK_SDK_URL,
    [string]$OsnetUrl = $env:VOLLYAI_OSNET_URL,
    [string]$DinoUrl = $env:VOLLYAI_DINO_URL,
    [ValidateSet("auto", "cu130", "cpu")][string]$TorchBackend = "auto",
    [string[]]$GpuIds,
    [switch]$WithReid,
    [switch]$SkipModelDownload,
    [switch]$NoStart,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($AssetsRoot)) { $AssetsRoot = Join-Path $projectRoot ".models" }
$assetsPath = if ([IO.Path]::IsPathRooted($AssetsRoot)) {
    [IO.Path]::GetFullPath($AssetsRoot)
}
else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $AssetsRoot))
}
$envPath = Join-Path $projectRoot ".env"

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required."
    }
}

function Get-NvidiaGpuIds {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return @() }
    $ids = @(nvidia-smi --query-gpu=index --format=csv,noheader 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    return $ids
}

function Resolve-WorkerWebSocketUrl([string]$Value) {
    $normalized = $Value.Trim().TrimEnd("/")
    if ($normalized.StartsWith("https://")) { $normalized = "wss://" + $normalized.Substring(8) }
    elseif ($normalized.StartsWith("http://")) { $normalized = "ws://" + $normalized.Substring(7) }
    $normalized = $normalized.Replace("/api/v1/ai/providers/ws", "/api/v2/ai/providers/ws")
    if (-not $normalized.EndsWith("/api/v2/ai/providers/ws")) { $normalized += "/api/v2/ai/providers/ws" }
    if (-not ($normalized.StartsWith("ws://") -or $normalized.StartsWith("wss://"))) {
        throw "ServerUrl must be an HTTP(S) base URL or WS(S) URL."
    }
    return $normalized
}

function Get-WorkerToken {
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
            -Body (@{ name = "$InstancePrefix-$(Get-Date -Format 'yyyyMMdd-HHmmss')" } | ConvertTo-Json -Compress)
        return $created.token
    }
    if ([string]::IsNullOrWhiteSpace($Token)) {
        $secureToken = Read-Host "Worker Access Token" -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        try { $Token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    }
    if ($Token.Length -lt 16 -or $Token.Contains("`n") -or $Token.Contains("`r")) {
        throw "Worker token must be a single line containing at least 16 characters."
    }
    return $Token
}

function Set-DotEnvValues([hashtable]$Values) {
    if (Test-Path -LiteralPath $envPath -PathType Leaf) {
        $source = @(Get-Content -LiteralPath $envPath)
    }
    else {
        $source = @(Get-Content -LiteralPath (Join-Path $projectRoot ".env.example"))
    }
    $written = @{}
    $output = foreach ($line in $source) {
        if ($line -match '^([A-Z0-9_]+)=') {
            $name = $Matches[1]
            if ($Values.ContainsKey($name)) {
                $written[$name] = $true
                "$name=$($Values[$name])"
                continue
            }
        }
        $line
    }
    foreach ($entry in $Values.GetEnumerator()) {
        if (-not $written.ContainsKey($entry.Key)) { $output += "$($entry.Key)=$($entry.Value)" }
    }
    [IO.File]::WriteAllLines($envPath, $output, [Text.UTF8Encoding]::new($false))
}

if ($Mode -eq "docker") {
    Require-Command "docker"
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "..\volleyball-monitoring-ai\sdk\pyproject.toml") -PathType Leaf)) {
        throw "Docker build requires volleyball-monitoring-ai beside volleyball-analysis-engine."
    }
}
else {
    Require-Command "uv"
}

$detectedGpuIds = @(Get-NvidiaGpuIds)
if ($TorchBackend -eq "auto") {
    $TorchBackend = if ($detectedGpuIds.Count -gt 0) { "cu130" } else { "cpu" }
}
if (-not $GpuIds -or $GpuIds.Count -eq 0) {
    $GpuIds = if ($TorchBackend -eq "cu130") { $detectedGpuIds } else { @() }
}
if ($TorchBackend -eq "cu130" -and $GpuIds.Count -eq 0) {
    throw "CUDA backend selected but no NVIDIA GPU was detected. Use -TorchBackend cpu or pass -GpuIds explicitly."
}
if ($TorchBackend -eq "cpu" -and $GpuIds.Count -gt 0) {
    throw "-GpuIds cannot be used with -TorchBackend cpu."
}

$token = if ($Mode -eq "uv" -and $CreateLocalToken) { "" } else { Get-WorkerToken }
$webSocketUrl = Resolve-WorkerWebSocketUrl $ServerUrl
$setupScript = Join-Path $projectRoot "scripts\setup-uv-worker.ps1"
if (-not $SkipModelDownload) {
    $setupParameters = @{
        AssetsRoot = $assetsPath
        TorchBackend = $TorchBackend
    }
    if ($MultitaskSdkRoot) { $setupParameters.MultitaskSdkRoot = $MultitaskSdkRoot }
    if ($MultitaskSdkUrl) { $setupParameters.MultitaskSdkUrl = $MultitaskSdkUrl }
    if ($OsnetUrl) { $setupParameters.OsnetUrl = $OsnetUrl }
    if ($DinoUrl) { $setupParameters.DinoUrl = $DinoUrl }
    if ($WithReid) { $setupParameters.WithReid = $true }
    & $setupScript @setupParameters
}

if ($Mode -eq "uv") {
    $configureScript = Join-Path $projectRoot "scripts\configure-uv-worker.ps1"
    $configureParameters = @{
        ServerUrl = $ServerUrl
        CentralHttpUrl = $CentralHttpUrl
        AssetsRoot = $assetsPath
        InstanceId = "$InstancePrefix-uv"
        Force = $true
    }
    if ($CreateLocalToken) { $configureParameters.CreateLocalToken = $true }
    else { $configureParameters.Token = $token }
    if ($MultitaskSdkRoot) { $configureParameters.MultitaskSdkRoot = $MultitaskSdkRoot }
    if ($WithReid) { $configureParameters.WithReid = $true }
    & $configureScript @configureParameters
    if (-not $NoStart) {
        $startScript = Join-Path $projectRoot "scripts\start-uv-worker.ps1"
        if ($GpuIds.Count -gt 0) {
            foreach ($gpuId in $GpuIds) {
                & $startScript -Background -Device "cuda:$gpuId" -InstanceId "$InstancePrefix-gpu$gpuId"
            }
        }
        else {
            & $startScript -Background -Device "cpu" -InstanceId "$InstancePrefix-cpu"
        }
    }
    Write-Host "uv worker installation complete."
    exit 0
}

$sdkHostRoot = if ($MultitaskSdkRoot) { [IO.Path]::GetFullPath($MultitaskSdkRoot) } else { Join-Path $assetsPath "volleyball_inference_sdk" }
$smpHostRoot = Join-Path $assetsPath "selective-mask-propagation"
$envValues = @{
    VOLLYAI_SERVER_WS_URL = $webSocketUrl
    VOLLYAI_TOKEN = $token
    VOLLYAI_INSTANCE_ID = "$InstancePrefix-docker"
    VOLLYAI_DEVICE = if ($TorchBackend -eq "cpu") { "cpu" } else { "cuda:0" }
    VOLLYAI_TORCH_BACKEND = $TorchBackend
    VOLLYAI_WORKER_IMAGE = "volleyball-analysis-engine:local"
    VOLLYAI_MULTITASK_SDK_HOST_ROOT = $sdkHostRoot.Replace("\", "/")
    VOLLYAI_SMP_HOST_ROOT = $smpHostRoot.Replace("\", "/")
}
Set-DotEnvValues $envValues

$composeBase = @("--env-file", $envPath, "-f", (Join-Path $projectRoot "compose.yaml"))
Push-Location $projectRoot
try {
    & docker compose @composeBase build analysis-worker
    if ($LASTEXITCODE -ne 0) { throw "Docker image build failed with exit code $LASTEXITCODE" }
    if (-not $NoStart) {
        if ($GpuIds.Count -gt 0) {
            foreach ($gpuId in $GpuIds) {
                $env:VOLLYAI_GPU_ID = [string]$gpuId
                $env:VOLLYAI_INSTANCE_ID = "$InstancePrefix-gpu$gpuId"
                $composeGpu = @("--project-name", "$InstancePrefix-gpu$gpuId", "--env-file", $envPath, "-f", (Join-Path $projectRoot "compose.yaml"), "-f", (Join-Path $projectRoot "compose.gpu.yaml"))
                & docker compose @composeGpu up -d --no-build
                if ($LASTEXITCODE -ne 0) { throw "Docker worker start failed for GPU $gpuId" }
            }
        }
        else {
            & docker compose @composeBase --project-name "$InstancePrefix-cpu" up -d --no-build
            if ($LASTEXITCODE -ne 0) { throw "Docker worker start failed" }
        }
    }
}
finally {
    Remove-Item Env:VOLLYAI_GPU_ID -ErrorAction SilentlyContinue
    Remove-Item Env:VOLLYAI_INSTANCE_ID -ErrorAction SilentlyContinue
    Pop-Location
}
Write-Host "Docker worker installation complete. Configuration: $envPath"
