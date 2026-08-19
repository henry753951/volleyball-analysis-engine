[CmdletBinding()]
param(
    [string]$AssetsRoot,
    [string]$MultitaskSdkRoot = $env:VOLLYAI_MULTITASK_SDK_ROOT,
    [string]$MultitaskSdkUrl = $env:VOLLYAI_MULTITASK_SDK_URL,
    [string]$OsnetUrl = $env:VOLLYAI_OSNET_URL,
    [string]$DinoUrl = $env:VOLLYAI_DINO_URL,
    [ValidateSet("cu130", "cpu")][string]$TorchBackend = "cu130",
    [switch]$WithReid,
    [string]$KprCheckpoint,
    [string]$KprCheckpointUrl
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($AssetsRoot)) { $AssetsRoot = Join-Path $projectRoot ".models" }
$assetsPath = [IO.Path]::GetFullPath($AssetsRoot)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it with: winget install --id=astral-sh.uv -e"
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "..\volleyball-monitoring-ai\sdk\pyproject.toml"))) {
    throw "Place volleyball-monitoring-ai beside volleyball-analysis-engine before running setup."
}
$workerPath = Join-Path $projectRoot ".venv\Scripts\volleyball-analysis-worker.exe"
$runningWorkers = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -eq $workerPath }
)
if ($runningWorkers.Count -gt 0) {
    $processIds = ($runningWorkers.ProcessId -join ", ")
    throw "Stop the running worker before updating its uv environment. Worker PID(s): $processIds"
}

Push-Location $projectRoot
try {
    uv sync --frozen --extra $TorchBackend --extra models
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE" }

    $downloadArguments = @(
        "run", "--no-sync", "python", "scripts/download_worker_models.py",
        "--assets-root", $assetsPath
    )
    if ($MultitaskSdkRoot -and -not (Test-Path -LiteralPath $MultitaskSdkRoot -PathType Container)) {
        if ($MultitaskSdkUrl) {
            Write-Host "Configured multitask SDK directory is absent; using the dynamic model URL instead."
            $MultitaskSdkRoot = $null
        }
        else {
            throw "Multitask SDK directory is missing: $MultitaskSdkRoot"
        }
    }
    if ($MultitaskSdkRoot) {
        $downloadArguments += @("--multitask-sdk-root", $MultitaskSdkRoot)
    }
    if ($MultitaskSdkUrl) {
        $downloadArguments += @("--multitask-sdk-url", $MultitaskSdkUrl)
    }
    if ($OsnetUrl) { $downloadArguments += @("--osnet-url", $OsnetUrl) }
    if ($DinoUrl) { $downloadArguments += @("--dino-url", $DinoUrl) }
    if ($WithReid) {
        $downloadArguments += "--with-reid"
        if ($KprCheckpoint) { $downloadArguments += @("--kpr-checkpoint", $KprCheckpoint) }
        if ($KprCheckpointUrl) {
            $downloadArguments += @("--kpr-checkpoint-url", $KprCheckpointUrl)
        }
    }
    & uv @downloadArguments
    if ($LASTEXITCODE -ne 0) { throw "Model preparation failed with exit code $LASTEXITCODE" }

    if ($WithReid) {
        $kprRoot = Join-Path $assetsPath "kpr"
        $kprVenv = Join-Path $kprRoot ".venv"
        $kprPython = Join-Path $kprVenv "Scripts\python.exe"
        uv python install 3.10
        if ($LASTEXITCODE -ne 0) { throw "Unable to install Python 3.10 for KPR" }
        if (-not (Test-Path -LiteralPath $kprPython -PathType Leaf)) {
            uv venv --python 3.10 --seed $kprVenv
            if ($LASTEXITCODE -ne 0) { throw "Unable to create the KPR Python 3.10 environment" }
        }
        $torchIndex = if ($TorchBackend -eq "cpu") {
            "https://download.pytorch.org/whl/cpu"
        }
        else {
            "https://download.pytorch.org/whl/cu130"
        }
        uv pip install --python $kprPython torch torchvision --index-url $torchIndex
        if ($LASTEXITCODE -ne 0) { throw "Unable to install KPR PyTorch runtime" }
        uv pip install --python $kprPython -r (Join-Path $kprRoot "requirements.txt")
        if ($LASTEXITCODE -ne 0) { throw "Unable to install KPR dependencies" }
        uv pip install --python $kprPython --editable $kprRoot
        if ($LASTEXITCODE -ne 0) { throw "Unable to install KPR source" }
    }
}
finally {
    Pop-Location
}

Write-Host "uv worker runtime is ready."
Write-Host "Next: .\scripts\configure-uv-worker.ps1 -ServerUrl <URL> -Token <TOKEN> -AssetsRoot `"$assetsPath`""
