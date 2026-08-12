[CmdletBinding()]
param(
    [string]$Rtv4Archive = "E:\User\Downloads\volleyball_ball_action.zip",
    [string]$Rtv4Checkpoint = "E:\User\Downloads\best_stg1.pth",
    [ValidateSet("cu130", "cpu")]
    [string]$TorchBackend = "cu130",
    [switch]$RefreshAssets
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$artifactRoot = Join-Path $projectRoot ".artifacts"
$rtv4Target = Join-Path $artifactRoot "rtv4"
$modelTarget = Join-Path $artifactRoot "models"

foreach ($required in @($Rtv4Archive, $Rtv4Checkpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required asset not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path $artifactRoot, $modelTarget | Out-Null
if ($RefreshAssets -and (Test-Path -LiteralPath $rtv4Target)) {
    $resolvedTarget = (Resolve-Path -LiteralPath $rtv4Target).Path
    if (-not $resolvedTarget.StartsWith($artifactRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to refresh asset path outside .artifacts: $resolvedTarget"
    }
    Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
}
if (-not (Test-Path -LiteralPath $rtv4Target)) {
    Expand-Archive -LiteralPath $Rtv4Archive -DestinationPath $rtv4Target
}

function Copy-ModelAsset {
    param(
        [Parameter(Mandatory)]
        [string]$Source,
        [Parameter(Mandatory)]
        [string]$Destination
    )

    $sourceItem = Get-Item -LiteralPath $Source
    $destinationItem = Get-Item -LiteralPath $Destination -ErrorAction SilentlyContinue
    if ($RefreshAssets -or $null -eq $destinationItem -or $destinationItem.Length -ne $sourceItem.Length) {
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

Copy-ModelAsset -Source $Rtv4Checkpoint -Destination (Join-Path $modelTarget "best_stg1.pth")

Push-Location $projectRoot
try {
    uv sync --extra $TorchBackend --extra models --extra dev
    uv run --no-sync volleyball-analysis doctor --load-models
}
finally {
    Pop-Location
}
