[CmdletBinding()]
param(
    [string]$MultitaskSdkRoot,
    [ValidateSet("cu130", "cpu")]
    [string]$TorchBackend = "cu130"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($MultitaskSdkRoot)) {
    $MultitaskSdkRoot = Join-Path $projectRoot "src"
}
$sdkPackage = Join-Path $MultitaskSdkRoot "volleyball_sdk\__init__.py"
$checkpoint = Join-Path $projectRoot ".models\volleyball_multitask\best.pth"
foreach ($required in @($sdkPackage, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required multitask SDK asset not found: $required"
    }
}
$env:VOLLYAI_MULTITASK_SDK_ROOT = $MultitaskSdkRoot
$env:VOLLYAI_MULTITASK_CHECKPOINT = $checkpoint

Push-Location $projectRoot
try {
    uv sync --extra $TorchBackend --extra models --extra dev
    uv run --no-sync volleyball-analysis doctor --load-models
}
finally {
    Pop-Location
}
