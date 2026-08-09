[CmdletBinding()]
param(
    [string]$Clip = "",
    [switch]$LoadModels
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$arguments = @("run", "volleyball-analysis", "doctor")
if ($Clip) {
    $arguments += @("--clip", $Clip)
}
if ($LoadModels) {
    $arguments += "--load-models"
}
Push-Location $projectRoot
try {
    & uv @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Doctor exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
