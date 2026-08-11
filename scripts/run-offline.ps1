[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Clip,
    [Parameter(Mandatory = $true)][string]$Job,
    [string]$Output = "outputs/local-run",
    [string]$Keypoints = "",
    [switch]$SkipClipVerification,
    [switch]$Prewarm,
    [switch]$NoDebugArtifacts
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$arguments = @("run", "volleyball-analysis", "offline", "--clip", $Clip, "--job", $Job, "--output", $Output)
if ($Keypoints) {
    $arguments += @("--keypoints", $Keypoints)
}
if ($SkipClipVerification) {
    $arguments += "--skip-clip-verification"
}
if ($Prewarm) {
    $arguments += "--prewarm"
}
if ($NoDebugArtifacts) {
    $arguments += "--no-debug-artifacts"
}
Push-Location $projectRoot
try {
    & uv @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Offline inference exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
