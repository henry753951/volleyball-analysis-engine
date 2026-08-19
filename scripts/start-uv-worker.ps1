[CmdletBinding()]
param(
    [string]$Config,
    [string]$Device,
    [string]$InstanceId,
    [switch]$Background
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($Config)) { $Config = Join-Path $projectRoot ".env.worker" }
$configPath = (Resolve-Path -LiteralPath $Config).Path
$workerExecutable = Join-Path $projectRoot ".venv\Scripts\volleyball-analysis-worker.exe"
if (-not (Test-Path -LiteralPath $workerExecutable -PathType Leaf)) {
    throw "Worker executable is missing. Run scripts/setup-uv-worker.ps1 first."
}

foreach ($rawLine in [IO.File]::ReadAllLines($configPath)) {
    $line = $rawLine.Trim()
    if (-not $line -or $line.StartsWith("#")) { continue }
    $separator = $line.IndexOf("=")
    if ($separator -lt 1) { throw "Invalid worker config line: $rawLine" }
    $name = $line.Substring(0, $separator)
    if ($name -notmatch "^VOLLYAI_[A-Z0-9_]+$") { throw "Invalid worker setting name: $name" }
    [Environment]::SetEnvironmentVariable($name, $line.Substring($separator + 1), "Process")
}
if ($Device) { $env:VOLLYAI_DEVICE = $Device }
if ($InstanceId) { $env:VOLLYAI_INSTANCE_ID = $InstanceId }

if ($Background) {
    $logRoot = Join-Path $projectRoot ".artifacts"
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $instanceSuffix = if ($InstanceId) { "-$InstanceId" } else { "" }
    $stdout = Join-Path $logRoot "uv-worker$instanceSuffix-$stamp.stdout.log"
    $stderr = Join-Path $logRoot "uv-worker$instanceSuffix-$stamp.stderr.log"
    $process = Start-Process `
        -FilePath $workerExecutable `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    [PSCustomObject]@{ process_id = $process.Id; stdout = $stdout; stderr = $stderr } |
        ConvertTo-Json -Compress
}
else {
    Push-Location $projectRoot
    try {
        & $workerExecutable
        if ($LASTEXITCODE -ne 0) { throw "Worker exited with code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}
