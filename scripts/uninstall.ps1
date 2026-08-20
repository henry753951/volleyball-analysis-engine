[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "High")]
param(
    [ValidateSet("docker", "uv", "all")][string]$Mode = "all",
    [string]$ProjectPrefix = "analysis-worker",
    [string]$AssetsRoot,
    [switch]$PurgeModels,
    [switch]$PurgeDockerVolumes,
    [switch]$PurgeDockerImage
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

if ($Mode -in @("docker", "all")) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Warning "docker is not installed; skipped Docker cleanup."
    }
    else {
        $projects = [Collections.Generic.HashSet[string]]::new()
        $containerNames = @(docker ps -a --format "{{.Names}}")
        $projectPattern = "^(?<project>$([Regex]::Escape($ProjectPrefix))-(?:gpu\d+|cpu))-analysis-worker-\d+$"
        foreach ($containerName in $containerNames) {
            if ($containerName -match $projectPattern) {
                $project = $Matches["project"]
                [void]$projects.Add($project)
            }
        }
        foreach ($project in $projects) {
            $composeArgs = @("--project-name", $project, "-f", (Join-Path $projectRoot "compose.yaml"), "-f", (Join-Path $projectRoot "compose.gpu.yaml"))
            $volumeFlag = if ($PurgeDockerVolumes) { @("--volumes") } else { @() }
            if ($PSCmdlet.ShouldProcess($project, "stop and remove Docker worker")) {
                & docker compose @composeArgs down --remove-orphans @volumeFlag
                if ($LASTEXITCODE -ne 0) { throw "Docker cleanup failed for project $project" }
            }
        }
        if ($PurgeDockerImage -and $PSCmdlet.ShouldProcess("volleyball-analysis-engine:local", "remove Docker image")) {
            & docker image rm volleyball-analysis-engine:local 2>$null
        }
    }
}

if ($Mode -in @("uv", "all")) {
    $venvRoot = Join-Path $projectRoot ".venv"
    $workerProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($venvRoot, [StringComparison]::OrdinalIgnoreCase)
    })
    foreach ($process in $workerProcesses) {
        if ($PSCmdlet.ShouldProcess("PID $($process.ProcessId)", "stop uv worker")) {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    $configPath = Join-Path $projectRoot ".env.worker"
    if ((Test-Path -LiteralPath $configPath -PathType Leaf) -and $PSCmdlet.ShouldProcess($configPath, "remove worker secret configuration")) {
        Remove-Item -LiteralPath $configPath -Force
    }
    if ((Test-Path -LiteralPath $venvRoot -PathType Container) -and $PSCmdlet.ShouldProcess($venvRoot, "remove uv environment")) {
        Remove-Item -LiteralPath $venvRoot -Recurse -Force
    }
}

if ($PurgeModels -and (Test-Path -LiteralPath $assetsPath)) {
    if ($PSCmdlet.ShouldProcess($assetsPath, "remove downloaded model assets")) {
        Remove-Item -LiteralPath $assetsPath -Recurse -Force
    }
}

Write-Host "Worker uninstall complete. Source code and .env were preserved."
