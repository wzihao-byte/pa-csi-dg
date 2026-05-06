[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = "C:\ProgramData\anaconda3\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

$OutputRoot = Join-Path $RepoRoot "outputs_diag_prototype_supcon_layer2"
$LogDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$RunnerLog = Join-Path $LogDir "layer2_k2_margin_runner.log"
$RunLog = Join-Path $LogDir "k2_margin_200e.log"
$Config = "configs\pa_csi_dg_subcenter_proto_layer2_200e_k2_margin_e2.json"
$Experiment = "pa_csi_dg_subcenter_proto_layer2_200e_k2_margin_e2"
$RunRoot = Join-Path $OutputRoot "dg_loeo\$Experiment"
$MetricsPath = Join-Path $RunRoot "target_E2\seed_42\metrics.json"

function Write-RunnerLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $RunnerLog -Value $line
    Write-Output $line
}

try {
    Write-RunnerLog "Layer 2 K2 margin runner started. Force=${Force}"

    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $Config))) {
        throw "Missing config: $Config"
    }

    if ($WhatIfPreference) {
        Write-RunnerLog "WhatIf: would run $Config"
        exit 0
    }

    if ((Test-Path -LiteralPath $MetricsPath) -and (-not $Force)) {
        Write-RunnerLog "Skipping run; metrics already exist at $MetricsPath"
        exit 0
    }

    if ($Force -and (Test-Path -LiteralPath $RunRoot)) {
        $resolvedRoot = (Resolve-Path -LiteralPath $RunRoot).Path
        $resolvedOutput = (Resolve-Path -LiteralPath $OutputRoot).Path
        if (-not $resolvedRoot.StartsWith($resolvedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove outside ${resolvedOutput}: ${resolvedRoot}"
        }
        Write-RunnerLog "Removing existing output: $RunRoot"
        Remove-Item -LiteralPath $RunRoot -Recurse -Force
    }

    Write-RunnerLog "Starting: $PythonExe -u train_dg.py --config $Config"
    $trainCommand = "`"$PythonExe`" -u train_dg.py --config `"$Config`" > `"$RunLog`" 2>&1"
    & cmd.exe /d /c $trainCommand
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-RunnerLog "FAILED with exit code $exitCode. See $RunLog"
        exit $exitCode
    }
    if (-not (Test-Path -LiteralPath $MetricsPath)) {
        Write-RunnerLog "FAILED; metrics.json was not written. See $RunLog"
        exit 1
    }

    Write-RunnerLog "Completed; metrics at $MetricsPath"
} catch {
    Write-RunnerLog "UNHANDLED ERROR: $($_.Exception.Message)"
    Write-RunnerLog "ERROR DETAIL: $($_ | Out-String)"
    exit 1
}
