[CmdletBinding()]
param(
    [ValidateSet(
        "stage-a",
        "stage-a-confirm-200",
        "postprocess",
        "build-margin-tables",
        "margin-raw-40",
        "margin-raw-joint-40",
        "margin-quantile-40",
        "margin-quantile-joint-40",
        "margin-search-40",
        "build-shortlist-plan",
        "margin-shortlist-40",
        "confirm-200",
        "aggregate",
        "all"
    )]
    [string]$Stage = "stage-a",

    [string]$Device = "cuda",
    [string]$OutputRoot = "",
    [string]$BaseConfig = "",
    [string[]]$Targets = @(),
    [string[]]$KValues = @(),
    [string[]]$Seeds = @(),
    [int]$MaxRuns = 0,
    [int]$Epochs = 0,
    [switch]$Force,
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = "C:\ProgramData\anaconda3\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "results\k_margin_search"
}
if ([string]::IsNullOrWhiteSpace($BaseConfig)) {
    $BaseConfig = Join-Path $RepoRoot "configs\pa_csi_dg_subcenter_proto_layer1_40e_k2_nopair_e2.json"
}

$LogDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutLog = Join-Path $LogDir "$Stamp`_$Stage.stdout.log"
$StderrLog = Join-Path $LogDir "$Stamp`_$Stage.stderr.log"

$ArgList = @(
    "-u",
    "scripts\k_margin_search.py",
    "--stage", $Stage,
    "--base-config", $BaseConfig,
    "--output-root", $OutputRoot,
    "--device", $Device
)

function Expand-CommaList {
    param([string[]]$Values)
    $expanded = @()
    foreach ($Value in $Values) {
        foreach ($Part in ($Value -split ",")) {
            $trimmed = $Part.Trim()
            if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
                $expanded += $trimmed
            }
        }
    }
    return $expanded
}

$ExpandedTargets = Expand-CommaList -Values $Targets
$ExpandedKValues = Expand-CommaList -Values $KValues
$ExpandedSeeds = Expand-CommaList -Values $Seeds

if ($ExpandedTargets.Count -gt 0) {
    $ArgList += "--targets"
    foreach ($TargetId in $ExpandedTargets) {
        if ($TargetId.StartsWith("E", [System.StringComparison]::OrdinalIgnoreCase)) {
            $ArgList += $TargetId.ToUpperInvariant()
        } else {
            $ArgList += "E$TargetId"
        }
    }
}
if ($ExpandedKValues.Count -gt 0) {
    $ArgList += "--k-values"
    foreach ($K in $ExpandedKValues) {
        $ArgList += [string]$K
    }
}
if ($ExpandedSeeds.Count -gt 0) {
    $ArgList += "--seeds"
    foreach ($Seed in $ExpandedSeeds) {
        $ArgList += [string]$Seed
    }
}
if ($MaxRuns -gt 0) {
    $ArgList += @("--max-runs", [string]$MaxRuns)
}
if ($Epochs -gt 0) {
    $ArgList += @("--epochs", [string]$Epochs)
}
if ($Force) {
    $ArgList += "--force"
}

Write-Output "Python: $PythonExe"
Write-Output "Stage: $Stage"
Write-Output "OutputRoot: $OutputRoot"
Write-Output "StdoutLog: $StdoutLog"
Write-Output "StderrLog: $StderrLog"

if ($Background) {
    $process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $ArgList `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -WindowStyle Hidden `
        -PassThru
    $PidPath = Join-Path $LogDir "$Stamp`_$Stage.pid"
    Set-Content -LiteralPath $PidPath -Value $process.Id
    Write-Output "Started background process PID=$($process.Id)"
    Write-Output "PID file: $PidPath"
    exit 0
}

& $PythonExe @ArgList 1> $StdoutLog 2> $StderrLog
$ExitCode = $LASTEXITCODE
Write-Output "ExitCode: $ExitCode"
if ($ExitCode -ne 0) {
    exit $ExitCode
}
