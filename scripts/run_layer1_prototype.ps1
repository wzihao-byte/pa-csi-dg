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

$OutputRoot = Join-Path $RepoRoot "outputs_diag_prototype_supcon_layer1"
$LogDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$RunnerLog = Join-Path $LogDir "layer1_runner.log"

function Write-RunnerLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $RunnerLog -Value $line
    Write-Output $line
}

function Assert-UnderPath {
    param(
        [string]$Path,
        [string]$Parent
    )
    $resolvedParent = (Resolve-Path -LiteralPath $Parent).Path
    if (Test-Path -LiteralPath $Path) {
        $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
        if (-not $resolvedPath.StartsWith($resolvedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to touch path outside ${resolvedParent}: ${resolvedPath}"
        }
    }
}

$Runs = @(
    @{
        Tag = "k2_nopair"
        Config = "configs\pa_csi_dg_subcenter_proto_layer1_40e_k2_nopair_e2.json"
        Experiment = "pa_csi_dg_subcenter_proto_layer1_40e_k2_nopair_e2"
    },
    @{
        Tag = "k3_nopair"
        Config = "configs\pa_csi_dg_subcenter_proto_layer1_40e_k3_nopair_e2.json"
        Experiment = "pa_csi_dg_subcenter_proto_layer1_40e_k3_nopair_e2"
    },
    @{
        Tag = "k3_margin"
        Config = "configs\pa_csi_dg_subcenter_proto_layer1_40e_k3_margin_e2.json"
        Experiment = "pa_csi_dg_subcenter_proto_layer1_40e_k3_margin_e2"
    }
)

try {
    Write-RunnerLog "Layer 1 prototype runner started. Force=${Force}"

    foreach ($run in $Runs) {
        $configPath = Join-Path $RepoRoot $run.Config
        $runRoot = Join-Path $OutputRoot "dg_loeo\$($run.Experiment)"
        $metricsPath = Join-Path $runRoot "target_E2\seed_42\metrics.json"
        $runLog = Join-Path $LogDir "$($run.Tag).log"

        if (-not (Test-Path -LiteralPath $configPath)) {
            throw "Missing config: $configPath"
        }

        if ($WhatIfPreference) {
            Write-RunnerLog "WhatIf: would run $($run.Tag) from $($run.Config)"
            continue
        }

        if ((Test-Path -LiteralPath $metricsPath) -and (-not $Force)) {
            Write-RunnerLog "Skipping $($run.Tag); metrics already exist at $metricsPath"
            continue
        }

        if ($Force -and (Test-Path -LiteralPath $runRoot)) {
            Assert-UnderPath -Path $runRoot -Parent $OutputRoot
            Write-RunnerLog "Removing existing output for $($run.Tag): $runRoot"
            Remove-Item -LiteralPath $runRoot -Recurse -Force
        }

        Write-RunnerLog "Starting $($run.Tag): $PythonExe -u train_dg.py --config $($run.Config)"
        $trainCommand = "`"$PythonExe`" -u train_dg.py --config `"$($run.Config)`" > `"$runLog`" 2>&1"
        & cmd.exe /d /c $trainCommand
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            Write-RunnerLog "FAILED $($run.Tag) with exit code $exitCode. See $runLog"
            exit $exitCode
        }
        if (-not (Test-Path -LiteralPath $metricsPath)) {
            Write-RunnerLog "FAILED $($run.Tag); metrics.json was not written. See $runLog"
            exit 1
        }
        Write-RunnerLog "Completed $($run.Tag); metrics at $metricsPath"
    }

    if ($WhatIfPreference) {
        Write-RunnerLog "WhatIf: summary generation skipped."
        exit 0
    }

    $summaryPath = Join-Path $LogDir "layer1_summary.txt"
    Write-RunnerLog "Writing summary to $summaryPath"
    $summaryCommand = "`"$PythonExe`" -u scripts\summarize_layer1_prototype.py > `"$summaryPath`" 2>&1"
    & cmd.exe /d /c $summaryCommand
    $summaryExitCode = $LASTEXITCODE
    if ($summaryExitCode -ne 0) {
        Write-RunnerLog "Summary failed with exit code $summaryExitCode. See $summaryPath"
        exit $summaryExitCode
    }

    Write-RunnerLog "Layer 1 prototype runner completed."
} catch {
    Write-RunnerLog "UNHANDLED ERROR: $($_.Exception.Message)"
    Write-RunnerLog "ERROR DETAIL: $($_ | Out-String)"
    exit 1
}
