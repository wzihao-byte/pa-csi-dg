[CmdletBinding()]
param(
    [double]$AccuracyThreshold = 0.8823,
    [int]$PollSeconds = 60,
    [int]$MaxWaitHours = 8
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Layer1OutputRoot = Join-Path $RepoRoot "outputs_diag_prototype_supcon_layer1"
$Layer1LogDir = Join-Path $Layer1OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $Layer1LogDir | Out-Null

$MonitorLog = Join-Path $Layer1LogDir "k2_margin_gate_monitor.log"
$SummaryPath = Join-Path $Layer1LogDir "k2_margin_40e_gate_summary.txt"
$Layer1Experiment = "pa_csi_dg_subcenter_proto_layer1_40e_k2_margin_e2"
$Layer1MetricsPath = Join-Path $Layer1OutputRoot "dg_loeo\$Layer1Experiment\target_E2\seed_42\metrics.json"
$Layer1ConfusionPath = Join-Path $Layer1OutputRoot "dg_loeo\$Layer1Experiment\target_E2\seed_42\confusion_matrix.csv"
$Layer2Runner = Join-Path $RepoRoot "scripts\run_layer2_k2_margin.ps1"

function Write-MonitorLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $MonitorLog -Value $line
    Write-Output $line
}

function Read-Confusion {
    param([string]$Path)
    $rows = @()
    foreach ($line in Get-Content -LiteralPath $Path) {
        $rows += ,($line -split "," | ForEach-Object { [int]$_ })
    }
    return $rows
}

try {
    Write-MonitorLog "Monitor started. threshold=$AccuracyThreshold poll=${PollSeconds}s max_wait=${MaxWaitHours}h"
    $deadline = (Get-Date).AddHours($MaxWaitHours)

    while (-not (Test-Path -LiteralPath $Layer1MetricsPath)) {
        if ((Get-Date) -gt $deadline) {
            Write-MonitorLog "Timed out waiting for layer1 metrics: $Layer1MetricsPath"
            exit 1
        }

        $running = Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "python" -and
                $_.CommandLine -match "pa_csi_dg_subcenter_proto_layer1_40e_k2_margin_e2"
            }
        if (-not $running) {
            Write-MonitorLog "Layer1 process is not running and metrics are missing: $Layer1MetricsPath"
            exit 1
        }

        Write-MonitorLog "Waiting for layer1 metrics; PID(s)=$($running.ProcessId -join ',')"
        Start-Sleep -Seconds $PollSeconds
    }

    $metrics = Get-Content -LiteralPath $Layer1MetricsPath -Raw | ConvertFrom-Json
    $accuracy = [double]$metrics.test_metrics.accuracy
    $f1 = [double]$metrics.test_metrics.f1_macro

    $summary = @()
    $summary += "K=2 + PairMargin Layer-1 40e E2"
    $summary += "metrics: $Layer1MetricsPath"
    $summary += ("accuracy: {0:P2}" -f $accuracy)
    $summary += ("macro_f1: {0:P2}" -f $f1)

    if (Test-Path -LiteralPath $Layer1ConfusionPath) {
        $conf = Read-Confusion -Path $Layer1ConfusionPath
        $summary += "key_confusions: 1->4=$($conf[1][4]), 4->1=$($conf[4][1]), 0->4=$($conf[0][4])"
    }

    if ($accuracy -ge $AccuracyThreshold) {
        $summary += ("decision: RUN 200e because {0:P2} >= threshold {1:P2}" -f $accuracy, $AccuracyThreshold)
        $summary | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
        Write-MonitorLog ($summary -join " | ")
        Write-MonitorLog "Starting layer2 runner: $Layer2Runner"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Layer2Runner
        $exitCode = $LASTEXITCODE
        Write-MonitorLog "Layer2 runner exited with code $exitCode"
        exit $exitCode
    }

    $summary += ("decision: DO NOT RUN 200e because {0:P2} < threshold {1:P2}" -f $accuracy, $AccuracyThreshold)
    $summary | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
    Write-MonitorLog ($summary -join " | ")
    exit 0
} catch {
    Write-MonitorLog "UNHANDLED ERROR: $($_.Exception.Message)"
    Write-MonitorLog "ERROR DETAIL: $($_ | Out-String)"
    exit 1
}
