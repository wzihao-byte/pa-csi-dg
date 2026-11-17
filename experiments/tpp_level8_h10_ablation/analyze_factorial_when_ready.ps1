$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$experimentRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $experimentRoot)
$h10Output = Join-Path $repoRoot 'outputs_tpp_level8_ablation\l1248_h10_v1'
$h10Analysis = Join-Path $h10Output 'analysis\level8_analysis.json'
$runnerError = Join-Path $h10Output 'formal_runner.stderr.log'
$pythonExe = (Get-Command python.exe -ErrorAction Stop).Source
$analysisScript = Join-Path $experimentRoot 'analyze_factorial_l1248.py'

while (-not (Test-Path -LiteralPath $h10Analysis)) {
    if (-not (Get-Process -Id 4776 -ErrorAction SilentlyContinue)) {
        $tail = if (Test-Path -LiteralPath $runnerError) { Get-Content -LiteralPath $runnerError -Tail 40 } else { @('runner log missing') }
        throw "H10 pipeline exited before analysis was created: $($tail -join ' | ')"
    }
    Start-Sleep -Seconds 60
}

& $pythonExe $analysisScript 1> (Join-Path $h10Output 'factorial_analysis.stdout.log') 2> (Join-Path $h10Output 'factorial_analysis.stderr.log')
if ($LASTEXITCODE -ne 0) {
    throw "L1248 factorial analysis failed with exit code $LASTEXITCODE"
}

