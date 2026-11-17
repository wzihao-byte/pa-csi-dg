$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$experimentRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $experimentRoot)
$outputRoot = Join-Path $repoRoot 'outputs_tpp_level8_ablation\l1248_h10_v1'
$pythonExe = (Get-Command python.exe -ErrorAction Stop).Source
$runner = Join-Path $experimentRoot 'run_level8.py'
$analysis = Join-Path $experimentRoot 'analyze_level8.py'
$factorialAnalysis = Join-Path $experimentRoot 'analyze_factorial_l1248.py'

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

& $pythonExe $runner --execute 1> (Join-Path $outputRoot 'formal_runner.stdout.log') 2> (Join-Path $outputRoot 'formal_runner.stderr.log')
if ($LASTEXITCODE -ne 0) {
    throw "H10 Level-8 runner failed with exit code $LASTEXITCODE"
}

& $pythonExe $analysis 1> (Join-Path $outputRoot 'analysis_runner.stdout.log') 2> (Join-Path $outputRoot 'analysis_runner.stderr.log')
if ($LASTEXITCODE -ne 0) {
    throw "H10 Level-8 analysis failed with exit code $LASTEXITCODE"
}

& $pythonExe $factorialAnalysis 1> (Join-Path $outputRoot 'factorial_analysis.stdout.log') 2> (Join-Path $outputRoot 'factorial_analysis.stderr.log')
if ($LASTEXITCODE -ne 0) {
    throw "L1248 factorial analysis failed with exit code $LASTEXITCODE"
}
