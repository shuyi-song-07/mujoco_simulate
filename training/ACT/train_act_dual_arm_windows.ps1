param(
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [string]$OutputDir,
    [string]$RepoId = 'local/task3_dual_arm',
    [int]$Steps = 100000,
    [int]$BatchSize = 8,
    [int]$Seed = 1000,
    [int]$NumWorkers = 0,
    [double]$EvalSplit = 0.1,
    [int]$ChunkSize = 50,
    [string]$Resume,
    [string]$PretrainedPath,
    [switch]$NoPretrainedBackbone,
    [switch]$SmokeTest,
    [switch]$PrintCommand
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $projectRoot
. (Join-Path $projectRoot 'scripts\use-conda.ps1')
$trainArgs = @((Join-Path $PSScriptRoot 'train_act_dual_arm.py'), '--dataset-root', $DatasetRoot,
    '--repo-id', $RepoId, '--device', 'cuda', '--seed', "$Seed", '--num-workers', "$NumWorkers",
    '--eval-split', "$EvalSplit", '--chunk-size', "$ChunkSize")
if ((-not $SmokeTest -and -not $Resume) -or $PSBoundParameters.ContainsKey('Steps')) { $trainArgs += @('--steps', "$Steps") }
if ((-not $SmokeTest -and -not $Resume) -or $PSBoundParameters.ContainsKey('BatchSize')) { $trainArgs += @('--batch-size', "$BatchSize") }
if ($OutputDir) { $trainArgs += @('--output-dir', $OutputDir) }
if ($Resume) { $trainArgs += @('--resume', $Resume) }
if ($PretrainedPath) { $trainArgs += @('--pretrained-path', $PretrainedPath) }
if ($NoPretrainedBackbone) { $trainArgs += '--no-pretrained-backbone' }
if ($SmokeTest) { $trainArgs += '--smoke-test' }
if ($PrintCommand) { $trainArgs += '--print-command' }
& $projectPython @trainArgs
if ($LASTEXITCODE -ne 0) { throw "Task 3 ACT training exited with code $LASTEXITCODE" }
