$ErrorActionPreference = 'Stop'
$condaPrefix = 'D:\Conda\envs\mujoco_repro'
$condaExe = 'D:\Anaconda3\Scripts\conda.exe'
if (-not (Test-Path -LiteralPath (Join-Path $condaPrefix 'python.exe'))) {
    throw "The mujoco_repro environment was not found at $condaPrefix"
}
(& $condaExe 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate $condaPrefix
if ($LASTEXITCODE -ne 0) { throw 'Could not activate mujoco_repro' }
$projectPython = Join-Path $condaPrefix 'python.exe'
$projectNode = Join-Path $condaPrefix 'node.exe'
if (-not (Test-Path -LiteralPath $projectNode)) {
    $projectNode = Join-Path $condaPrefix 'Library\bin\node.exe'
}
if (-not (Test-Path -LiteralPath $projectNode)) { throw 'Node.js is missing from mujoco_repro' }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
