param([string]$Task = '', [int]$Port = 0, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$settings = Get-Content -LiteralPath (Join-Path $projectRoot 'config/environments.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$pythonExe = $settings.training_python
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Dataset Python missing: $pythonExe. Edit config/environments.json -> training_python."
}
$prefix = Split-Path -Parent $pythonExe
$env:PATH = "$prefix;$(Join-Path $prefix 'Library/bin');$(Join-Path $prefix 'Scripts');$env:PATH"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONNOUSERSITE = '1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$arguments = @('dataset/manage_dataset.py')
if ($Task) { $arguments += @('--task', $Task) }
$arguments += 'preview-menu'
if ($Port -ne 0) { $arguments += @('--port', "$Port") }
if ($NoBrowser) { $arguments += '--no-browser' }
& $pythonExe @arguments
exit $LASTEXITCODE
