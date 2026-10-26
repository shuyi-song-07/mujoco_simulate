param([ValidateSet('collection','training')][string]$Environment='collection',[string[]]$Arguments=@('-m','collection','menu'))
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
$settings=Get-Content -LiteralPath (Join-Path $project 'config/environments.json') -Raw | ConvertFrom-Json
$python=if($Environment -eq 'collection'){$settings.collection_python}else{$settings.training_python}
if(-not(Test-Path -LiteralPath $python)){throw "Python missing: $python; edit config/environments.json"}
$prefix=Split-Path -Parent $python
$env:PATH="$prefix;$(Join-Path $prefix 'Library/bin');$(Join-Path $prefix 'Scripts');$env:PATH"
$env:PYTHONUTF8='1'
$env:PYTHONNOUSERSITE='1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $python @Arguments
exit $LASTEXITCODE
