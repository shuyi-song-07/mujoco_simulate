@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1"
pause
Set-Location 'D:\轻量化数据采集\paperA'
$envs = Get-Content config/environments.json -Raw | ConvertFrom-Json
$collect = $envs.collection_python
$train = $envs.training_python
& $collect -m collection list
& $collect -m collection start --task tray_transfer --operator-id operator01