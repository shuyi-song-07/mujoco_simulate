param(
    [ValidateSet('pick_place', 'push_t', 'dual_arm')][string]$Task = 'pick_place',
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
. (Join-Path $PSScriptRoot 'use-conda.ps1')
$localDir = Join-Path $projectRoot '.local'
New-Item -ItemType Directory -Path $localDir -Force | Out-Null
$webUrl = if ($Task -eq 'dual_arm') { 'http://127.0.0.1:8000/dual_arm.html' } else { 'http://127.0.0.1:8000/' }
$backendPort = if ($Task -eq 'dual_arm') { 5002 } else { 5001 }
$healthUrl = "http://127.0.0.1:$backendPort/health"
$pidName = if ($Task -eq 'dual_arm') { 'task3.pid' } else { 'simulation.pid' }
$logName = if ($Task -eq 'dual_arm') { 'task3' } else { 'simulation' }
$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:PORT = '8000'
$env:BACKEND_PORT = '5001'
$env:DUAL_BACKEND_PORT = '5002'
$webReady = $false
$response = $null
try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $webUrl -TimeoutSec 2
    $webReady = $response.StatusCode -eq 200 -and $response.Headers['X-Project-Server'] -eq 'mujoco-local'
    if (-not $webReady) { throw 'Port 8000 is occupied by another service. Close it before starting.' }
} catch {
    if ($response) { throw }
}
if (-not $webReady) {
    $web = Start-Process -FilePath $projectNode -ArgumentList 'teleoperation/mediapipe/server.js' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'web.stdout.log') -RedirectStandardError (Join-Path $logDir 'web.stderr.log') -PassThru
    $web.Id | Set-Content -LiteralPath (Join-Path $localDir 'web.pid')
    $deadline = (Get-Date).AddSeconds(15)
    do {
        Start-Sleep -Milliseconds 300
        if ($web.HasExited) { throw 'Web service exited. See logs/web.stderr.log.' }
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $webUrl -TimeoutSec 2
            $webReady = $response.StatusCode -eq 200 -and $response.Headers['X-Project-Server'] -eq 'mujoco-local'
        } catch {}
    } until ($webReady -or (Get-Date) -gt $deadline)
    if (-not $webReady) { throw 'Web service startup timed out. See logs/web.stderr.log.' }
}
$simulationReady = $false
try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    $simulationReady = $health.ok
    if ($simulationReady -and $health.task_mode -ne $Task) {
        throw 'Another task is running. Save the episode and finish recording before switching tasks.'
    }
} catch {
    if ($simulationReady) { throw }
}
if (-not $simulationReady) {
    $entryArgs = if ($Task -eq 'dual_arm') { @('-u', '-m', 'simulation.mujoco.dual_arm.record_mujoco_dual_arm') } elseif ($Task -eq 'push_t') { @('-u', 'simulation/mujoco/record_mujoco_push_t.py') } else { @('-u', 'simulation/mujoco/record_mujoco_panda.py') }
    $simulation = Start-Process -FilePath $projectPython -ArgumentList $entryArgs -WorkingDirectory $projectRoot -WindowStyle Normal -RedirectStandardOutput (Join-Path $logDir "$logName.stdout.log") -RedirectStandardError (Join-Path $logDir "$logName.stderr.log") -PassThru
    $simulation.Id | Set-Content -LiteralPath (Join-Path $localDir $pidName)
    $deadline = (Get-Date).AddSeconds(150)
    do {
        Start-Sleep -Seconds 2
        if ($simulation.HasExited) {
            Get-Content -LiteralPath (Join-Path $logDir "$logName.stderr.log") -Tail 30
            throw "Simulation exited. See logs/$logName.stderr.log."
        }
        try { $simulationReady = (Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2).ok } catch {}
    } until ($simulationReady -or (Get-Date) -gt $deadline)
    if (-not $simulationReady) { throw 'Simulation startup timed out. Check logs.' }
}
Write-Host 'Desktop simulation is ready.'
Write-Host "Open $webUrl on this desktop and allow camera access."
Write-Host 'Keep the MuJoCo window open. Save success/failure before ending recording.'
if (-not $NoBrowser) { Start-Process $webUrl }
