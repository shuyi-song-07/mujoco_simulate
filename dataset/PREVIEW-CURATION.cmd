@echo off
chcp 65001 >nul
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-preview.ps1" %*
set "previewExit=%ERRORLEVEL%"
if not "%previewExit%"=="0" echo Preview stopped with an error. Read the message above.
pause
exit /b %previewExit%
