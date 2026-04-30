@echo off
REM Bot keeper: relaunches the Telegram listener if it's not running.
REM Scheduled to run every 15 min by install_schedule.cmd.

setlocal
cd /d "%~dp0\.."

REM Check if a python process from this venv is alive
powershell -Command "$p = Get-Process | Where-Object { $_.ProcessName -eq 'python' -and $_.Path -like '*football-prediction-bot*venv*' }; if ($p) { exit 0 } else { exit 1 }"

if %errorlevel%==0 (
  REM Bot already running; nothing to do
  exit /b 0
)

REM Launch the bot in a minimized window so it survives this Task Scheduler run
start /min cmd /c "%~dp0run_bot.cmd"

endlocal
