@echo off
REM Weekly retrain wrapper for Windows Task Scheduler.
REM Schedule: Sundays at 04:00 via install_schedule.cmd or manually.

setlocal
cd /d "%~dp0\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set LOGFILE=logs\retrain_%date:~10,4%%date:~4,2%%date:~7,2%.log
echo Retrain run %date% %time% >> %LOGFILE%
.venv\Scripts\python.exe -W ignore::ResourceWarning scripts\weekly_retrain.py >> %LOGFILE% 2>&1

endlocal
