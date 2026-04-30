@echo off
REM Hourly auto-resolution wrapper for Windows Task Scheduler.
REM Runs scripts\resolve_picks.py and logs to logs\resolve_YYYYMMDD.log

setlocal
cd /d "%~dp0\.."

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set LOGFILE=logs\resolve_%date:~10,4%%date:~4,2%%date:~7,2%.log

echo Resolve run %date% %time% >> %LOGFILE%
.venv\Scripts\python.exe -W ignore::ResourceWarning scripts\resolve_picks.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

endlocal
