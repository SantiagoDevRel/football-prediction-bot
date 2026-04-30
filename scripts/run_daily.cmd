@echo off
REM Wrapper for Windows Task Scheduler.
REM
REM Schedule with:
REM   schtasks /create /tn "FootballBotDaily" /tr "C:\Users\STZTR\Desktop\claude-code-environment\football-prediction-bot\scripts\run_daily.cmd" /sc daily /st 09:00
REM
REM This activates the venv and runs the daily pipeline. Logs go to logs/daily_YYYYMMDD.log

setlocal
cd /d "%~dp0\.."

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set LOGFILE=logs\daily_%date:~10,4%%date:~4,2%%date:~7,2%.log

echo Run started %date% %time% >> %LOGFILE%
.venv\Scripts\python.exe -W ignore::ResourceWarning scripts\daily_pipeline.py >> %LOGFILE% 2>&1
echo Run finished %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%

endlocal
