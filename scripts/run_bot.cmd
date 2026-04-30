@echo off
REM Run the Telegram bot in this window. Keep the window open while you want
REM the bot to be responsive. Close it (or Ctrl+C) to stop the bot.
REM
REM To run minimized in the background, use:
REM   start /min cmd /c run_bot.cmd

setlocal
cd /d "%~dp0\.."

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

.venv\Scripts\python.exe -W ignore::ResourceWarning scripts\telegram_bot.py

endlocal
