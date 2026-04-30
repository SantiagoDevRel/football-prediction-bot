@echo off
REM One-shot installer for two Windows scheduled tasks:
REM   1. FootballBotDaily   - daily pipeline at 09:00
REM   2. FootballBotResolve - hourly auto-resolution
REM   3. FootballBotPersist - keep the Telegram bot listener up (every 15 min)
REM
REM Run this ONCE from an elevated (Administrator) PowerShell or CMD.
REM Uninstall: schtasks /delete /tn FootballBotDaily /f
REM           schtasks /delete /tn FootballBotResolve /f
REM           schtasks /delete /tn FootballBotPersist /f

setlocal
set REPO=%~dp0..

echo === Installing Windows Task Scheduler entries ===

REM 1. Daily pipeline at 9 AM local time
schtasks /create /tn "FootballBotDaily" ^
  /tr "\"%REPO%\scripts\run_daily.cmd\"" ^
  /sc daily /st 09:00 ^
  /f
if errorlevel 1 echo WARNING: FootballBotDaily creation failed

REM 2. Auto-resolver every hour
schtasks /create /tn "FootballBotResolve" ^
  /tr "\"%REPO%\scripts\run_resolve.cmd\"" ^
  /sc hourly /mo 1 ^
  /f
if errorlevel 1 echo WARNING: FootballBotResolve creation failed

REM 3. Bot keeper: relaunch listener every 15 min if not running
schtasks /create /tn "FootballBotPersist" ^
  /tr "\"%REPO%\scripts\run_bot_keeper.cmd\"" ^
  /sc minute /mo 15 ^
  /f
if errorlevel 1 echo WARNING: FootballBotPersist creation failed

REM 4. Weekly retrain: Sundays 04:00 with auto-revert if regression
schtasks /create /tn "FootballBotRetrain" ^
  /tr "\"%REPO%\scripts\run_weekly_retrain.cmd\"" ^
  /sc weekly /d SUN /st 04:00 ^
  /f
if errorlevel 1 echo WARNING: FootballBotRetrain creation failed

echo.
echo Tasks installed. List with: schtasks /query /tn FootballBotDaily
echo Logs: %REPO%\logs\daily_*.log and %REPO%\logs\resolve_*.log
echo.
endlocal
