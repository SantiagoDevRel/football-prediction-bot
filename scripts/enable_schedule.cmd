@echo off
REM Self-elevating script. Re-enables the 4 FootballBot* scheduled tasks
REM that disable_schedule.cmd turned off.

NET FILE 1>NUL 2>NUL
if '%errorlevel%' == '0' (goto :doit)

powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
exit /b

:doit
echo Enabling FootballBot scheduled tasks...
echo.
schtasks /Change /TN "FootballBotDaily"   /ENABLE
schtasks /Change /TN "FootballBotPersist" /ENABLE
schtasks /Change /TN "FootballBotResolve" /ENABLE
schtasks /Change /TN "FootballBotRetrain" /ENABLE
echo.
echo --- Verification ---
schtasks /Query /TN "FootballBotDaily"   /FO LIST | findstr /C:"Status"
schtasks /Query /TN "FootballBotPersist" /FO LIST | findstr /C:"Status"
schtasks /Query /TN "FootballBotResolve" /FO LIST | findstr /C:"Status"
schtasks /Query /TN "FootballBotRetrain" /FO LIST | findstr /C:"Status"
echo.
pause
