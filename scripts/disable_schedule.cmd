@echo off
REM Self-elevating script. Disables the 4 FootballBot* scheduled tasks.
REM Use the matching enable_schedule.cmd to bring them back later.

NET FILE 1>NUL 2>NUL
if '%errorlevel%' == '0' (goto :doit)

REM Not admin — re-launch elevated
powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
exit /b

:doit
echo Disabling FootballBot scheduled tasks...
echo.
schtasks /Change /TN "FootballBotDaily"   /DISABLE
schtasks /Change /TN "FootballBotPersist" /DISABLE
schtasks /Change /TN "FootballBotResolve" /DISABLE
schtasks /Change /TN "FootballBotRetrain" /DISABLE
echo.
echo --- Verification ---
schtasks /Query /TN "FootballBotDaily"   /FO LIST | findstr /C:"Status"
schtasks /Query /TN "FootballBotPersist" /FO LIST | findstr /C:"Status"
schtasks /Query /TN "FootballBotResolve" /FO LIST | findstr /C:"Status"
schtasks /Query /TN "FootballBotRetrain" /FO LIST | findstr /C:"Status"
echo.
pause
