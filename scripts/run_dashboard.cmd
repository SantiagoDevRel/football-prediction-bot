@echo off
REM Launch the Streamlit dashboard. Open http://localhost:8501 in your browser.

setlocal
cd /d "%~dp0\.."
set PYTHONUTF8=1

.venv\Scripts\streamlit run scripts\dashboard.py

endlocal
