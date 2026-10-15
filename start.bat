@echo off
rem WeatherWise - one-click setup and start for Windows.
rem First run: creates .venv, installs packages, trains the models. Later runs just start the server.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python was not found. Install Python 3.9 or newer from https://www.python.org/downloads/
  echo During install, tick "Add python.exe to PATH", then run start.bat again.
  pause
  exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 (
  echo This project needs Python 3.9 or newer. Found:
  %PY% --version
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
  echo [2/3] Installing packages. This needs internet access and may take a few minutes...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Package installation failed. See the Troubleshooting section of README.md.
    rmdir /s /q .venv
    pause
    exit /b 1
  )
  echo [3/3] Training the models on this computer...
  ".venv\Scripts\python.exe" -m ml.train_models
  if errorlevel 1 (
    echo Model training failed. See the messages above and the Troubleshooting section of README.md.
    rmdir /s /q .venv
    pause
    exit /b 1
  )
)

echo.
echo Starting WeatherWise at http://127.0.0.1:5000 - close this window to stop it.
".venv\Scripts\python.exe" app.py --open
pause
