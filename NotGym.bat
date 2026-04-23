@echo off
title NotGym - AI Fitness Trainer
REM Always run from the folder where this bat file lives
cd /d "%~dp0"

echo.
echo  ==============================
echo   NotGym - AI Fitness Trainer
echo  ==============================
echo.

REM Check bundled Python exists
if not exist "%~dp0python\python.exe" (
    echo  ERROR: Bundled Python not found.
    echo  Please make sure the "python" folder is present.
    echo  If this is your first time, run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

REM Check packages are installed
"%~dp0python\python.exe" -c "import flask, cv2, mediapipe, numpy" >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Required packages not found.
    echo  Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

REM Check activeai.py exists
if not exist "%~dp0activeai.py" (
    echo  ERROR: activeai.py not found.
    echo  Make sure all files are in the same folder as NotGym.bat.
    echo.
    pause
    exit /b 1
)

echo  Starting server...
echo.

REM Start Flask server using bundled Python
start "NotGym Server" "%~dp0python\python.exe" "%~dp0activeai.py"

REM Wait for server and mediapipe to load
echo  Loading AI models, please wait...
timeout /t 7 /nobreak >nul

REM Open browser
start http://localhost:5000

echo.
echo  NotGym is running at http://localhost:5000
echo.
echo  NOTE: Keep the "NotGym Server" window open.
echo        Close it to stop the server.
echo.
pause
