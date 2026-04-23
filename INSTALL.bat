@echo off
title NotGym - First Time Setup
REM Always run from the folder where this bat file lives
cd /d "%~dp0"

echo.
echo  ==============================
echo   NotGym - First Time Setup
echo  ==============================
echo.
echo  This will install everything needed to run NotGym.
echo  Takes about 3-5 minutes on first run.
echo  Internet connection required for this step only.
echo.
pause

REM Check that bundled Python exists
if not exist "%~dp0python\python.exe" (
    echo.
    echo  ERROR: Bundled Python not found.
    echo  Make sure the "python" folder is in the same
    echo  folder as this INSTALL.bat file.
    echo.
    pause
    exit /b 1
)
echo  [1/3] Bundled Python found. OK.

REM Bootstrap pip into the embedded Python
echo  [2/3] Setting up pip...
"%~dp0python\python.exe" -m ensurepip --upgrade >nul 2>&1
if errorlevel 1 (
    echo  pip not found via ensurepip, trying get-pip...
    "%~dp0python\python.exe" -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
    "%~dp0python\python.exe" get-pip.py --quiet
    del get-pip.py >nul 2>&1
)
echo  [2/3] pip ready. OK.

REM Install required packages
echo  [3/3] Installing packages (this may take a few minutes)...
echo         flask, opencv-python, mediapipe, numpy
echo.
"%~dp0python\python.exe" -m pip install flask opencv-python mediapipe numpy --quiet
if errorlevel 1 (
    echo.
    echo  ERROR: Package installation failed.
    echo  Please check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo  ==============================
echo   Setup Complete!
echo  ==============================
echo.
echo  Double-click NotGym.bat to start your session.
echo.
pause
