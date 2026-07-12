@echo off
REM ── Build NotGym.exe with PyInstaller (uses NotGym.spec) ──
cd /d "%~dp0"

set PY=C:\Users\graja\AppData\Local\Programs\Python\Python311\python.exe
if not exist "%PY%" set PY=py -3.11

REM Intro videos must be inside static\ BEFORE building (bundled into the EXE)
if not exist "static\videos\squat.mp4" (
  echo WARNING: static\videos\ is missing the intro clips.
  echo Run GET_VIDEOS.bat in C:\Projects\NotGym first, or the intro
  echo will skip straight to the logo. Continuing in 5 seconds...
  timeout /t 5 >nul
)

%PY% -m pip show pyinstaller >nul 2>nul || %PY% -m pip install pyinstaller

echo Building... this takes a few minutes.
%PY% -m PyInstaller NotGym.spec --noconfirm

if exist "dist\NotGym.exe" (
  echo.
  echo SUCCESS: dist\NotGym.exe is ready.
) else (
  echo.
  echo BUILD FAILED — scroll up for the error.
)
pause
