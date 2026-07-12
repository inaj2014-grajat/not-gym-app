@echo off
REM ── NotGym desktop — commit app code + intro assets to not-gym-app ──
cd /d "%~dp0"

git add activeai.py static NotGym.bat INSTALL.bat .gitignore BUILD_EXE.bat COMMIT_FIXES.bat
git commit -m "Cinematic intro assets + launcher update" -m "static/videos (5 clips, JJ text cropped), static/sfx (music/whoosh/impact), brand logos incl. dark variant, NotGym.bat launches Chrome with autoplay + dedicated profile, BUILD_EXE.bat."
git push origin main

echo.
echo Done. Check https://github.com/inaj2014-grajat/not-gym-app
pause
