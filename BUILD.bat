@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Multi-Image Optimizer - local build for Windows
echo ============================================================
echo  The first run downloads a portable Python and the build
echo  dependencies into this folder. Nothing is installed
echo  system-wide and no admin rights are needed.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1"
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo Build finished. The executable is here:
  echo   %~dp0out\MultiImageOptimizer.exe
) else (
  echo Build failed with exit code %CODE%.
  echo See "%~dp0build-log.txt" for the full output.
)
echo.
pause
exit /b %CODE%
