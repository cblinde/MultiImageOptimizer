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

rem The repository and the release kit keep the script in src\; the fallback
rem keeps older flat kits (build.ps1 beside this file) working.
set "SCRIPT=%~dp0build.ps1"
if not exist "%SCRIPT%" set "SCRIPT=%~dp0src\build.ps1"
if not exist "%SCRIPT%" (
  echo Build script not found. Extract the complete archive first.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" -ProjectRoot "%~dp0."
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
rem --no-pause keeps unattended runs (CI, scripts) from blocking at the end.
if /i not "%~1"=="--no-pause" pause
exit /b %CODE%
