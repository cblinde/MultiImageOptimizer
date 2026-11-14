#requires -Version 5.1
<#
    Builds MultiImageOptimizer.exe on Windows without any installed tooling.

    BUILD.bat calls this script, either from src\ in the repository or from the
    root of the flat release kit. It downloads the official Python 3.12
    embeddable runtime into <project>\.python, enables site-packages, installs
    the requirements plus PyInstaller into it, runs build.spec, and copies the
    executable and licence files into <project>\out.

    Nothing outside the project folder is modified: there is no system Python
    install, no registry change and no admin prompt. A 64-bit Windows 10/11
    machine and an internet connection on the first run are the only
    requirements.
#>
[CmdletBinding()]
param(
    # Folder that receives .python, dist, out and the zip (the kit root).
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$SourceDir = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $SourceDir
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

$PythonDir = Join-Path $ProjectRoot '.python'
$PythonExe = Join-Path $PythonDir 'python.exe'
$Downloads = Join-Path $ProjectRoot '.downloads'
$LogFile = Join-Path $ProjectRoot 'build-log.txt'
$OutDir = Join-Path $ProjectRoot 'out'
$DistDir = Join-Path $ProjectRoot 'dist'
$WorkDir = Join-Path $ProjectRoot 'build'
$PythonVersion = '3.12.7'
$PythonZipUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$GetPipUrl = 'https://bootstrap.pypa.io/get-pip.py'
$RequiredSources = @(
    'main.py', 'gui.py', 'pipeline.py', 'aligner.py', 'stacker.py', 'rawio.py',
    'build.spec', 'requirements.txt'
)
$RequiredLicences = @('LICENSE', 'THIRD_PARTY_NOTICES.md')

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Get-KitFile {
    param([string]$Uri, [string]$Destination)
    if (Test-Path -LiteralPath $Destination) { return }
    Write-Host "  downloading $Uri"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination
    }
    catch {
        throw "Download failed: $Uri`n$($_.Exception.Message)"
    }
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Activity
    )
    Write-Host "  $Activity" -ForegroundColor DarkGray
    & $FilePath @Arguments 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Activity failed with exit code $LASTEXITCODE. See build-log.txt."
    }
}

try {
    "== Multi-Image Optimizer build kit ==" | Set-Content -LiteralPath $LogFile -Encoding UTF8
    Add-Content -LiteralPath $LogFile -Value ("started " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))

    if (-not [Environment]::Is64BitOperatingSystem) {
        throw '64-bit Windows is required.'
    }
    if ($ProjectRoot.Length -gt 110) {
        throw "Extract the kit to a shorter path; the current one is $ProjectRoot."
    }
    foreach ($name in $RequiredSources) {
        if (-not (Test-Path -LiteralPath (Join-Path $SourceDir $name))) {
            throw "Missing $name next to build.ps1. Extract the complete zip archive before running BUILD.bat."
        }
    }
    foreach ($name in $RequiredLicences) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $name))) {
            throw "Missing $name in $ProjectRoot. Extract the complete zip archive before running BUILD.bat."
        }
    }

    Write-Step "Portable Python $PythonVersion"
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        $zip = Join-Path $Downloads 'python-embed-amd64.zip'
        Get-KitFile -Uri $PythonZipUrl -Destination $zip
        Write-Host "  extracting into .python"
        Expand-Archive -LiteralPath $zip -DestinationPath $PythonDir -Force

        $pth = Get-ChildItem -LiteralPath $PythonDir -Filter 'python*._pth' |
            Select-Object -First 1
        if (-not $pth) {
            throw 'The Python archive did not contain a ._pth file. Delete .python and retry.'
        }
        # The embeddable runtime ships with site-packages disabled; pip needs it.
        (Get-Content -LiteralPath $pth.FullName) -replace '^\s*#\s*import site\s*$', 'import site' |
            Set-Content -LiteralPath $pth.FullName -Encoding ASCII
        if (-not (Select-String -LiteralPath $pth.FullName -Pattern '^import site' -Quiet)) {
            throw "Could not enable site-packages in $($pth.FullName)."
        }
        New-Item -ItemType Directory -Force -Path (Join-Path $PythonDir 'Lib\site-packages') |
            Out-Null
    }
    $version = & $PythonExe -c "import sys; print(sys.version.split()[0])"
    if ($LASTEXITCODE -ne 0) {
        throw "The portable Python at $PythonExe did not start. Delete .python and retry."
    }
    Write-Host "  using Python $version"

    Write-Step 'pip'
    & $PythonExe -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        $getPip = Join-Path $Downloads 'get-pip.py'
        Get-KitFile -Uri $GetPipUrl -Destination $getPip
        Invoke-Native -FilePath $PythonExe -Arguments @($getPip) -Activity 'bootstrapping pip'
    }

    Write-Step 'Build dependencies (the first run downloads several hundred MB)'
    Invoke-Native -FilePath $PythonExe `
        -Arguments @(
            '-m', 'pip', 'install', '--disable-pip-version-check',
            '--no-warn-script-location', '-r', (Join-Path $SourceDir 'requirements.txt'),
            'pyinstaller>=6.10,<7'
        ) `
        -Activity 'pip install'

    Write-Step 'Checking the installed dependencies'
    Invoke-Native -FilePath $PythonExe `
        -Arguments @('-c', 'import numpy, cv2, tifffile, rawpy, PySide6.QtWidgets') `
        -Activity 'import check'

    Write-Step 'PyInstaller'
    Invoke-Native -FilePath $PythonExe `
        -Arguments @(
            '-m', 'PyInstaller', '--noconfirm', '--clean',
            '--distpath', $DistDir, '--workpath', $WorkDir,
            (Join-Path $SourceDir 'build.spec')
        ) `
        -Activity 'pyinstaller'

    $exe = Join-Path $DistDir 'MultiImageOptimizer.exe'
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "PyInstaller finished without producing $exe."
    }

    Write-Step 'Result'
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Copy-Item -LiteralPath $exe -Destination $OutDir -Force
    foreach ($name in $RequiredLicences) {
        Copy-Item -LiteralPath (Join-Path $ProjectRoot $name) -Destination $OutDir -Force
    }

    $bundle = Join-Path $ProjectRoot 'MultiImageOptimizer-windows-x64.zip'
    if (Test-Path -LiteralPath $bundle) { Remove-Item -LiteralPath $bundle -Force }
    Compress-Archive -Path (Join-Path $OutDir '*') -DestinationPath $bundle

    $built = Join-Path $OutDir 'MultiImageOptimizer.exe'
    $size = (Get-Item -LiteralPath $built).Length / 1MB
    Add-Content -LiteralPath $LogFile -Value ("finished " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host ("  {0}  ({1:N1} MB)" -f $built, $size) -ForegroundColor Green
    Write-Host ("  {0}" -f $bundle) -ForegroundColor Green
    Write-Host "  Licence files sit next to the executable."
    Write-Host ""
    Write-Host "Build complete." -ForegroundColor Green
    exit 0
}
catch {
    Write-Host ""
    Write-Host "BUILD FAILED" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Add-Content -LiteralPath $LogFile -Value ("ERROR: " + $_.Exception.Message)
    exit 1
}
