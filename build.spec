# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the standalone Multi-Image Optimizer executable.

Build with::

    pyinstaller --noconfirm build.spec

The result is ``dist/MultiImageOptimizer.exe``: a single windowed binary that
bundles Python, Qt, NumPy, OpenCV, LibRaw (via rawpy) and the MSVC runtime, so
the target machine needs no Python installation.
"""

import glob
import os
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs

# LibRaw (raw_r.dll) and its OpenMP runtime ship inside the rawpy package.
binaries = collect_dynamic_libs("rawpy")

# raw_r.dll also links against the Microsoft C/C++ runtime.  When it is only
# present in the virtual environment (msvc-runtime) rather than in System32,
# bundle it explicitly so the frozen build works on a clean machine.
_runtime_dirs = [
    os.path.join(sys.prefix, "Scripts"),
    os.path.join(sys.prefix, "Lib", "site-packages"),
    sys.prefix,
    sys.base_prefix,
]
for _directory in dict.fromkeys(_runtime_dirs):
    for _pattern in ("msvcp140*.dll", "vcruntime140*.dll", "concrt140.dll", "vcomp140.dll"):
        for _dll in glob.glob(os.path.join(_directory, _pattern)):
            binaries.append((_dll, "."))

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=["rawpy", "tifffile"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "pytest",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtMultimedia",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MultiImageOptimizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI mode must not open a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
