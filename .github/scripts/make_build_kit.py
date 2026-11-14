"""Create ``MultiImageOptimizer-build-kit.zip`` from the tracked sources.

Used by the release job (to publish the kit) and by the Windows CI job (to
extract and actually run it), so both always test the same archive layout::

    python .github/scripts/make_build_kit.py

The kit mirrors the repository: BUILD.bat, README.md, LICENSE and
THIRD_PARTY_NOTICES.md at the root, the application sources and build recipe in
src/, and the README screenshot in docs/.  BUILD.bat resolves the script and
licence paths from its own location.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "MultiImageOptimizer-build-kit.zip"

ROOT_FILES = ("BUILD.bat", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md")
DOC_FILES = ("docs/gui.png",)
SOURCE_FILES = (
    "main.py",
    "gui.py",
    "pipeline.py",
    "aligner.py",
    "stacker.py",
    "rawio.py",
    "build.spec",
    "build.ps1",
    "requirements.txt",
)


def main() -> int:
    root_files = [ROOT / name for name in ROOT_FILES]
    source_files = [ROOT / "src" / name for name in SOURCE_FILES]
    doc_files = [ROOT / name for name in DOC_FILES]
    missing = [
        path for path in root_files + source_files + doc_files if not path.is_file()
    ]
    if missing:
        for path in missing:
            print(f"missing: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in root_files:
            archive.write(path, path.name)
        for path in source_files:
            archive.write(path, f"src/{path.name}")
        for path in doc_files:
            archive.write(path, path.relative_to(ROOT).as_posix())

    print(f"wrote {ARCHIVE.name} ({ARCHIVE.stat().st_size} bytes, "
          f"{len(root_files) + len(source_files) + len(doc_files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
