"""Fail when a release tag does not match ``APP_VERSION`` in ``main.py``.

The release workflow runs this before uploading anything, so a tag such as
``v1.0.1`` cannot publish an executable that still reports ``1.0.0``::

    python .github/scripts/check_release_version.py v1.0.1

Exit codes: 0 when the tag matches, 1 on a mismatch, 2 when the tag is missing.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def read_app_version(path: Path = Path("main.py")) -> str:
    """Return APP_VERSION from the module body of *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "APP_VERSION":
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise SystemExit(f"APP_VERSION in {path} is not a string")
                return value
    raise SystemExit(f"APP_VERSION is not defined in {path}")


def main(argv: list[str]) -> int:
    tag = argv[1].strip() if len(argv) > 1 else ""
    if not tag:
        print("usage: check_release_version.py <tag>", file=sys.stderr)
        return 2

    version = read_app_version()
    version_from_tag = tag[1:] if tag[:1] in ("v", "V") else tag
    if version_from_tag != version:
        print(
            f"::error::Tag '{tag}' does not match APP_VERSION '{version}' in "
            "main.py. Bump APP_VERSION to the released version, or retag the "
            "commit that declares it."
        )
        return 1

    print(f"Tag '{tag}' matches APP_VERSION '{version}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
