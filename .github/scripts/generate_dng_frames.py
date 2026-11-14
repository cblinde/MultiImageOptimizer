"""Write a few tiny synthetic DNG frames for the release smoke test.

DNG input makes the packaged executable decode through rawpy/LibRaw, so a
missing ``raw_r.dll`` or MSVC runtime fails the build instead of shipping.

Usage: ``python .github/scripts/generate_dng_frames.py smoke_raw``
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import rawio  # noqa: E402  (needs the repository root on sys.path)


def main(directory: str) -> int:
    """Write three shifted DNG frames into *directory*."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    height, width = 96, 128

    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack(
        [
            0.40 + 0.25 * np.sin(xx / 7.0),
            0.40 + 0.25 * np.cos(yy / 9.0),
            0.40 + 0.20 * np.sin((xx + yy) / 11.0),
        ],
        axis=-1,
    )
    base = np.clip(base + rng.normal(0.0, 0.05, base.shape).astype(np.float32), 0.0, 1.0)

    for index, (dx, dy) in enumerate(((0.0, 0.0), (0.5, 0.3), (-0.4, 0.6))):
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        frame = cv2.warpAffine(
            base,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT101,
        )
        metadata = rawio.FrameMetadata(
            width=width,
            height=height,
            is_raw=True,
            camera_model="CI-Synthetic",
            color_matrix=np.eye(3),
            as_shot_neutral=np.array([0.5, 1.0, 0.7]),
        )
        written = rawio.save_image(out / f"frame_{index}", frame, "dng", metadata)
        if written.suffix.lower() != ".dng":
            print(f"could not write a DNG frame: {written}", file=sys.stderr)
            return 1

    print(f"generated {len(list(out.glob('*.dng')))} DNG frames in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "smoke_raw"))
