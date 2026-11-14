"""Alignment-aware stacking and drizzle super-resolution.

Two combination engines are provided:

* :func:`stack_noise_reduction` -- warp every frame into the reference frame
  and combine pixels with a weighted mean or median.  The mean is optimal for
  zero-mean Gaussian sensor noise, the median rejects transient outliers
  (hot pixels, moving objects).
* :func:`stack_super_resolution` -- accumulate the sub-pixel dithered frames on
  a grid ``scale`` times finer using a drizzle style, area-weighted splat.  The
  result is blended with a bicubic upscale of the native resolution mean stack
  wherever the drizzle coverage is too sparse (holes).

All accumulation happens in float32 linear space, so 16-bit input precision is
preserved end to end.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np

from rawio import load_frame

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]
#: Receives ``(frame_index, aligned_native_resolution_image)`` for each frame.
AlignedCallback = Callable[[int, np.ndarray], None]

#: Target working set for a single median row band (bytes).
_MEDIAN_BAND_BUDGET = 96 * 1024 * 1024
#: Target number of drizzle scatter elements per band.
_DRIZZLE_BAND_ELEMENTS = 4_000_000


class StackError(RuntimeError):
    """Raised when the stack cannot be computed."""


@dataclass
class StackStats:
    """Diagnostics describing a completed stack."""

    frames_used: int
    mode: str
    method: str
    upscale: int
    output_shape: Tuple[int, int, int]
    coverage: float = 1.0
    pixfrac: float = 0.0
    median_band_rows: int = 0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def warp_into(
    rgb: np.ndarray,
    matrix: np.ndarray,
    out_shape: Tuple[int, int],
    border: int = cv2.BORDER_REFLECT101,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp *rgb* with bicubic interpolation and return ``(image, weight)``.

    Args:
        rgb: Input frame, float32 ``(H, W, 3)``.
        matrix: 3x3 forward transform mapping input coordinates to output
            coordinates.
        out_shape: ``(height, width)`` of the destination.
        border: OpenCV border mode for the image data.  Areas outside the
            source frame get zero weight and are therefore excluded from the
            combination.

    Returns:
        Warped float32 image and a per-pixel ``(H, W, 1)`` float32 weight mask.
    """
    height, width = out_shape
    affine = np.asarray(matrix, dtype=np.float32)[:2]
    warped = cv2.warpAffine(
        rgb,
        affine,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=border,
    )
    source_mask = np.ones(rgb.shape[:2], dtype=np.float32)
    weight = cv2.warpAffine(
        source_mask,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    np.clip(weight, 0.0, 1.0, out=weight)
    return warped, weight[..., None]


def _check_cancelled(cancelled: Optional[CancelCallback]) -> None:
    if cancelled is not None and cancelled():
        raise StackError("Processing cancelled")


def _report(
    progress: Optional[ProgressCallback],
    fraction: float,
    message: str,
) -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, fraction)), message)


def _open_scratch(
    total: int,
    out_shape: Tuple[int, int],
    temp_dir: Optional[str],
) -> Tuple[np.memmap, str]:
    """Create the on-disk frame stack used by the median path."""
    handle = tempfile.NamedTemporaryFile(
        prefix="mio_stack_", suffix=".dat", dir=temp_dir, delete=False
    )
    handle.close()
    stack = np.memmap(
        handle.name,
        dtype=np.uint16,
        mode="w+",
        shape=(total, out_shape[0], out_shape[1], 3),
    )
    return stack, handle.name


# ---------------------------------------------------------------------------
# Noise reduction (mean / median)
# ---------------------------------------------------------------------------
def stack_noise_reduction(
    paths: Sequence[str],
    matrices: Sequence[np.ndarray],
    method: str = "mean",
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
    temp_dir: Optional[str] = None,
    aligned: Optional[AlignedCallback] = None,
) -> Tuple[np.ndarray, StackStats]:
    """Combine aligned frames with a mean or median.

    Args:
        paths: Frame paths, already validated.
        matrices: Forward transforms, one per frame (full resolution).
        method: ``"mean"`` or ``"median"``.
        progress: Optional ``(fraction, message)`` callback.
        cancelled: Optional predicate polled between frames.
        temp_dir: Directory for the median scratch stack (defaults to the
            system temp directory).
        aligned: Optional callback receiving every warped frame, so callers can
            export the aligned stack without warping the data a second time.

    Returns:
        ``(image, stats)`` with a float32 linear RGB image in ``[0, 1]``.
    """
    method = method.lower()
    if method not in {"mean", "median"}:
        raise StackError(f"Unknown stack method '{method}'")
    if len(paths) != len(matrices):
        raise StackError("The number of frames and transforms differ")
    if not paths:
        raise StackError("No frames to stack")

    if method == "mean":
        return _mean_stack(paths, matrices, progress, cancelled, aligned)
    return _median_stack(paths, matrices, progress, cancelled, temp_dir, aligned)


def _mean_stack(
    paths: Sequence[str],
    matrices: Sequence[np.ndarray],
    progress: Optional[ProgressCallback],
    cancelled: Optional[CancelCallback],
    aligned: Optional[AlignedCallback],
) -> Tuple[np.ndarray, StackStats]:
    """Weighted mean accumulation, streaming one frame at a time."""
    accumulator: Optional[np.ndarray] = None
    weight_sum: Optional[np.ndarray] = None
    out_shape: Optional[Tuple[int, int]] = None
    total = len(paths)

    for index, (path, matrix) in enumerate(zip(paths, matrices)):
        _check_cancelled(cancelled)
        _report(progress, index / total, f"Aligning and stacking frame {index + 1}/{total}")
        frame = load_frame(path)
        if out_shape is None:
            out_shape = frame.shape[:2]
            accumulator = np.zeros((*out_shape, 3), dtype=np.float32)
            weight_sum = np.zeros((out_shape[0], out_shape[1], 1), dtype=np.float32)
        elif frame.shape[:2] != out_shape:
            raise StackError(
                f"Frame {os.path.basename(path)} has size {frame.shape[:2]}, "
                f"expected {out_shape}"
            )

        warped, weight = warp_into(frame, matrix, out_shape)
        if aligned is not None:
            aligned(index, warped)
        accumulator += warped * weight
        weight_sum += weight
        del frame, warped

    assert accumulator is not None and weight_sum is not None and out_shape is not None
    _check_cancelled(cancelled)
    result = accumulator / np.maximum(weight_sum, 1e-6)
    coverage = float(np.mean(weight_sum > 0.5))
    np.clip(result, 0.0, 1.0, out=result)
    stats = StackStats(
        frames_used=len(paths),
        mode="noise_reduction",
        method="mean",
        upscale=1,
        output_shape=result.shape,
        coverage=coverage,
    )
    return result, stats


def _median_stack(
    paths: Sequence[str],
    matrices: Sequence[np.ndarray],
    progress: Optional[ProgressCallback],
    cancelled: Optional[CancelCallback],
    temp_dir: Optional[str],
    aligned: Optional[AlignedCallback],
) -> Tuple[np.ndarray, StackStats]:
    """Median across the stack, staged on disk to bound RAM usage."""
    total = len(paths)
    scratch_path: Optional[str] = None
    stack: Optional[np.memmap] = None
    weight_sum: Optional[np.ndarray] = None
    out_shape: Optional[Tuple[int, int]] = None

    try:
        for index, (path, matrix) in enumerate(zip(paths, matrices)):
            _check_cancelled(cancelled)
            _report(progress, 0.8 * index / total, f"Aligning frame {index + 1}/{total}")
            frame = load_frame(path)
            if out_shape is None:
                out_shape = frame.shape[:2]
                weight_sum = np.zeros((out_shape[0], out_shape[1], 1), dtype=np.float32)
                stack, scratch_path = _open_scratch(total, out_shape, temp_dir)
            elif frame.shape[:2] != out_shape:
                raise StackError(
                    f"Frame {os.path.basename(path)} has size {frame.shape[:2]}, "
                    f"expected {out_shape}"
                )

            warped, weight = warp_into(frame, matrix, out_shape)
            if aligned is not None:
                aligned(index, warped)
            stack[index] = np.rint(np.clip(warped, 0.0, 1.0) * 65535.0).astype(np.uint16)
            weight_sum += weight
            del frame, warped

        assert stack is not None and weight_sum is not None and out_shape is not None
        _check_cancelled(cancelled)
        result, coverage = _median_finish(
            stack, weight_sum, out_shape, total, progress, cancelled
        )
        stats = StackStats(
            frames_used=len(paths),
            mode="noise_reduction",
            method="median",
            upscale=1,
            output_shape=result.shape,
            coverage=coverage,
            median_band_rows=_median_band_rows(total, out_shape[1]),
        )
        return result, stats
    finally:
        if stack is not None:
            del stack
        if scratch_path and os.path.exists(scratch_path):
            try:
                os.unlink(scratch_path)
            except OSError as exc:  # pragma: no cover - Windows file locking
                LOGGER.debug("Could not remove scratch file %s: %s", scratch_path, exc)


def _median_finish(
    stack: np.memmap,
    weight_sum: np.ndarray,
    out_shape: Tuple[int, int],
    total: int,
    progress: Optional[ProgressCallback],
    cancelled: Optional[CancelCallback],
    low: float = 0.82,
    high: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Median the gathered band stack and return ``(image, coverage)``."""
    stack.flush()
    band_rows = _median_band_rows(total, out_shape[1])
    _report(progress, low, f"Computing median over {total} frames")
    result = np.empty((*out_shape, 3), dtype=np.float32)
    bands = math.ceil(out_shape[0] / band_rows)
    for band_index, y0 in enumerate(range(0, out_shape[0], band_rows)):
        _check_cancelled(cancelled)
        y1 = min(out_shape[0], y0 + band_rows)
        chunk = np.asarray(stack[:, y0:y1], dtype=np.float32)
        result[y0:y1] = np.median(chunk, axis=0)
        _report(
            progress,
            low + (high - low) * (band_index + 1) / bands,
            f"Median rows {y1}/{out_shape[0]}",
        )
        del chunk

    result *= np.float32(1.0 / 65535.0)
    return result, float(np.mean(weight_sum > 0.5))


def _median_band_rows(frame_count: int, width: int, channels: int = 3) -> int:
    """Choose a row band that keeps the float32 median working set bounded."""
    bytes_per_row = max(1, frame_count * width * channels * 4)
    return max(1, min(1024, _MEDIAN_BAND_BUDGET // bytes_per_row))


# ---------------------------------------------------------------------------
# Super-resolution (drizzle)
# ---------------------------------------------------------------------------
def stack_super_resolution(
    paths: Sequence[str],
    matrices: Sequence[np.ndarray],
    upscale: int = 2,
    pixfrac: float = 0.75,
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
    aligned: Optional[AlignedCallback] = None,
) -> Tuple[np.ndarray, StackStats]:
    """Reconstruct a higher resolution image from sub-pixel dithered frames.

    Each input pixel is projected into the high resolution grid and splatted
    with the exact area of overlap between its (pixfrac sized) drop and the
    target pixels.  Splatting is flux conserving, so dividing the accumulated
    flux by the accumulated weight returns surface brightness in the original
    ``[0, 1]`` range.

    Args:
        paths: Frame paths.
        matrices: Forward transforms, one per frame, full resolution.
        upscale: Integer linear resolution multiplier (2..4 in practice).
        pixfrac: Drop size as a fraction of an *input* pixel.  Smaller values
            give a sharper but sparser reconstruction.
        progress: Optional progress callback.
        cancelled: Optional cancel predicate.

    Returns:
        ``(image, stats)`` with a float32 linear RGB image at
        ``(H * upscale, W * upscale)``.
    """
    if upscale < 2:
        raise StackError("Super-resolution requires an upscale factor of at least 2")
    if not 0.0 < pixfrac <= 1.0:
        raise StackError("pixfrac must be in the range (0, 1]")
    if len(paths) != len(matrices):
        raise StackError("The number of frames and transforms differ")
    if len(paths) < 2:
        raise StackError("Super-resolution needs at least two frames")

    total = len(paths)
    native_shape: Optional[Tuple[int, int]] = None
    base_accumulator: Optional[np.ndarray] = None
    base_weight: Optional[np.ndarray] = None
    flux: Optional[np.ndarray] = None
    coverage: Optional[np.ndarray] = None

    for index, (path, matrix) in enumerate(zip(paths, matrices)):
        _check_cancelled(cancelled)
        _report(progress, 0.9 * index / total, f"Drizzling frame {index + 1}/{total}")
        frame = load_frame(path)

        if native_shape is None:
            native_shape = frame.shape[:2]
            high_shape = (native_shape[0] * upscale, native_shape[1] * upscale)
            if high_shape[0] * high_shape[1] > 250_000_000:
                raise StackError(
                    f"Requested output {high_shape[1]}x{high_shape[0]} is too large; "
                    "use a smaller upscale factor"
                )
            base_accumulator = np.zeros((*native_shape, 3), dtype=np.float32)
            base_weight = np.zeros((native_shape[0], native_shape[1], 1), dtype=np.float32)
            flux = np.zeros((*high_shape, 3), dtype=np.float32)
            coverage = np.zeros(high_shape, dtype=np.float32)
        elif frame.shape[:2] != native_shape:
            raise StackError(
                f"Frame {os.path.basename(path)} has size {frame.shape[:2]}, "
                f"expected {native_shape}"
            )

        warped, weight = warp_into(frame, matrix, native_shape)
        if aligned is not None:
            aligned(index, warped)
        base_accumulator += warped * weight
        base_weight += weight
        del warped, weight

        _drizzle_frame(frame, matrix, flux, coverage, upscale, pixfrac)
        del frame

    assert (
        base_accumulator is not None
        and base_weight is not None
        and flux is not None
        and coverage is not None
        and native_shape is not None
    )

    _check_cancelled(cancelled)
    _report(progress, 0.92, "Reconstructing high resolution image")
    result, drizzle_coverage = _drizzle_finish(
        base_accumulator, base_weight, flux, coverage, pixfrac
    )

    stats = StackStats(
        frames_used=len(paths),
        mode="super_resolution",
        method="drizzle",
        upscale=upscale,
        output_shape=result.shape,
        coverage=drizzle_coverage,
        pixfrac=pixfrac,
    )
    return result, stats


def _drizzle_finish(
    base_accumulator: np.ndarray,
    base_weight: np.ndarray,
    flux: np.ndarray,
    coverage: np.ndarray,
    pixfrac: float,
) -> Tuple[np.ndarray, float]:
    """Blend the drizzle result with the Lanczos-upscaled mean and report coverage."""
    base = base_accumulator / np.maximum(base_weight, 1e-6)
    high_shape = flux.shape[:2]
    fallback = cv2.resize(
        np.clip(base, 0.0, 1.0), (high_shape[1], high_shape[0]), interpolation=cv2.INTER_LANCZOS4
    )

    safe_weight = np.maximum(coverage, 1e-6)
    drizzle = flux / safe_weight[..., None]

    # Nominal coverage of an interior high resolution pixel.  Each input pixel
    # deposits a drop of area (pixfrac*upscale)^2, and input pixels are packed
    # at one per upscale^2 output pixels, so the mean interior coverage is
    # simply (pixfrac*upscale)^2 / upscale^2 == pixfrac^2.
    nominal = float(pixfrac) ** 2
    alpha = np.clip(coverage / max(nominal * 0.5, 1e-6), 0.0, 1.0)
    alpha = (alpha * alpha * (3.0 - 2.0 * alpha))[..., None]  # smoothstep feather

    result = drizzle * alpha + fallback * (1.0 - alpha)
    np.clip(result, 0.0, 1.0, out=result)
    return result, float(np.mean(coverage >= nominal * 0.5))


# ---------------------------------------------------------------------------
# Combined run: stack and drizzle from one decode
# ---------------------------------------------------------------------------
def stack_combined(
    paths: Sequence[str],
    matrices: Sequence[np.ndarray],
    method: str = "mean",
    upscale: int = 2,
    pixfrac: float = 0.75,
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
    aligned: Optional[AlignedCallback] = None,
    temp_dir: Optional[str] = None,
) -> Tuple[np.ndarray, StackStats, np.ndarray, StackStats]:
    """Produce the stack *and* the drizzle image from one decode per frame.

    Both engines consume the frames sequentially, so running them in a single
    pass reads and warps every frame once instead of once per pipeline.  The
    images and statistics match calling :func:`stack_noise_reduction` and
    :func:`stack_super_resolution` one after the other.

    Args:
        paths: Frame paths, already validated.
        matrices: Forward transforms, one per frame (full resolution).
        method: ``"mean"`` or ``"median"`` for the stacked image.
        upscale: Drizzle resolution multiplier (at least 2).
        pixfrac: Drop size as a fraction of an input pixel.
        progress: Optional ``(fraction, message)`` callback.
        cancelled: Optional cancel predicate polled between frames.
        aligned: Optional callback receiving every warped frame, called once per
            frame for both results.
        temp_dir: Directory for the median scratch stack.

    Returns:
        ``(stack_image, stack_stats, super_image, super_stats)``.
    """
    method = method.lower()
    if method not in {"mean", "median"}:
        raise StackError(f"Unknown stack method '{method}'")
    if upscale < 2:
        raise StackError("Super-resolution requires an upscale factor of at least 2")
    if not 0.0 < pixfrac <= 1.0:
        raise StackError("pixfrac must be in the range (0, 1]")
    if len(paths) != len(matrices):
        raise StackError("The number of frames and transforms differ")
    if len(paths) < 2:
        raise StackError("Super-resolution needs at least two frames")

    total = len(paths)
    native_shape: Optional[Tuple[int, int]] = None
    stack_accumulator: Optional[np.ndarray] = None
    weight_sum: Optional[np.ndarray] = None
    base_accumulator: Optional[np.ndarray] = None
    base_weight: Optional[np.ndarray] = None
    band_stack: Optional[np.memmap] = None
    scratch_path: Optional[str] = None
    flux: Optional[np.ndarray] = None
    coverage: Optional[np.ndarray] = None

    try:
        for index, (path, matrix) in enumerate(zip(paths, matrices)):
            _check_cancelled(cancelled)
            _report(progress, 0.90 * index / total, f"Aligning frame {index + 1}/{total}")
            frame = load_frame(path)

            if native_shape is None:
                native_shape = frame.shape[:2]
                high_shape = (native_shape[0] * upscale, native_shape[1] * upscale)
                if high_shape[0] * high_shape[1] > 250_000_000:
                    raise StackError(
                        f"Requested output {high_shape[1]}x{high_shape[0]} is too large; "
                        "use a smaller upscale factor"
                    )
                weight_sum = np.zeros((*native_shape, 1), dtype=np.float32)
                if method == "mean":
                    # The drizzle fallback *is* the mean stack, so a single pair
                    # of accumulators feeds both results.
                    stack_accumulator = np.zeros((*native_shape, 3), dtype=np.float32)
                    base_accumulator, base_weight = stack_accumulator, weight_sum
                else:
                    base_accumulator = np.zeros((*native_shape, 3), dtype=np.float32)
                    base_weight = np.zeros((*native_shape, 1), dtype=np.float32)
                    band_stack, scratch_path = _open_scratch(total, native_shape, temp_dir)
                flux = np.zeros((*high_shape, 3), dtype=np.float32)
                coverage = np.zeros(high_shape, dtype=np.float32)
            elif frame.shape[:2] != native_shape:
                raise StackError(
                    f"Frame {os.path.basename(path)} has size {frame.shape[:2]}, "
                    f"expected {native_shape}"
                )

            warped, weight = warp_into(frame, matrix, native_shape)
            if aligned is not None:
                aligned(index, warped)
            weight_sum += weight
            if method == "mean":
                stack_accumulator += warped * weight
            else:
                assert band_stack is not None
                band_stack[index] = np.rint(np.clip(warped, 0.0, 1.0) * 65535.0).astype(
                    np.uint16
                )
                base_accumulator += warped * weight
                base_weight += weight
            del warped, weight

            _drizzle_frame(frame, matrix, flux, coverage, upscale, pixfrac)
            del frame

        assert native_shape is not None
        assert weight_sum is not None and base_accumulator is not None and base_weight is not None
        assert flux is not None and coverage is not None
        _check_cancelled(cancelled)

        if method == "mean":
            assert stack_accumulator is not None
            stack_image = stack_accumulator / np.maximum(weight_sum, 1e-6)
            np.clip(stack_image, 0.0, 1.0, out=stack_image)
            stack_coverage = float(np.mean(weight_sum > 0.5))
            band_rows = 0
        else:
            assert band_stack is not None
            stack_image, stack_coverage = _median_finish(
                band_stack, weight_sum, native_shape, total, progress, cancelled, 0.90, 0.97
            )
            band_rows = _median_band_rows(total, native_shape[1])

        stack_stats = StackStats(
            frames_used=total,
            mode="noise_reduction",
            method=method,
            upscale=1,
            output_shape=stack_image.shape,
            coverage=stack_coverage,
            median_band_rows=band_rows,
        )

        _check_cancelled(cancelled)
        _report(progress, 0.98, "Reconstructing high resolution image")
        super_image, drizzle_coverage = _drizzle_finish(
            base_accumulator, base_weight, flux, coverage, pixfrac
        )
        super_stats = StackStats(
            frames_used=total,
            mode="super_resolution",
            method="drizzle",
            upscale=upscale,
            output_shape=super_image.shape,
            coverage=drizzle_coverage,
            pixfrac=pixfrac,
        )
        _report(progress, 1.0, "Combined")
        return stack_image, stack_stats, super_image, super_stats
    finally:
        if band_stack is not None:
            del band_stack
        if scratch_path and os.path.exists(scratch_path):
            try:
                os.unlink(scratch_path)
            except OSError as exc:  # pragma: no cover - Windows file locking
                LOGGER.debug("Could not remove scratch file %s: %s", scratch_path, exc)


def _drizzle_frame(
    rgb: np.ndarray,
    matrix: np.ndarray,
    flux: np.ndarray,
    weight: np.ndarray,
    scale: int,
    pixfrac: float,
) -> None:
    """Splat one frame into the high resolution accumulators.

    The image is processed in horizontal bands so the temporary index arrays
    stay small regardless of sensor size.
    """
    height, width = rgb.shape[:2]
    high_height, high_width = flux.shape[:2]
    drop = max(1e-3, pixfrac * scale)
    half = 0.5 * drop
    taps = int(math.ceil(drop)) + 1

    affine = np.asarray(matrix, dtype=np.float64)
    offset = 0.5 * scale - 0.5
    # Forward map: input pixel centre -> high resolution index coordinates.
    m00, m01, m02 = scale * affine[0, 0], scale * affine[0, 1], scale * affine[0, 2] + offset
    m10, m11, m12 = scale * affine[1, 0], scale * affine[1, 1], scale * affine[1, 2] + offset

    band_rows = max(1, min(height, _DRIZZLE_BAND_ELEMENTS // max(1, width * taps * taps)))
    x_coords = np.arange(width, dtype=np.float32)[None, :]
    y_all = np.arange(height, dtype=np.float32)[:, None]

    for y0 in range(0, height, band_rows):
        y1 = min(height, y0 + band_rows)
        y_coords = y_all[y0:y1]

        qx = m00 * x_coords + m01 * y_coords + m02
        qy = m10 * x_coords + m11 * y_coords + m12

        low_x = qx - half + 0.5
        high_x = qx + half + 0.5
        low_y = qy - half + 0.5
        high_y = qy + half + 0.5

        base_ix = np.floor(low_x).astype(np.int64)
        base_iy = np.floor(low_y).astype(np.int64)

        weights_x = np.empty((taps, y1 - y0, width), dtype=np.float32)
        weights_y = np.empty((taps, y1 - y0, width), dtype=np.float32)
        for tap in range(taps):
            weights_x[tap] = np.clip(
                np.minimum(high_x, base_ix + tap + 1) - np.maximum(low_x, base_ix + tap), 0.0, None
            )
            weights_y[tap] = np.clip(
                np.minimum(high_y, base_iy + tap + 1) - np.maximum(low_y, base_iy + tap), 0.0, None
            )

        # Local accumulation window keeps bincount arrays small.
        row_low = max(0, int(base_iy.min()))
        row_high = min(high_height, int(base_iy.max()) + taps)
        if row_high <= row_low:
            continue
        local_rows = row_high - row_low
        local_size = local_rows * high_width
        flux_local = np.zeros(local_size * 3, dtype=np.float64)
        weight_local = np.zeros(local_size, dtype=np.float64)

        band = np.asarray(rgb[y0:y1], dtype=np.float32)

        for tap_y in range(taps):
            iy = base_iy + tap_y
            weight_y = weights_y[tap_y]
            row_ok = (iy >= row_low) & (iy < row_high)
            if not row_ok.any():
                continue
            for tap_x in range(taps):
                ix = base_ix + tap_x
                weight_xy = weight_y * weights_x[tap_x]
                valid = row_ok & (ix >= 0) & (ix < high_width) & (weight_xy > 0.0)
                if not valid.any():
                    continue
                valid_rows, valid_cols = valid.nonzero()
                local_index = ((iy - row_low) * high_width + ix)[valid_rows, valid_cols]
                current_weight = weight_xy[valid_rows, valid_cols].astype(np.float64)
                weighted_values = (
                    band[valid_rows, valid_cols].astype(np.float64) * current_weight[:, None]
                )

                weight_local += np.bincount(
                    local_index, weights=current_weight, minlength=local_size
                )
                flux_local += np.bincount(
                    (local_index[:, None] * 3 + np.arange(3)[None, :]).ravel(),
                    weights=weighted_values.ravel(),
                    minlength=local_size * 3,
                )

        flux[row_low:row_high] += flux_local.reshape(local_rows, high_width, 3).astype(np.float32)
        weight[row_low:row_high] += weight_local.reshape(local_rows, high_width).astype(np.float32)
