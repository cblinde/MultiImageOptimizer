"""Sub-pixel frame registration (feature matching + ECC refinement).

The registration pipeline estimates, for every frame, an affine matrix ``A``
that maps *input* pixel coordinates into *reference* pixel coordinates::

    reference(x, y) ~= frame(A . [x, y, 1])

Matrices are homogeneous ``3 x 3`` float64 arrays expressed in full resolution
pixel coordinates, which makes them directly usable by ``cv2.warpAffine`` and
by the drizzle accumulator.

Strategy per frame:

1. ORB keypoints + Hamming ratio test + RANSAC partial-affine estimate.  This
   survives translation, small rotation and small scale changes.
2. Enhanced Correlation Coefficient (ECC) refinement seeded with the RANSAC
   estimate, which pushes the result to sub-pixel accuracy.
3. Phase-correlation translation fallback when ORB finds too few matches
   (low texture / heavy noise / very large shake).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

#: Reject transforms whose scale drifts further than this from unity.
MAX_SCALE_DEVIATION = 0.15
#: Reject transforms with a larger rotation than this (degrees).
MAX_ROTATION_DEG = 25.0
#: Reject translations larger than this fraction of the image diagonal.
MAX_TRANSLATION_FRACTION = 0.35

ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]


class AlignmentError(RuntimeError):
    """Raised when the frames cannot be registered at all."""


@dataclass
class AlignmentResult:
    """Registration result for a single frame."""

    index: int
    matrix: np.ndarray
    method: str = "identity"
    matches: int = 0
    ecc_score: float = 0.0
    accepted: bool = True
    message: str = ""

    def describe(self) -> str:
        """Human readable one-line summary used in logs and the GUI."""
        scale, angle = decompose(self.matrix)
        dx, dy = self.matrix[0, 2], self.matrix[1, 2]
        return (
            f"frame {self.index}: {self.method} dx={dx:+.3f}px dy={dy:+.3f}px "
            f"rot={angle:+.3f}deg scale={scale:.5f} "
            f"matches={self.matches} ecc={self.ecc_score:.3f}"
        )


@dataclass
class GrayPreview:
    """Grayscale, downscaled preview used for registration."""

    gray: np.ndarray
    scale: float  # preview pixels per full resolution pixel
    shape: Tuple[int, int]  # full resolution (height, width)


def identity_matrix() -> np.ndarray:
    """Return a 3x3 identity transform."""
    return np.eye(3, dtype=np.float64)


def to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Convert a 2x3 affine matrix into a 3x3 homogeneous matrix."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape == (3, 3):
        return array
    if array.shape != (2, 3):
        raise ValueError(f"Expected a 2x3 or 3x3 matrix, got {array.shape}")
    return np.vstack([array, [0.0, 0.0, 1.0]])


def decompose(matrix: np.ndarray) -> Tuple[float, float]:
    """Return ``(scale, rotation_degrees)`` of an affine transform."""
    linear = np.asarray(matrix, dtype=np.float64)[:2, :2]
    scale = float(math.sqrt(abs(np.linalg.det(linear))))
    angle = float(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
    return scale, angle


def prepare_gray(
    rgb: np.ndarray,
    max_dim: int = 2000,
    pixel_size: float = 1.0,
) -> GrayPreview:
    """Convert a linear RGB frame to a stretched 8-bit grayscale preview.

    Args:
        rgb: Linear float32 ``(H, W, 3)`` frame.
        max_dim: Longest edge of the preview; larger inputs are area-averaged.
        pixel_size: Full resolution pixels covered by one pixel of *rgb*
            (2.0 when the array came from a half-size LibRaw demosaic).

    Returns:
        A :class:`GrayPreview` carrying the downscale factor so transforms can
        be mapped back to full resolution coordinates.
    """
    height, width = rgb.shape[:2]
    if height == 0 or width == 0:
        raise AlignmentError("Empty frame supplied for registration")
    full_shape = (
        int(round(height * pixel_size)),
        int(round(width * pixel_size)),
    )

    luminance = (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    )

    scale = 1.0 / pixel_size
    longest = max(height, width)
    if max_dim and longest > max_dim:
        resize = max_dim / float(longest)
        new_size = (max(16, int(round(width * resize))), max(16, int(round(height * resize))))
        luminance = cv2.resize(luminance, new_size, interpolation=cv2.INTER_AREA)
        scale = (new_size[0] / float(width)) / pixel_size

    gray = _stretch_to_u8(luminance)
    return GrayPreview(gray=gray, scale=scale, shape=full_shape)


def _stretch_to_u8(luminance: np.ndarray) -> np.ndarray:
    """Percentile-stretch linear luminance into a robust 8-bit image."""
    finite = luminance[np.isfinite(luminance)]
    if finite.size == 0:
        raise AlignmentError("Frame contains no finite pixel values")
    low, high = np.percentile(finite, (0.5, 99.5))
    if high - low < 1e-6:
        # Flat frames (lens cap) cannot be registered.
        return np.zeros(luminance.shape, dtype=np.uint8)
    stretched = (luminance - low) * (255.0 / (high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def sharpness(preview: GrayPreview) -> float:
    """Variance of the Laplacian, used to pick the sharpest reference frame."""
    gray = preview.gray.astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def align_frames(
    previews: Sequence[GrayPreview],
    reference_index: int = 0,
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
    refine: bool = True,
) -> List[AlignmentResult]:
    """Register every preview against ``previews[reference_index]``.

    Frames that cannot be registered are returned with ``accepted=False`` and a
    diagnostic message instead of aborting the whole run; the caller decides
    how many failures it can tolerate.

    Raises:
        AlignmentError: When the reference frame itself is unusable or no frame
            other than the reference could be registered.
    """
    if not previews:
        raise AlignmentError("No frames supplied for registration")
    if not 0 <= reference_index < len(previews):
        raise AlignmentError(
            f"Reference index {reference_index} is out of range for {len(previews)} frames"
        )

    reference = previews[reference_index]
    if float(reference.gray.max()) == 0.0:
        raise AlignmentError(
            f"Reference frame {reference_index} is featureless (all black); "
            "choose a different reference or check the capture"
        )

    results: List[AlignmentResult] = []
    total = len(previews)
    for index, preview in enumerate(previews):
        if cancelled is not None and cancelled():
            raise AlignmentError("Registration cancelled")
        if progress is not None:
            progress(index, f"Registering frame {index + 1}/{total}")

        if index == reference_index:
            results.append(
                AlignmentResult(
                    index=index, matrix=identity_matrix(), method="reference", accepted=True
                )
            )
            continue

        if preview.shape != reference.shape:
            results.append(
                AlignmentResult(
                    index=index,
                    matrix=identity_matrix(),
                    accepted=False,
                    message=(
                        f"Frame size {preview.shape} differs from reference "
                        f"{reference.shape}; frame skipped"
                    ),
                )
            )
            continue

        try:
            matrix, method, matches, score = _estimate_transform(
                reference.gray, preview.gray, refine=refine
            )
        except AlignmentError as exc:
            results.append(
                AlignmentResult(
                    index=index,
                    matrix=identity_matrix(),
                    accepted=False,
                    message=str(exc),
                )
            )
            continue

        # Map the preview-space transform back to full resolution coordinates.
        scale = preview.scale
        scale_matrix = np.diag([scale, scale, 1.0])
        full_matrix = np.linalg.inv(scale_matrix) @ matrix @ scale_matrix

        accepted, reason = _validate(full_matrix, reference.shape)
        results.append(
            AlignmentResult(
                index=index,
                matrix=full_matrix,
                method=method,
                matches=matches,
                ecc_score=score,
                accepted=accepted,
                message=reason,
            )
        )
        if not accepted:
            LOGGER.warning("Frame %d rejected: %s", index, reason)

    usable = sum(1 for r in results if r.accepted)
    if usable < 2:
        details = "; ".join(r.message for r in results if not r.accepted) or "unknown reason"
        raise AlignmentError(
            "Could not register at least two frames. Common causes: the frames are "
            f"unrelated, motion is too large or the scene is textureless ({details})"
        )
    return results


def _estimate_transform(
    reference_gray: np.ndarray,
    moving_gray: np.ndarray,
    refine: bool = True,
) -> Tuple[np.ndarray, str, int, float]:
    """Estimate the transform mapping *moving* coordinates into *reference*.

    Returns:
        ``(matrix_3x3, method_label, match_count, ecc_score)``.
    """
    ransac_matrix, match_count = _orb_ransac(reference_gray, moving_gray)

    seed = ransac_matrix
    if seed is None:
        seed = _phase_correlation(reference_gray, moving_gray)
        method = "phase-correlation"
    else:
        method = "orb+ransac"

    if seed is None:
        raise AlignmentError(
            "Not enough matching keypoints (ORB) and phase correlation failed; "
            "the frame may be blurry, textureless or too far displaced"
        )

    if not refine:
        return seed, method, match_count, 0.0

    try:
        refined, score = _refine_ecc(reference_gray, moving_gray, seed)
    except Exception as exc:  # ECC throws cv2.error on degenerate input
        LOGGER.debug("ECC refinement failed (%s); keeping %s result", exc, method)
        return seed, method, match_count, 0.0

    if score <= 0.0:
        return seed, method, match_count, 0.0
    return refined, f"{method}+ecc", match_count, score


def _orb_ransac(
    reference_gray: np.ndarray, moving_gray: np.ndarray
) -> Tuple[Optional[np.ndarray], int]:
    """Estimate a partial affine transform with ORB features and RANSAC."""
    orb = cv2.ORB_create(
        nfeatures=6000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        patchSize=31,
        fastThreshold=12,
    )
    # CLAHE keeps keypoint detection stable on very linear (low contrast) data.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_enhanced = clahe.apply(reference_gray)
    mov_enhanced = clahe.apply(moving_gray)

    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_enhanced, None)
    mov_keypoints, mov_descriptors = orb.detectAndCompute(mov_enhanced, None)
    if ref_descriptors is None or mov_descriptors is None:
        return None, 0
    if len(ref_keypoints) < 8 or len(mov_keypoints) < 8:
        return None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        knn = matcher.knnMatch(mov_descriptors, ref_descriptors, k=2)
    except cv2.error:  # pragma: no cover - defensive
        return None, 0

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < 0.8 * second.distance:
            good.append(best)

    if len(good) < 12:
        return None, len(good)

    source = np.float32([mov_keypoints[m.queryIdx].pt for m in good])  # moving coords
    target = np.float32([ref_keypoints[m.trainIdx].pt for m in good])  # reference coords

    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return None, len(good)

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < 8:
        return None, inlier_count
    return to_homogeneous(matrix), inlier_count


def _refine_ecc(
    reference_gray: np.ndarray,
    moving_gray: np.ndarray,
    seed: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Refine the transform with ECC maximisation at sub-pixel precision.

    ``findTransformECC`` returns ``W`` mapping *template* (reference)
    coordinates to *input* (moving) coordinates, therefore the forward
    input -> reference matrix is its inverse.
    """
    seed_inverse = np.linalg.inv(seed)
    warp = seed_inverse[:2].astype(np.float32)

    blurred_ref = cv2.GaussianBlur(reference_gray, (3, 3), 0)
    blurred_mov = cv2.GaussianBlur(moving_gray, (3, 3), 0)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
    score, refined_warp = cv2.findTransformECC(
        blurred_ref,
        blurred_mov,
        warp,
        cv2.MOTION_EUCLIDEAN,
        criteria,
        None,
        5,
    )
    if not np.all(np.isfinite(refined_warp)):
        raise AlignmentError("ECC produced a non-finite transform")

    matrix = np.linalg.inv(to_homogeneous(refined_warp))
    return matrix, float(score)


def _phase_correlation(
    reference_gray: np.ndarray, moving_gray: np.ndarray
) -> Optional[np.ndarray]:
    """Estimate a pure translation with phase correlation."""
    window = cv2.createHanningWindow(
        (reference_gray.shape[1], reference_gray.shape[0]), cv2.CV_32F
    )
    ref = reference_gray.astype(np.float32) / 255.0
    mov = moving_gray.astype(np.float32) / 255.0
    try:
        (dx, dy), response = cv2.phaseCorrelate(ref, mov, window)
    except cv2.error:
        return None
    if not math.isfinite(dx) or not math.isfinite(dy) or response < 0.05:
        return None
    # phaseCorrelate reports the shift of `mov` relative to `ref`, so the
    # forward matrix moving -> reference is the negated translation.
    return np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _validate(matrix: np.ndarray, shape: Tuple[int, int]) -> Tuple[bool, str]:
    """Sanity check a transform against physically plausible camera shake."""
    if not np.all(np.isfinite(matrix)):
        return False, "transform contains non-finite values"

    scale, angle = decompose(matrix)
    if abs(scale - 1.0) > MAX_SCALE_DEVIATION:
        return False, f"implausible scale {scale:.3f} (expected ~1.0)"
    if abs(angle) > MAX_ROTATION_DEG:
        return False, f"implausible rotation {angle:.2f} deg"

    height, width = shape
    diagonal = math.hypot(width, height)
    dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
    shift = math.hypot(dx, dy)
    if shift > MAX_TRANSLATION_FRACTION * diagonal:
        limit = f"{MAX_TRANSLATION_FRACTION:.0%} diagonal"
        return False, f"implausible shift {shift:.1f}px (limit {limit})"
    return True, ""
