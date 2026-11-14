"""Linear RAW / raster frame input and 16-bit output helpers.

Every frame is loaded as a linear floating point RGB array in the range
``[0, 1]``.  RAW files are demosaiced by ``rawpy``/LibRaw with a linear
transfer curve (``gamma=(1, 1)``), no auto brightening and no white balance
applied, so that consecutive frames stay photometrically comparable and can be
averaged without colour dependent bias.

Outputs are written as 16 bit TIFF, as 16 bit PNG, or - when a usable colour
matrix is available - as a 3 channel ``LinearRaw`` DNG.  DNG serialisation is
verified by reopening the file with LibRaw and falls back to TIFF if the reader
rejects it.

Sources are not limited to RAW: JPEG, PNG, BMP and TIFF files are accepted too.
PNG/JPEG/BMP (and 8 bit TIFFs) are display referred, so their sRGB transfer
curve is removed on load to keep the whole pipeline linear.  RAW and 16 bit TIFF
samples are already linear and are used as-is.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile

#: Handles returned by :func:`os.add_dll_directory`; kept alive for the
#: lifetime of the process so Windows keeps searching those directories.
_DLL_DIRECTORY_HANDLES: List[Any] = []


def _register_windows_runtime() -> None:
    """Make the MSVC runtime DLLs findable by rawpy's native extension.

    ``raw_r.dll`` links against ``MSVCP140.dll`` and friends.  These are
    installed system wide by the Visual C++ redistributable, but a plain
    Python install (or a venv where ``msvc-runtime`` was pip-installed) may
    keep them somewhere the extension loader does not search.  Registering
    every candidate directory that holds the runtime avoids the classic
    "DLL load failed while importing _rawpy" error without requiring
    administrator rights.
    """
    if os.name != "nt":
        return

    candidates = [
        Path(sys.prefix) / "Scripts",
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sys.base_prefix) / "Scripts",
        Path(getattr(sys, "_MEIPASS", sys.prefix)),  # PyInstaller onefile root
    ]
    for directory in candidates:
        try:
            if not (directory / "msvcp140.dll").is_file():
                continue
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
        except (OSError, AttributeError):
            continue


_register_windows_runtime()

try:  # rawpy is optional so that raster-only workflows still import cleanly.
    import rawpy
except ImportError:  # pragma: no cover - exercised only on broken installs
    rawpy = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

RAW_EXTENSIONS = {
    ".dng",
    ".cr2",
    ".cr3",
    ".nef",
    ".nrw",
    ".arw",
    ".srf",
    ".sr2",
    ".raf",
    ".rw2",
    ".orf",
    ".pef",
    ".srw",
    ".raw",
    ".3fr",
    ".erf",
    ".mrw",
    ".x3f",
}
RASTER_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | RASTER_EXTENSIONS

#: Containers whose samples carry the sRGB transfer curve by convention.
_DISPLAY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

#: Output format name -> file suffix actually written.
_OUTPUT_SUFFIXES = {"dng": ".dng", "tiff": ".tif", "png": ".png"}

#: Infix that marks a file as written by this tool.  It keeps a run from
#: stacking its own previous result when the output lives in the input folder.
OUTPUT_MARKER = "_stack_"
OUTPUT_NAME_RE = re.compile(
    rf"{re.escape(OUTPUT_MARKER)}"
    r"(?:noise_(?:mean|median)|sr\d*x|mean|median|sr)\Z",
    re.IGNORECASE,
)

#: Lazily filled 16 bit linear -> sRGB-encoded lookup table.
_SRGB_LUT: Optional[np.ndarray] = None

#: Standard XYZ (D65) -> linear sRGB matrix.
_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

#: DNG tags copied from the reference file when writing a derived DNG.
#: Geometry describing tags (active area, crop size, CFA pattern, opcodes...)
#: are intentionally dropped: the output is a demosaiced RGB image, not a
#: mosaic, so those tags would make the file invalid.
_DNG_COLOUR_TAGS = (
    "DNGVersion",
    "DNGBackwardVersion",
    "UniqueCameraModel",
    "LocalizedCameraModel",
    "ColorMatrix1",
    "ColorMatrix2",
    "CameraCalibration1",
    "CameraCalibration2",
    "ReductionMatrix1",
    "ReductionMatrix2",
    "AnalogBalance",
    "AsShotNeutral",
    "AsShotWhiteXY",
    "BaselineExposure",
    "BaselineNoise",
    "BaselineSharpness",
    "LinearResponseLimit",
    "CameraSerialNumber",
    "LensInfo",
    "ChromaBlurRadius",
    "AntiAliasStrength",
    "ShadowScale",
    "CalibrationIlluminant1",
    "CalibrationIlluminant2",
    "BestQualityScale",
    "RawDataUniqueID",
    "OriginalRawFileName",
    "NoiseProfile",
)

#: TIFF datatype code -> tifffile letter used in ``extratags``.
_DATATYPE_LETTERS = {
    1: "B",
    2: "s",
    3: "H",
    4: "I",
    5: "2I",
    6: "b",
    7: "B",
    8: "h",
    9: "i",
    10: "2i",
    11: "f",
    12: "d",
    13: "I",
    16: "Q",
    17: "q",
    18: "Q",
}

_TAG_DNG_VERSION = 50706
_TAG_DNG_BACKWARD_VERSION = 50707
_TAG_UNIQUE_CAMERA_MODEL = 50708
_TAG_WHITE_LEVEL = 50717
_TAG_COLOR_MATRIX1 = 50721
_TAG_AS_SHOT_NEUTRAL = 50728
_TAG_PHOTOMETRIC_LINEAR_RAW = 34892


class FrameReadError(RuntimeError):
    """Raised when an input frame cannot be decoded."""


class OutputWriteError(RuntimeError):
    """Raised when the processed result cannot be written."""


@dataclass
class FrameMetadata:
    """Colour information required to write a colourimetrically sane output."""

    width: int
    height: int
    is_raw: bool = False
    camera_model: Optional[str] = None
    color_matrix: Optional[np.ndarray] = None
    as_shot_neutral: Optional[np.ndarray] = None
    source_path: str = ""
    dng_tags: List[Tuple[int, str, int, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def is_supported(path: os.PathLike[str] | str) -> bool:
    """Return ``True`` when *path* has a supported image extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def is_raw(path: os.PathLike[str] | str) -> bool:
    """Return ``True`` when *path* looks like a RAW file."""
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def is_tool_output(path: os.PathLike[str] | str) -> bool:
    """Return ``True`` for files named like this tool's own output.

    Default outputs are named ``<folder>_stack_<tag>.<ext>``; such files are
    ignored during discovery so repeated runs never stack a previous result
    that was written next to the sources.
    """
    return bool(OUTPUT_NAME_RE.search(Path(path).stem))


def _is_skipped_dir(name: str) -> bool:
    """Return ``True`` for directories that never hold source frames."""
    lowered = name.lower()
    return lowered == "output" or name.startswith("_") or lowered.endswith("_aligned")


def discover_frames(
    directory: os.PathLike[str] | str,
    recursive: bool = False,
    exclude: Optional[Sequence[os.PathLike[str] | str]] = None,
) -> List[Path]:
    """Return the sorted list of supported frames inside *directory*.

    Sorting uses a natural key so that ``IMG_2.DNG`` comes before
    ``IMG_10.DNG``.  Previous results of this tool are never fed back into the
    stack: files named like ``<folder>_stack_<tag>.<ext>`` are skipped, as are
    directories named ``output``, ``*_aligned`` or starting with an underscore.

    Args:
        directory: Folder to scan.
        recursive: Also search sub-folders.
        exclude: Additional paths to ignore (for example the output file of the
            current run when it is written next to the sources).
    """
    root = Path(directory)
    if not root.is_dir():
        raise FrameReadError(f"Input directory does not exist: {root}")

    skip = {Path(path).expanduser().resolve() for path in (exclude or [])}

    def usable(path: Path) -> bool:
        if not path.is_file() or not is_supported(path) or is_tool_output(path):
            return False
        return path.resolve() not in skip

    if recursive:
        iterator = (
            p
            for p in root.rglob("*")
            if usable(p)
            and not any(
                _is_skipped_dir(part) for part in p.relative_to(root).parts[:-1]
            )
        )
    else:
        iterator = (p for p in root.iterdir() if usable(p))

    return sorted(iterator, key=lambda p: _natural_key(p.name))


def natural_key(text: str) -> List[Any]:
    """Public alias of :func:`_natural_key` for callers sorting file lists."""
    return _natural_key(text)


def _natural_key(text: str) -> List[Any]:
    """Split *text* into digit/non-digit chunks for human friendly sorting."""
    key: List[Any] = []
    digits = ""
    letters = ""
    for char in text.lower():
        if char.isdigit():
            if letters:
                key.append(letters)
                letters = ""
            digits += char
        else:
            if digits:
                key.append((1, int(digits)))
                digits = ""
            letters += char
    if digits:
        key.append((1, int(digits)))
    if letters:
        key.append(letters)
    return key


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def load_frame(path: os.PathLike[str] | str, half_size: bool = False) -> np.ndarray:
    """Load *path* as a linear float32 ``(H, W, 3)`` RGB array in ``[0, 1]``.

    Args:
        path: Input file.
        half_size: Request a half resolution demosaic from LibRaw.  Only used
            for fast previews/registration, never for the final stack.

    Raises:
        FrameReadError: When the file is missing, corrupt or unsupported.
    """
    file_path = str(path)
    if not os.path.isfile(file_path):
        raise FrameReadError(f"Frame not found: {file_path}")

    if is_raw(file_path):
        rgb = _load_raw(file_path, half_size=half_size)
    else:
        rgb = _load_raster(file_path, half_size=half_size)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise FrameReadError(f"Unexpected image shape {rgb.shape} for {file_path}")
    if rgb.size == 0:
        raise FrameReadError(f"Empty image data in {file_path}")
    return np.ascontiguousarray(rgb, dtype=np.float32)


def _load_raw(path: str, half_size: bool) -> np.ndarray:
    """Decode a RAW file to linear camera RGB using rawpy."""
    if rawpy is None:
        raise FrameReadError(
            "rawpy is not installed; install it with 'pip install rawpy' "
            "or provide 16-bit TIFF frames instead."
        )
    try:
        with rawpy.imread(path) as raw:
            # Three-channel LinearRaw DNGs (including this tool's own output)
            # are already demosaiced.  Reading the raw buffer directly avoids
            # LibRaw's postprocess normalisation, which would rescale and clip
            # the samples.
            linear = _linear_raw_rgb(raw)
            if linear is not None:
                if half_size:
                    height, width = linear.shape[:2]
                    linear = cv2.resize(
                        linear,
                        (max(1, width // 2), max(1, height // 2)),
                        interpolation=cv2.INTER_AREA,
                    )
                return np.ascontiguousarray(linear, dtype=np.float32)

            data = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                half_size=half_size,
                four_color_rgb=False,
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                output_color=rawpy.ColorSpace.raw,  # camera RGB, no colour matrix
                output_bps=16,
                gamma=(1, 1),  # linear, no tone curve
                no_auto_bright=True,
                use_camera_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0],  # rawpy requires a list
                highlight_mode=rawpy.HighlightMode.Clip,
            )
    except Exception as exc:  # rawpy raises LibRawError and friends
        raise FrameReadError(f"Cannot decode RAW file {path}: {exc}") from exc

    if data.dtype != np.uint16:
        raise FrameReadError(f"Unexpected RAW output type {data.dtype} for {path}")
    return data.astype(np.float32) * np.float32(1.0 / 65535.0)


def _linear_raw_rgb(raw: "rawpy.RawPy") -> Optional[np.ndarray]:
    """Return exact linear RGB from a demosaiced LinearRaw DNG, else ``None``.

    Mosaiced camera files expose a 2D ``raw_image`` and must go through
    ``postprocess``; 3-channel DNGs expose an ``(H, W, >=3)`` buffer whose
    samples are already camera RGB and can be normalised by hand.
    """
    image = raw.raw_image
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None

    samples = image[..., :3].astype(np.float32)
    black = np.asarray(raw.black_level_per_channel, dtype=np.float32)[:3]
    white = raw.white_level
    white_array = np.asarray(white, dtype=np.float32).reshape(-1)
    white_rgb = white_array[:3] if white_array.size >= 3 else np.repeat(white_array[0], 3)
    scale = np.maximum(white_rgb - black, 1.0)
    return np.clip((samples - black) / scale, 0.0, 1.0).astype(np.float32)


def _load_raster(path: str, half_size: bool) -> np.ndarray:
    """Decode a TIFF/PNG/JPEG frame to linear float32 RGB."""
    suffix = Path(path).suffix.lower()
    data: Optional[np.ndarray] = None
    try:
        if suffix in {".tif", ".tiff"}:
            data = tifffile.imread(path)
        else:
            flags = cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION
            data = cv2.imread(path, flags)
            if data is not None and data.ndim == 3:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        raise FrameReadError(f"Cannot decode image {path}: {exc}") from exc

    if data is None:
        raise FrameReadError(f"Cannot decode image {path}: unsupported or corrupt file")

    array = np.asarray(data)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3:
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.shape[2] >= 4:
            array = array[..., :3]
        elif array.shape[2] != 3:
            raise FrameReadError(f"Unsupported channel count in {path}: {array.shape}")

    array = _to_float01(array)
    if _is_display_referred(path):
        # PNG/JPEG and 8 bit TIFF values are gamma encoded; remove the sRGB
        # curve so stacking and drizzle operate on linear light.
        array = _srgb_to_linear(array)
    if half_size:
        height, width = array.shape[:2]
        array = cv2.resize(
            array,
            (max(1, width // 2), max(1, height // 2)),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def _is_display_referred(path: str) -> bool:
    """Return ``True`` when *path* stores sRGB-encoded (gamma) samples.

    PNG, JPEG and BMP are display containers and always carry the sRGB curve.
    16 bit TIFF is the linear container this tool writes, while 8 bit TIFF is
    treated like the other display formats.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _DISPLAY_EXTENSIONS:
        return True
    if suffix in {".tif", ".tiff"}:
        try:
            with tifffile.TiffFile(path) as tif:
                return int(tif.pages[0].bitspersample) <= 8
        except Exception:  # pragma: no cover - header is read again on failure
            return False
    return False


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer curve on float data in ``[0, 1]``."""
    encoded = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    linear = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4, dtype=np.float32),
    )
    return linear.astype(np.float32)


def _srgb_lut() -> np.ndarray:
    """Return (and cache) the 16 bit linear -> sRGB-encoded lookup table."""
    global _SRGB_LUT
    if _SRGB_LUT is None:
        linear = np.arange(65536, dtype=np.float64) / 65535.0
        encoded = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
        )
        _SRGB_LUT = np.rint(np.clip(encoded, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return _SRGB_LUT


def _linear_to_srgb16(data: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer curve to 16 bit linear samples."""
    return _srgb_lut()[np.asarray(data, dtype=np.uint16)]


def _to_float01(array: np.ndarray) -> np.ndarray:
    """Normalise integer image data to float32 in ``[0, 1]``."""
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    if np.issubdtype(array.dtype, np.floating):
        result = array.astype(np.float32)
        peak = float(np.nanmax(result)) if result.size else 0.0
        if peak > 1.0 + 1e-3:
            # Floating point TIFFs frequently store 0..65535 or 0..1 values.
            result = result / (65535.0 if peak > 255.0 else 1.0)
        return np.clip(np.nan_to_num(result), 0.0, 1.0)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return array.astype(np.float32) / float(max(1, info.max))
    raise FrameReadError(f"Unsupported sample type: {array.dtype}")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def read_metadata(path: os.PathLike[str] | str) -> FrameMetadata:
    """Read the colour metadata needed for output serialisation."""
    file_path = str(path)
    if is_raw(file_path) and rawpy is not None:
        try:
            with rawpy.imread(file_path) as raw:
                sizes = raw.sizes
                height = int(sizes.height)
                width = int(sizes.width)
        except Exception as exc:
            raise FrameReadError(f"Cannot inspect RAW file {file_path}: {exc}") from exc
    else:
        try:
            with tifffile.TiffFile(file_path) as tif:
                page = tif.pages[0]
                width, height = int(page.imagewidth), int(page.imagelength)
        except Exception:
            data = _load_raster(file_path, half_size=True)
            height, width = data.shape[:2]

    meta = FrameMetadata(
        width=width, height=height, is_raw=is_raw(file_path), source_path=file_path
    )
    if is_raw(file_path):
        _read_dng_colour_tags(file_path, meta)
    return meta


def _tag_floats(tag: Any) -> np.ndarray:
    """Return a TIFF tag value as floats, resolving RATIONAL pairs.

    tifffile exposes RATIONAL/SRATIONAL values as a flat sequence of
    numerator/denominator pairs, so they need explicit division.
    """
    values = np.asarray(tag.value, dtype=np.float64).reshape(-1)
    if int(tag.dtype) in (5, 10) and values.size % 2 == 0:  # RATIONAL, SRATIONAL
        pairs = values.reshape(-1, 2)
        divisors = np.where(pairs[:, 1] == 0, 1.0, pairs[:, 1])
        return pairs[:, 0] / divisors
    return values


def _rational_values(values: np.ndarray, signed: bool) -> List[int]:
    """Serialise floats as the flat numerator/denominator list tifffile wants."""
    result: List[int] = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        numerator = int(round(float(value) * 1_000_000.0))
        if not signed:
            numerator = max(0, numerator)
        result.extend((numerator, 1_000_000))
    return result


def _read_dng_colour_tags(path: str, meta: FrameMetadata) -> None:
    """Populate *meta* with DNG colour tags when the file carries them.

    Tag values must be materialised while the file is still open: tifffile
    reads values lazily and raises once the handle is closed.
    """
    entries: List[Tuple[int, str, int, Any]] = []
    model: Optional[str] = None
    matrix: Optional[np.ndarray] = None
    neutral: Optional[np.ndarray] = None

    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                return
            tags = {tag.name: tag for tag in tif.pages[0].tags.values()}

            for name in _DNG_COLOUR_TAGS:
                tag = tags.get(name)
                if tag is None:
                    continue
                try:
                    letter = _DATATYPE_LETTERS.get(int(tag.dtype))
                    if letter is None:
                        continue
                    entries.append((int(tag.code), letter, int(tag.count), tag.value))
                except Exception:  # pragma: no cover - malformed tag
                    continue

            model_tag = tags.get("UniqueCameraModel") or tags.get("LocalizedCameraModel")
            if model_tag is not None:
                try:
                    raw_model = model_tag.value
                    model = raw_model.decode("utf-8", "replace") if isinstance(
                        raw_model, bytes
                    ) else str(raw_model)
                except Exception:
                    model = None

            matrix_tag = tags.get("ColorMatrix2") or tags.get("ColorMatrix1")
            if matrix_tag is not None:
                try:
                    candidate = _tag_floats(matrix_tag).reshape(3, 3)
                    if np.all(np.isfinite(candidate)) and abs(np.linalg.det(candidate)) > 1e-9:
                        matrix = candidate
                except Exception:
                    matrix = None

            neutral_tag = tags.get("AsShotNeutral")
            if neutral_tag is not None:
                try:
                    candidate_neutral = _tag_floats(neutral_tag).reshape(-1)
                    if candidate_neutral.size >= 3 and np.all(candidate_neutral[:3] > 0):
                        neutral = candidate_neutral[:3]
                except Exception:
                    neutral = None
    except Exception as exc:  # pragma: no cover - defensive, some RAWs are odd
        LOGGER.debug("Could not read DNG tags from %s: %s", path, exc)
        return

    meta.dng_tags.extend(entries)
    meta.camera_model = model
    meta.color_matrix = matrix
    meta.as_shot_neutral = neutral


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def save_image(
    path: os.PathLike[str] | str,
    rgb: np.ndarray,
    fmt: str = "tiff",
    metadata: Optional[FrameMetadata] = None,
) -> Path:
    """Write *rgb* (float32, ``[0, 1]``, linear) as 16 bit image data.

    Args:
        path: Destination path; its format is selected by *fmt*.
        rgb: Linear float32 RGB image.
        fmt: ``"tiff"``, ``"png"`` or ``"dng"``.  DNG failures fall back to
            TIFF.
        metadata: Source colour metadata, used for camera -> sRGB conversion
            and for DNG tag serialisation.

    Returns:
        The path that was actually written (may differ from *path* after a
        fallback).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _to_uint16(rgb)
    fmt = fmt.lower()

    if fmt == "dng":
        try:
            dng_path = destination.with_suffix(".dng")
            _write_dng16(dng_path, data, metadata)
            LOGGER.info("Wrote 16-bit DNG: %s", dng_path)
            return dng_path
        except Exception as exc:
            fallback = destination.with_suffix(".tif")
            LOGGER.warning(
                "DNG serialisation unavailable (%s); writing 16-bit TIFF instead: %s",
                exc,
                fallback,
            )
            _write_tiff16(fallback, data, metadata)
            return fallback

    if fmt == "png":
        png_path = destination.with_suffix(".png")
        _write_png16(png_path, data, metadata)
        LOGGER.info("Wrote 16-bit PNG: %s", png_path)
        return png_path

    tiff_path = (
        destination
        if destination.suffix.lower() in {".tif", ".tiff"}
        else destination.with_suffix(".tif")
    )
    _write_tiff16(tiff_path, data, metadata)
    LOGGER.info("Wrote 16-bit TIFF: %s", tiff_path)
    return tiff_path


def _to_uint16(rgb: np.ndarray) -> np.ndarray:
    """Quantise a float image in ``[0, 1]`` to 16 bit unsigned integers."""
    scaled = np.clip(np.nan_to_num(np.asarray(rgb, dtype=np.float32)), 0.0, 1.0)
    return np.rint(scaled * 65535.0).astype(np.uint16)


def _write_tiff16(path: Path, data: np.ndarray, metadata: Optional[FrameMetadata]) -> None:
    """Write 16 bit RGB TIFF, converting camera RGB to linear sRGB if possible."""
    rgb = _camera_to_srgb(data, metadata) if metadata is not None else data
    description = (
        "Linear 16-bit RGB produced by MultiImageOptimizer"
        + (f" from {Path(metadata.source_path).name}" if metadata and metadata.source_path else "")
    )
    try:
        tifffile.imwrite(
            str(path),
            rgb,
            photometric="rgb",
            planarconfig="contig",
            compression="zlib",
            metadata={"description": description},
            datetime=_now_string(),
        )
    except Exception as exc:
        raise OutputWriteError(f"Cannot write TIFF {path}: {exc}") from exc


def _write_png16(path: Path, data: np.ndarray, metadata: Optional[FrameMetadata]) -> None:
    """Write a display referred 16 bit RGB PNG.

    PNG viewers assume sRGB, so the camera RGB samples are converted to sRGB
    and then gamma encoded.  Writing linear values straight into a PNG would
    produce a visibly underexposed image.
    """
    rgb = _camera_to_srgb(data, metadata) if metadata is not None else data
    encoded = _linear_to_srgb16(rgb)
    # OpenCV expects channel order BGR.
    bgr = np.ascontiguousarray(encoded[..., ::-1])
    try:
        ok, buffer = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    except cv2.error as exc:
        raise OutputWriteError(f"Cannot encode PNG {path}: {exc}") from exc
    if not ok:  # pragma: no cover - defensive, imencode raises instead
        raise OutputWriteError(f"Cannot encode PNG {path}")
    try:
        path.write_bytes(buffer.tobytes())
    except OSError as exc:
        raise OutputWriteError(f"Cannot write PNG {path}: {exc}") from exc


def _camera_to_srgb(data: np.ndarray, metadata: FrameMetadata) -> np.ndarray:
    """Convert camera RGB samples to linear sRGB using DNG colour tags.

    Values are normalised so that the as-shot neutral maps to a neutral white
    point.  When no matrix is available the data is returned untouched, which
    is the correct behaviour for already-sRGB raster inputs.
    """
    if metadata.color_matrix is None:
        return data

    try:
        # ColorMatrix1 maps XYZ -> camera; invert for camera -> XYZ.
        camera_to_xyz = np.linalg.inv(metadata.color_matrix)
        matrix = _XYZ_TO_SRGB @ camera_to_xyz
        if metadata.as_shot_neutral is not None:
            white_camera = 1.0 / metadata.as_shot_neutral
            white_srgb = matrix @ white_camera
            white_srgb = np.where(np.abs(white_srgb) < 1e-6, 1.0, white_srgb)
            matrix = matrix / white_srgb[:, None]

        samples = np.asarray(data, dtype=np.float32) * np.float32(1.0 / 65535.0)
        converted = samples.reshape(-1, 3).astype(np.float64) @ matrix.T
        converted = np.clip(converted, 0.0, 1.0).astype(np.float32)
        return np.rint(converted.reshape(samples.shape) * 65535.0).astype(np.uint16)
    except Exception as exc:  # pragma: no cover - numerically odd matrices
        LOGGER.debug("Colour conversion skipped: %s", exc)
        return data


def _write_dng16(path: Path, data: np.ndarray, metadata: Optional[FrameMetadata]) -> None:
    """Write a 3 channel LinearRaw DNG and verify that LibRaw can reopen it."""
    if metadata is None or metadata.color_matrix is None:
        raise OutputWriteError(
            "no colour matrix available in the source frames; cannot build a valid DNG"
        )

    extratags: List[Tuple[int, str, int, Any]] = []
    seen = set()
    for code, dtype, count, value in metadata.dng_tags:
        if code == _TAG_WHITE_LEVEL:
            extratags.append((_TAG_WHITE_LEVEL, "H", 1, 65535))
            seen.add(_TAG_WHITE_LEVEL)
            continue
        if code in (_TAG_PHOTOMETRIC_LINEAR_RAW,):
            continue
        extratags.append((code, dtype, count, value))
        seen.add(code)

    if _TAG_DNG_VERSION not in seen:
        extratags.append((_TAG_DNG_VERSION, "B", 4, (1, 4, 0, 0)))
    if _TAG_DNG_BACKWARD_VERSION not in seen:
        extratags.append((_TAG_DNG_BACKWARD_VERSION, "B", 4, (1, 1, 0, 0)))
    if _TAG_UNIQUE_CAMERA_MODEL not in seen:
        extratags.append(
            (_TAG_UNIQUE_CAMERA_MODEL, "s", 0, metadata.camera_model or "MultiImageOptimizer")
        )
    if _TAG_WHITE_LEVEL not in seen:
        extratags.append((_TAG_WHITE_LEVEL, "H", 1, 65535))
    # The source may carry its colour tags only as parsed metadata (for example
    # when it was produced in memory), so rebuild them when they are missing.
    if metadata.color_matrix is not None and _TAG_COLOR_MATRIX1 not in seen:
        extratags.append(
            (_TAG_COLOR_MATRIX1, "2i", 9, _rational_values(metadata.color_matrix, signed=True))
        )
    if metadata.as_shot_neutral is not None and _TAG_AS_SHOT_NEUTRAL not in seen:
        extratags.append(
            (
                _TAG_AS_SHOT_NEUTRAL,
                "2I",
                3,
                _rational_values(metadata.as_shot_neutral, signed=False),
            )
        )

    try:
        tifffile.imwrite(
            str(path),
            data,
            byteorder="<",
            photometric=_TAG_PHOTOMETRIC_LINEAR_RAW,
            planarconfig="contig",
            compression=None,
            extratags=extratags,
            metadata=None,
            datetime=_now_string(),
        )
    except Exception as exc:
        raise OutputWriteError(f"Cannot serialise DNG {path}: {exc}") from exc

    _verify_dng(path, data)


def _verify_dng(path: Path, data: np.ndarray) -> None:
    """Reopen *path* with LibRaw to make sure the DNG is actually usable."""
    if rawpy is None:
        return
    try:
        with rawpy.imread(str(path)) as raw:
            linear = _linear_raw_rgb(raw)
            if linear is not None:
                decoded = np.rint(linear * 65535.0).astype(np.uint16)
            else:
                decoded = raw.postprocess(
                    output_bps=16,
                    gamma=(1, 1),
                    no_auto_bright=True,
                    use_camera_wb=False,
                    user_wb=[1.0, 1.0, 1.0, 1.0],  # rawpy requires a list
                    output_color=rawpy.ColorSpace.raw,
                )
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise OutputWriteError(f"LibRaw rejected the generated DNG: {exc}") from exc

    decoded = np.asarray(decoded)
    if decoded.shape != data.shape:
        path.unlink(missing_ok=True)
        raise OutputWriteError(
            f"Generated DNG decodes to {decoded.shape}, expected {data.shape}"
        )
    # A tolerance check catches byte order / stride mistakes.
    sample = decoded[:: max(1, decoded.shape[0] // 64), :: max(1, decoded.shape[1] // 64)]
    expected = data[:: max(1, data.shape[0] // 64), :: max(1, data.shape[1] // 64)]
    if sample.std() > 1.0 and np.mean(np.abs(sample.astype(np.float32) - expected)) > 256:
        path.unlink(missing_ok=True)
        raise OutputWriteError("Generated DNG does not round-trip through LibRaw")


def _now_string() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y:%m:%d %H:%M:%S")


def output_tag(mode: str, method: str, upscale: int = 2) -> str:
    """Return the filename tag that names the applied method.

    ``noise_mean``, ``noise_median`` or ``sr2x`` (the drizzle factor is part
    of the tag because it changes the pixel dimensions).
    """
    if mode == "super_resolution":
        return f"sr{max(2, int(upscale))}x"
    if mode == "noise_reduction":
        return f"noise_{method}"
    return "stack"


def default_output_path(
    input_dir: os.PathLike[str] | str,
    mode: str,
    method: str,
    fmt: str,
    upscale: int = 2,
) -> Path:
    """Build the default output path inside the input folder."""
    root = Path(input_dir)
    suffix = _OUTPUT_SUFFIXES.get(fmt.lower(), ".tif")
    tag = output_tag(mode, method, upscale)
    name = f"{root.name or 'stack'}{OUTPUT_MARKER}{tag}{suffix}"
    # Default next to the sources; the marker in the name keeps that folder
    # usable as input for the next run.
    return root / name
