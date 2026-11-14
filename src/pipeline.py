"""End-to-end orchestration shared by the command line and the GUI.

The pipeline is deliberately UI agnostic: it reports progress through a plain
callback, emits log lines through another callback and honours a cancellation
predicate, which keeps it usable from a worker thread without touching any Qt
API.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

import aligner
import rawio
import stacker
from aligner import AlignmentResult, GrayPreview
from rawio import FrameReadError, FrameMetadata
from stacker import StackStats

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]

MODES = ("noise_reduction", "super_resolution", "both")
STACK_METHODS = ("mean", "median")
OUTPUT_FORMATS = ("dng", "tiff", "png")
MODE_LABELS = {
    "noise_reduction": "Noise Reduction (Stacking)",
    "super_resolution": "Super-Resolution",
    "both": "Noise Reduction + Super-Resolution",
}
METHOD_LABELS = {"mean": "Mean", "median": "Median"}


class PipelineError(RuntimeError):
    """Raised for user facing pipeline failures."""


class PipelineCancelled(PipelineError):
    """Raised when the user cancels a running job."""


@dataclass
class PipelineConfig:
    """All tunables of one processing run."""

    #: Folder to scan; may be empty when ``input_files`` is given.
    input_dir: str = ""
    #: Explicit file selection (dropped or picked images); overrides discovery.
    input_files: List[str] = field(default_factory=list)
    output_path: str = ""
    mode: str = "noise_reduction"
    stack_method: str = "mean"
    output_format: str = "dng"
    upscale: int = 2
    pixfrac: float = 0.75
    reference_index: Optional[int] = None
    max_dim: int = 2000
    recursive: bool = False
    limit: Optional[int] = None
    temp_dir: Optional[str] = None
    export_aligned: bool = False
    aligned_dir: str = ""

    def validate(self) -> None:
        """Raise :class:`PipelineError` when the configuration is invalid."""
        if not self.input_dir.strip() and not self.input_files:
            raise PipelineError("No input folder or file selection given")
        if self.mode not in MODES:
            raise PipelineError(f"Unknown mode '{self.mode}'")
        if self.stack_method not in STACK_METHODS:
            raise PipelineError(f"Unknown stack method '{self.stack_method}'")
        if self.output_format not in OUTPUT_FORMATS:
            raise PipelineError(f"Unknown output format '{self.output_format}'")
        # Combined mode drizzles too, so it needs the same limits.
        if self.mode in ("super_resolution", "both"):
            if not 2 <= self.upscale <= 6:
                raise PipelineError("Upscale factor must be between 2 and 6")
            if not 0.0 < self.pixfrac <= 1.0:
                raise PipelineError("pixfrac must be in the range (0, 1]")
        if self.max_dim < 320:
            raise PipelineError("max_dim must be at least 320 pixels")


@dataclass
class FrameReport:
    """Per-frame outcome, shown in logs and used for skipped frame warnings."""

    index: int
    name: str
    accepted: bool
    message: str = ""
    matches: int = 0
    ecc_score: float = 0.0


@dataclass
class _Preparation:
    """Discovery and registration results shared by the pipelines of one run.

    Registration is the expensive and mode independent stage: stacking and
    drizzle both warp the frames with the same per-frame transforms, so a
    combined run computes them once instead of once per pipeline.
    """

    frames: List[Path]
    reference_index: int
    reports: List[FrameReport]
    warnings: List[str]
    usable_paths: List[Path]
    usable_matrices: List[np.ndarray]


@dataclass
class RunResult:
    """Everything the caller needs to report a completed run."""

    output_path: Path
    requested_format: str
    stats: StackStats
    frame_count: int
    used_frames: int
    reference_name: str
    frames: List[FrameReport] = field(default_factory=list)
    elapsed: float = 0.0
    warnings: List[str] = field(default_factory=list)
    aligned_dir: Optional[Path] = None
    aligned_count: int = 0

    @property
    def height(self) -> int:
        return int(self.stats.output_shape[0])

    @property
    def width(self) -> int:
        return int(self.stats.output_shape[1])


def resolve_output_path(config: PipelineConfig) -> Path:
    """Return the output path, falling back to ``<input>/output/<name>``."""
    if config.output_path.strip():
        return Path(config.output_path).expanduser()
    return rawio.default_output_path(
        config.input_dir,
        config.mode,
        config.stack_method,
        config.output_format,
        config.upscale,
    )


def resolve_aligned_dir(config: PipelineConfig, destination: Path) -> Path:
    """Return the folder that receives the aligned frames.

    Defaults to ``<output>_aligned`` next to the result, so a run never mixes
    its intermediates with the source frames.
    """
    if config.aligned_dir.strip():
        return Path(config.aligned_dir).expanduser()
    return destination.parent / f"{destination.stem}_aligned"


def _safe_stem(path: Path) -> str:
    """Return a filename-safe stem so suffixes cannot be misread as extensions."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_-")
    return stem or "frame"


def run_pipelines(
    config: PipelineConfig,
    progress: Optional[ProgressCallback] = None,
    log: Optional[LogCallback] = None,
    cancel: Optional[CancelCallback] = None,
) -> List[RunResult]:
    """Run every pipeline requested by *config* and return all results.

    ``mode="both"`` returns the stacked image and the super-resolution frame
    and writes both.  Frames are discovered, registered, decoded and warped
    once; the stack and the drizzle accumulate from that single pass, so the
    combined mode does not pay for either stage twice.

    The progress fractions cover the shared preparation first and then the
    single combination pass.
    """
    if config.mode != "both":
        return [run_pipeline(config, progress, log, cancel)]

    started = time.perf_counter()
    config = _with_derived_input_dir(config)
    config.validate()
    emit, say, check_cancel = _reporters(progress, log, cancel)

    # Both pipelines need their own destination, and both of them must stay out
    # of frame discovery.
    outputs = [_mode_config(config, mode) for mode in ("noise_reduction", "super_resolution")]
    prepared = _prepare(
        config,
        [resolve_output_path(sub_config) for sub_config in outputs],
        _window(emit, 0.0, 0.30),
        say,
        check_cancel,
        cancel,
    )

    say(
        f"=== {MODE_LABELS['noise_reduction']} + {MODE_LABELS['super_resolution']} "
        "(single pass) ==="
    )
    return _finish_combined(
        outputs,
        prepared,
        _window(emit, 0.30, 1.0),
        say,
        check_cancel,
        cancel,
        time.perf_counter() - started,
    )


def _mode_config(config: PipelineConfig, mode: str) -> PipelineConfig:
    """Return a single-pipeline copy of *config* with its own output path."""
    sub_config = replace(config, mode=mode)
    if sub_config.output_path.strip():
        sub_config.output_path = _tagged_output_path(config, mode)
    return sub_config


def _tagged_output_path(config: PipelineConfig, mode: str) -> str:
    """Derive a per-mode output path from an explicit one (combined mode)."""
    path = Path(config.output_path).expanduser()
    tag = rawio.output_tag(mode, config.stack_method, config.upscale)
    return str(path.with_name(f"{path.stem}_{tag}{path.suffix}"))


def _with_derived_input_dir(config: PipelineConfig) -> PipelineConfig:
    """Fill ``input_dir`` from an explicit file selection when it is missing."""
    if not config.input_dir.strip() and config.input_files:
        first = Path(config.input_files[0]).expanduser()
        config = replace(config, input_dir=str(first.parent))
    return config


def _reporters(
    progress: Optional[ProgressCallback],
    log: Optional[LogCallback],
    cancel: Optional[CancelCallback],
) -> tuple[ProgressCallback, LogCallback, CancelCallback]:
    """Bundle the clamping progress, log and cancellation callbacks."""

    def emit(fraction: float, message: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, fraction)), message)

    def say(message: str) -> None:
        # The caller owns presentation (console logger, GUI log pane); logging
        # here as well would duplicate every line in the CLI.
        if log is not None:
            log(message)

    def check_cancel() -> None:
        if cancel is not None and cancel():
            raise PipelineCancelled("Processing cancelled by user")

    return emit, say, check_cancel


def _window(emit: ProgressCallback, low: float, high: float) -> ProgressCallback:
    """Map a stage's own ``0..1`` progress into the overall ``low..high`` range."""

    def mapped(fraction: float, message: str) -> None:
        # Land exactly on the window edges; 0.65 + 0.35 * 1.0 is not 1.0 in
        # floating point and the GUI reads 100 % from the last value.
        if fraction >= 1.0:
            emit(high, message)
        else:
            emit(low + (high - low) * max(0.0, fraction), message)

    return mapped


def run_pipeline(
    config: PipelineConfig,
    progress: Optional[ProgressCallback] = None,
    log: Optional[LogCallback] = None,
    cancel: Optional[CancelCallback] = None,
) -> RunResult:
    """Run discovery, registration, combination and serialisation.

    Args:
        config: Validated run configuration.
        progress: ``(fraction, message)`` callback; fractions are monotonic.
        log: Receives human readable status lines.
        cancel: Predicate polled between frames; ``True`` aborts the run.

    Raises:
        PipelineError: On invalid input or processing failure.
        PipelineCancelled: When *cancel* is triggered.

    Note:
        ``mode="both"`` must go through :func:`run_pipelines`, which produces
        both results in a single pass over the frames.
    """
    started = time.perf_counter()
    if config.mode == "both":
        raise PipelineError("The combined mode must be run with run_pipelines()")
    config = _with_derived_input_dir(config)
    config.validate()
    emit, say, check_cancel = _reporters(progress, log, cancel)

    destination = resolve_output_path(config)
    prepared = _prepare(
        config, [destination], _window(emit, 0.0, 0.36), say, check_cancel, cancel
    )
    return _finish(
        config,
        prepared,
        _window(emit, 0.36, 1.0),
        say,
        check_cancel,
        cancel,
        time.perf_counter() - started,
    )


def _prepare(
    config: PipelineConfig,
    excludes: Sequence[Path],
    emit: ProgressCallback,
    say: LogCallback,
    check_cancel: CancelCallback,
    cancel: Optional[CancelCallback] = None,
) -> _Preparation:
    """Discover the frames, rank them and register them against a reference.

    Emitted fractions span ``0..1`` of this stage; the caller maps them into
    the overall progress.  The returned transforms are mode independent and
    are reused by every pipeline of a run.
    """
    warnings: List[str] = []

    # -- 1. Discovery -------------------------------------------------------
    if config.input_files:
        frames = _selected_frames(config, excludes)
        source = "the selection"
    else:
        try:
            frames = rawio.discover_frames(
                config.input_dir, recursive=config.recursive, exclude=list(excludes)
            )
        except FrameReadError as exc:
            raise PipelineError(str(exc)) from exc
        source = config.input_dir
    if config.limit:
        frames = frames[: max(1, int(config.limit))]
    if len(frames) < 2:
        raise PipelineError(
            f"Found {len(frames)} supported frame(s) in {source}; "
            "at least two are required. Supported formats: DNG/CR2/NEF/ARW... and TIFF/PNG."
        )
    say(f"Found {len(frames)} frames in {source}")
    emit(0.03, "Scanning input folder")

    # -- 2. Previews and sharpness -----------------------------------------
    previews: List[GrayPreview] = []
    sharpness: List[float] = []
    for index, path in enumerate(frames):
        check_cancel()
        emit(0.06 + 0.28 * index / len(frames), f"Preparing frame {index + 1}/{len(frames)}")
        try:
            pixel_size = 2.0 if rawio.is_raw(path) else 1.0
            preview_rgb = rawio.load_frame(path, half_size=pixel_size > 1.0)
        except FrameReadError as exc:
            raise PipelineError(f"Frame {path.name} could not be decoded: {exc}") from exc
        try:
            preview = aligner.prepare_gray(preview_rgb, config.max_dim, pixel_size=pixel_size)
        except aligner.AlignmentError as exc:
            raise PipelineError(f"Frame {path.name} is unusable: {exc}") from exc
        previews.append(preview)
        sharpness.append(aligner.sharpness(preview))
        del preview_rgb

    reference_index = _choose_reference(config, frames, sharpness)
    say(
        f"Reference frame: {frames[reference_index].name} "
        f"(sharpness {sharpness[reference_index]:.1f})"
    )

    # -- 3. Registration ----------------------------------------------------
    try:
        alignments: List[AlignmentResult] = aligner.align_frames(
            previews,
            reference_index,
            progress=lambda index, message: emit(
                0.34 + 0.63 * index / len(previews), message
            ),
            cancelled=cancel,
        )
    except aligner.AlignmentError as exc:
        check_cancel()
        raise PipelineError(str(exc)) from exc
    finally:
        previews.clear()

    reports: List[FrameReport] = []
    usable_matrices: List[np.ndarray] = []
    usable_paths: List[Path] = []
    for result in alignments:
        path = frames[result.index]
        reports.append(
            FrameReport(
                index=result.index,
                name=path.name,
                accepted=result.accepted,
                message=result.message,
                matches=result.matches,
                ecc_score=result.ecc_score,
            )
        )
        if result.accepted:
            usable_paths.append(path)
            usable_matrices.append(result.matrix)
            say(result.describe())
        else:
            warning = f"Skipped {path.name}: {result.message}"
            warnings.append(warning)
            say(warning)

    check_cancel()
    if len(usable_paths) < 2:
        raise PipelineError(
            "Not enough frames could be registered to build a stack. "
            "Check that the frames overlap and share visible detail."
        )
    emit(1.0, f"Registered {len(usable_paths)}/{len(frames)} frames")
    return _Preparation(
        frames=frames,
        reference_index=reference_index,
        reports=reports,
        warnings=warnings,
        usable_paths=usable_paths,
        usable_matrices=usable_matrices,
    )


def _finish(
    config: PipelineConfig,
    prepared: _Preparation,
    emit: ProgressCallback,
    say: LogCallback,
    check_cancel: CancelCallback,
    cancel: Optional[CancelCallback],
    elapsed: float,
) -> RunResult:
    """Combine the prepared frames with one pipeline and write the result.

    The emitted fractions span ``0..1`` of this stage, so the caller can place
    it inside the overall progress bar.
    """
    destination = resolve_output_path(config)
    aligned_dir, aligned_callback, aligned_written = _aligned_exporter(
        config, prepared, destination, say
    )

    # -- 4. Combination -----------------------------------------------------
    try:
        paths = [str(path) for path in prepared.usable_paths]
        if config.mode == "super_resolution":
            image, stats = stacker.stack_super_resolution(
                paths,
                prepared.usable_matrices,
                upscale=config.upscale,
                pixfrac=config.pixfrac,
                progress=lambda fraction, message: emit(0.84 * fraction, message),
                cancelled=cancel,
                aligned=aligned_callback,
            )
        else:
            image, stats = stacker.stack_noise_reduction(
                paths,
                prepared.usable_matrices,
                method=config.stack_method,
                temp_dir=config.temp_dir,
                progress=lambda fraction, message: emit(0.84 * fraction, message),
                cancelled=cancel,
                aligned=aligned_callback,
            )
    except (stacker.StackError, FrameReadError) as exc:
        check_cancel()
        raise PipelineError(str(exc)) from exc

    _say_combination(say, stats)
    if aligned_dir is not None:
        say(f"Exported {len(aligned_written)} aligned frames to {aligned_dir}")

    # -- 5. Serialisation ---------------------------------------------------
    emit(0.88, "Writing output")
    result = _save_result(
        config,
        destination,
        image,
        stats,
        prepared,
        say,
        check_cancel,
        elapsed,
        aligned_dir,
        len(aligned_written),
    )
    emit(1.0, "Done")
    return result


def _finish_combined(
    configs: Sequence[PipelineConfig],
    prepared: _Preparation,
    emit: ProgressCallback,
    say: LogCallback,
    check_cancel: CancelCallback,
    cancel: Optional[CancelCallback],
    elapsed: float,
) -> List[RunResult]:
    """Stack and drizzle in a single pass, then write both results.

    Both engines stream the frames, so combining them means every frame is
    decoded and warped once instead of once per pipeline.  The aligned export
    happens here as well, once: duplicating it would write identical files.
    """
    stack_config, super_config = configs
    destinations = [resolve_output_path(config) for config in configs]
    aligned_dir, aligned_callback, aligned_written = _aligned_exporter(
        stack_config, prepared, destinations[0], say
    )

    # -- 4. Combination -----------------------------------------------------
    try:
        stack_image, stack_stats, super_image, super_stats = stacker.stack_combined(
            [str(path) for path in prepared.usable_paths],
            prepared.usable_matrices,
            method=stack_config.stack_method,
            upscale=super_config.upscale,
            pixfrac=super_config.pixfrac,
            progress=lambda fraction, message: emit(0.92 * fraction, message),
            cancelled=cancel,
            aligned=aligned_callback,
            temp_dir=stack_config.temp_dir,
        )
    except (stacker.StackError, FrameReadError) as exc:
        check_cancel()
        raise PipelineError(str(exc)) from exc

    _say_combination(say, stack_stats)
    _say_combination(say, super_stats)
    if aligned_dir is not None:
        say(f"Exported {len(aligned_written)} aligned frames to {aligned_dir}")

    # -- 5. Serialisation ---------------------------------------------------
    emit(0.96, f"Writing {destinations[0].name}")
    results = [
        _save_result(
            stack_config,
            destinations[0],
            stack_image,
            stack_stats,
            prepared,
            say,
            check_cancel,
            elapsed,
            aligned_dir,
            len(aligned_written),
        )
    ]
    del stack_image
    emit(0.98, f"Writing {destinations[1].name}")
    results.append(
        _save_result(
            super_config,
            destinations[1],
            super_image,
            super_stats,
            prepared,
            say,
            check_cancel,
            elapsed,
            aligned_dir,
            len(aligned_written),
        )
    )
    emit(1.0, "Done")
    return results


def _aligned_exporter(
    config: PipelineConfig,
    prepared: _Preparation,
    destination: Path,
    say: LogCallback,
) -> tuple[Optional[Path], Optional[Callable[[int, np.ndarray], None]], List[Path]]:
    """Prepare the optional aligned-frame export for *destination*.

    Returns the target folder (or ``None``), the stacker callback and the list
    that collects the written files.
    """
    written: List[Path] = []
    if not config.export_aligned:
        return None, None, written

    aligned_dir = resolve_aligned_dir(config, destination)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    say(f"Exporting aligned frames to {aligned_dir}")
    usable_paths = prepared.usable_paths

    def write_aligned(index: int, image: np.ndarray) -> None:
        """Save one already-warped frame, reusing the stacker's work."""
        source = usable_paths[index]
        target = aligned_dir / f"{index + 1:02d}_{_safe_stem(source)}"
        # RAW sources need their colour tags for DNG/PNG output; rasters are
        # already sRGB and carry no matrix.
        metadata = rawio.read_metadata(source) if rawio.is_raw(source) else None
        try:
            written.append(rawio.save_image(target, image, config.output_format, metadata))
        except (rawio.OutputWriteError, OSError) as exc:
            raise PipelineError(
                f"Could not write the aligned frame {source.name}: {exc}"
            ) from exc

    return aligned_dir, write_aligned, written


def _save_result(
    config: PipelineConfig,
    destination: Path,
    image: np.ndarray,
    stats: StackStats,
    prepared: _Preparation,
    say: LogCallback,
    check_cancel: CancelCallback,
    elapsed: float,
    aligned_dir: Optional[Path] = None,
    aligned_count: int = 0,
) -> RunResult:
    """Write one result image and package it for the caller."""
    check_cancel()
    warnings = list(prepared.warnings)
    metadata = _read_reference_metadata(
        prepared.frames[prepared.reference_index], warnings, say
    )
    try:
        written = rawio.save_image(destination, image, config.output_format, metadata)
    except (rawio.OutputWriteError, OSError) as exc:
        raise PipelineError(f"Could not write the output image: {exc}") from exc

    if written.suffix.lower() != f".{config.output_format}" and config.output_format == "dng":
        message = (
            "DNG output was not available for this input; saved as 16-bit TIFF instead."
        )
        warnings.append(message)
        say(message)

    return RunResult(
        output_path=written,
        requested_format=config.output_format,
        stats=stats,
        frame_count=len(prepared.frames),
        used_frames=len(prepared.usable_paths),
        reference_name=prepared.frames[prepared.reference_index].name,
        frames=list(prepared.reports),
        elapsed=elapsed,
        warnings=warnings,
        aligned_dir=aligned_dir,
        aligned_count=aligned_count,
    )


def _say_combination(say: LogCallback, stats: StackStats) -> None:
    """Log the shape and method of a finished combination."""
    say(
        f"Combined {stats.frames_used} frames into "
        f"{stats.output_shape[1]}x{stats.output_shape[0]} pixels "
        f"({stats.mode.replace('_', ' ')} / method: {stats.method})"
    )


def _selected_frames(config: PipelineConfig, excludes: Sequence[Path]) -> List[Path]:
    """Validate and sort an explicit file selection (dropped images)."""
    frames: List[Path] = []
    missing: List[str] = []
    skipped: List[str] = []
    outputs = {path.expanduser().resolve() for path in excludes}
    for entry in config.input_files:
        path = Path(entry).expanduser()
        if not path.exists():
            missing.append(entry)
        elif not path.is_file() or not rawio.is_supported(path):
            skipped.append(path.name)
        elif rawio.is_tool_output(path):
            skipped.append(path.name)
        elif path.resolve() not in outputs:
            frames.append(path)
    if missing:
        raise PipelineError(
            "These selected file(s) no longer exist: " + ", ".join(missing[:5])
        )
    if skipped:
        raise PipelineError(
            "These selected file(s) are not supported frames: " + ", ".join(skipped[:5])
        )
    return sorted(frames, key=lambda p: rawio.natural_key(p.name))


def _choose_reference(
    config: PipelineConfig,
    frames: Sequence[Path],
    sharpness: Sequence[float],
) -> int:
    """Select the reference frame: explicit index or the sharpest frame."""
    if config.reference_index is not None:
        index = config.reference_index
        if not 0 <= index < len(frames):
            raise PipelineError(
                f"Reference frame {index + 1} is out of range (1..{len(frames)})"
            )
        return index
    best = int(np.argmax(np.asarray(sharpness, dtype=np.float64)))
    return best


def _read_reference_metadata(
    reference: Path,
    warnings: List[str],
    say: LogCallback,
) -> Optional[FrameMetadata]:
    """Read colour metadata for output serialisation, tolerating failures."""
    try:
        return rawio.read_metadata(reference)
    except Exception as exc:  # metadata is a bonus, never fatal
        warning = f"Colour metadata unavailable ({exc}); writing plain 16-bit data."
        warnings.append(warning)
        say(warning)
        return None


def format_summary(result: RunResult) -> str:
    """Build the multi-line summary printed by the CLI."""
    lines = [
        "Processing complete",
        f"  Output          : {result.output_path}",
        f"  Resolution      : {result.width} x {result.height}",
        f"  Frames used     : {result.used_frames} of {result.frame_count}",
        f"  Reference frame : {result.reference_name}",
        f"  Mode            : {MODE_LABELS.get(result.stats.mode, result.stats.mode)}",
    ]
    if result.aligned_dir is not None:
        lines.append(
            f"  Aligned frames  : {result.aligned_count} in {result.aligned_dir}"
        )
    lines += [
        f"  Method          : {result.stats.method}",
        f"  Coverage        : {result.stats.coverage * 100:.1f}%",
        f"  Elapsed         : {result.elapsed:.1f} s",
    ]
    skipped = [frame for frame in result.frames if not frame.accepted]
    if skipped:
        lines.append(f"  Skipped frames  : {len(skipped)}")
        for frame in skipped:
            lines.append(f"    - {frame.name}: {frame.message}")
    return "\n".join(lines)
