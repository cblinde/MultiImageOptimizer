"""Dual-mode entry point: headless CLI when arguments are given, GUI otherwise.

Usage examples::

    python main.py                       # opens the GUI (double-click friendly)
    python main.py --help                # CLI help
    python main.py -i ./frames --mode super_resolution --upscale 2 -o ./out/result.dng
    python main.py ./frames              # folder dropped on the exe: GUI + auto-start
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import List, Optional, Sequence

APP_NAME = "Multi-Image Optimizer"
APP_VERSION = "1.0.0"

LOGGER = logging.getLogger("mio")


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Multi-frame RAW super-resolution and noise reduction engine. "
            "Runs headless when processing options are supplied and opens the GUI "
            "otherwise, including for a folder dropped onto the executable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="",
        help="Folder containing the DNG (or TIFF/PNG) frames.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help=(
            "Output file path. Defaults to "
            "<input>/<folder>_stack_<tag>.dng|tif|png, next to the source frames."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("noise_reduction", "super_resolution", "both"),
        default="noise_reduction",
        help=(
            "Pipeline to run; 'both' produces a noise-reduced stack and a "
            "super-resolution frame in one go."
        ),
    )
    parser.add_argument(
        "--stack-method",
        choices=("mean", "median"),
        default="mean",
        help="Combination rule used by the noise reduction pipeline.",
    )
    parser.add_argument(
        "--output-format",
        choices=("dng", "tiff", "png"),
        default="dng",
        help=(
            "16-bit container to write. dng/tiff stay linear, png is sRGB "
            "encoded for viewing; DNG falls back to TIFF when unavailable."
        ),
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="Linear resolution multiplier for super-resolution (2-6).",
    )
    parser.add_argument(
        "--pixfrac",
        type=float,
        default=0.75,
        help="Drizzle drop size as a fraction of an input pixel (0 < pixfrac <= 1).",
    )
    parser.add_argument(
        "--reference",
        type=int,
        default=None,
        metavar="N",
        help="1-based index of the reference frame. Defaults to the sharpest frame.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2000,
        help="Longest edge of the registration preview in pixels.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search sub-folders for frames (ignores 'output' folders).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only use the first N frames (useful for quick tests).",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Directory for the median scratch file (defaults to the system temp).",
    )
    parser.add_argument(
        "--save-aligned",
        action="store_true",
        help=(
            "Also write every aligned frame, in the selected output format, to "
            "<output>_aligned/ next to the result."
        ),
    )
    parser.add_argument(
        "--aligned-dir",
        default="",
        help="Folder for the aligned frames; implies --save-aligned.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force the graphical interface even when other arguments are given.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "Load the bundled native libraries (Qt, LibRaw, OpenCV) and exit; "
            "useful for diagnosing a broken installation."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help=(
            "Folder or image files dropped onto the executable. Opens the GUI, "
            "selects them and immediately runs both pipelines."
        ),
    )
    parser.add_argument("--log-file", default="", help="Also append the log to this file.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (repeatable).",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print errors.")
    parser.add_argument(
        "--version", action="version", version=f"{APP_NAME} {APP_VERSION}"
    )
    return parser


def configure_logging(verbose: int, quiet: bool, log_file: str) -> None:
    """Configure the root logger for console and optional file output."""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    handlers: List[logging.Handler] = []
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stderr))
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


class ConsoleProgress:
    """Single-line progress renderer for headless runs."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self._last_bucket = -1

    def __call__(self, fraction: float, message: str) -> None:
        if not self.enabled:
            return
        bucket = int(fraction * 100)
        if bucket == self._last_bucket:
            return
        self._last_bucket = bucket
        filled = int(round(fraction * 28))
        bar = "#" * filled + "-" * (28 - filled)
        text = f"\r[{bar}] {bucket:3d}%  {message[:70]:<70}"
        sys.stderr.write(text)
        sys.stderr.flush()

    def finish(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _config_from_args(args: argparse.Namespace):
    from pipeline import PipelineConfig

    # Keep 1-based -> 0-based exact; the pipeline rejects an out-of-range index,
    # so --reference 0 fails instead of silently using the first frame.
    reference = None if args.reference is None else args.reference - 1
    return PipelineConfig(
        input_dir=str(Path(args.input_dir).expanduser()),
        output_path=args.output,
        mode=args.mode,
        stack_method=args.stack_method,
        output_format=args.output_format,
        upscale=args.upscale,
        pixfrac=args.pixfrac,
        reference_index=reference,
        max_dim=args.max_dim,
        recursive=args.recursive,
        limit=args.limit,
        temp_dir=args.temp_dir,
        export_aligned=bool(args.save_aligned or args.aligned_dir),
        aligned_dir=args.aligned_dir,
    )


def run_cli(args: argparse.Namespace) -> int:
    """Execute one headless processing run and return the process exit code."""
    from pipeline import PipelineCancelled, PipelineError, format_summary, run_pipelines

    config = _config_from_args(args)
    progress = ConsoleProgress(enabled=not args.quiet)

    try:
        results = run_pipelines(
            config,
            progress=progress,
            log=lambda message: LOGGER.info(message),
        )
    except PipelineCancelled:
        LOGGER.warning("Cancelled")
        return 130
    except PipelineError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    finally:
        # Closes the progress line exactly once, on every exit path.
        progress.finish()

    for index, result in enumerate(results):
        if index:
            print()
        print(format_summary(result))
        for warning in result.warnings:
            print(f"  Warning: {warning}", file=sys.stderr)
        if result.requested_format == "dng" and result.output_path.suffix.lower() != ".dng":
            print(
                "Note: a 16-bit TIFF was written because DNG colour metadata "
                "was unavailable.",
                file=sys.stderr,
            )
    return 0


def launch_gui(
    initial_dir: Optional[str] = None,
    initial_files: Optional[Sequence[str]] = None,
    auto_start: bool = False,
    forced_mode: Optional[str] = None,
) -> int:
    """Import and start the PySide6 interface.

    Args:
        initial_dir: Folder to pre-select, used for folders dropped onto the
            executable and for ``--gui -i <folder>``.
        initial_files: Images to pre-select instead of a whole folder.
        auto_start: Press "Run Processing" once the window is up.
        forced_mode: Override the saved mode (drop always runs ``both``).
    """
    try:
        from gui import run_gui
    except ImportError as exc:
        message = (
            "The graphical interface requires PySide6.\n"
            "Install it with:  pip install PySide6\n"
            f"Details: {exc}"
        )
        LOGGER.error("%s", message)
        _show_fatal_dialog(message)
        return 1

    try:
        return run_gui(
            initial_dir=initial_dir,
            initial_files=initial_files,
            auto_start=auto_start,
            forced_mode=forced_mode,
        )
    except Exception as exc:  # pragma: no cover - last resort for frozen builds
        LOGGER.exception("GUI crashed")
        _show_fatal_dialog(f"The graphical interface failed to start:\n{exc}")
        return 1


def _emit(message: str, error: bool = False) -> None:
    """Write a line to stdout/stderr, tolerating windowed builds without one."""
    stream = sys.stderr if error else sys.stdout
    if stream is not None:
        stream.write(message + "\n")
        stream.flush()


def run_selftest() -> int:
    """Import every bundled native dependency and report what failed.

    ``--version`` exits before these imports happen; this reports a frozen build
    with a missing Qt or LibRaw DLL and names the libraries that cannot load.
    """
    checks = (
        ("NumPy", "numpy"),
        ("OpenCV", "cv2"),
        ("tifffile", "tifffile"),
        ("rawpy/LibRaw", "rawpy"),
        ("PySide6/Qt", "PySide6.QtWidgets"),
    )
    failed = False
    for label, module in checks:
        try:
            __import__(module)
        except Exception as exc:  # missing DLLs raise ImportError/OSError
            failed = True
            _emit(f"{label}: failed - {exc}", error=True)
        else:
            _emit(f"{label}: ok")
    if failed:
        _emit("selftest failed", error=True)
        return 1
    _emit("selftest ok")
    return 0


def _fail(message: str) -> int:
    """Report a start-up problem and return the failure exit code."""
    LOGGER.error("%s", message)
    _show_fatal_dialog(message)
    return 2


def _handle_dropped(paths: Sequence[str], input_dir: str) -> int:
    """Open the GUI for items dropped onto the executable.

    Windows Explorer passes every dropped item as a command line argument, so a
    drop arrives as bare paths with no option attached.  The result cannot be
    shown in a console (the standalone build is windowed), therefore the GUI
    opens, selects those items and starts the run automatically.

    A drop always runs the combined mode, so the user gets a noise-reduced
    stack *and* a super-resolution frame from a single gesture.
    """
    if input_dir:
        return _fail(
            "Pass the frames either with -i/--input-dir or as dropped items, not both."
        )

    from rawio import is_supported

    items = [Path(path).expanduser() for path in paths]
    missing = [str(item) for item in items if not item.exists()]
    if missing:
        return _fail("These dropped item(s) no longer exist:\n" + "\n".join(missing[:5]))

    folders = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    if folders and files:
        return _fail("Drop either one folder or a set of image files, not both.")
    if len(folders) > 1:
        return _fail("Drop a single folder at a time.")

    if files:
        unsupported = [item.name for item in files if not is_supported(item)]
        if unsupported:
            return _fail(
                "These dropped files are not supported image formats:\n"
                + "\n".join(unsupported[:5])
            )
        LOGGER.info("Images dropped on the executable: %d file(s)", len(files))
        return launch_gui(
            initial_dir=str(sorted(files)[0].parent),
            initial_files=[str(item) for item in files],
            auto_start=True,
            forced_mode="both",
        )

    LOGGER.info("Folder dropped on the executable: %s", folders[0])
    return launch_gui(
        initial_dir=str(folders[0]), auto_start=True, forced_mode="both"
    )


def _show_fatal_dialog(message: str) -> None:
    """Best effort error dialog for windowed (console-less) builds."""
    if sys.platform != "win32":
        print(message, file=sys.stderr)
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
    except Exception:  # pragma: no cover
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Route between GUI (no arguments) and CLI (arguments present)."""
    multiprocessing.freeze_support()
    arguments = list(sys.argv[1:] if argv is None else argv)

    if not arguments:
        configure_logging(verbose=0, quiet=False, log_file="")
        return launch_gui()

    parser = build_parser()
    args = parser.parse_args(arguments)
    configure_logging(args.verbose, args.quiet, args.log_file)
    if args.selftest:
        return run_selftest()
    if args.paths:
        return _handle_dropped(args.paths, args.input_dir)
    if args.gui:
        return launch_gui(initial_dir=args.input_dir or None)
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
