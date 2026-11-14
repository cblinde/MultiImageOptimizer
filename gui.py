"""PySide6 front-end for the multi-frame RAW optimizer.

Heavy work always happens in :class:`ProcessingWorker`, a ``QThread`` that
calls :func:`pipeline.run_pipeline` and reports back through signals.  The
widget tree therefore never blocks and the progress bar stays animated.
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path
from string import Template
from typing import Optional

from PySide6.QtCore import (
    QPointF,
    QSettings,
    QTemporaryDir,
    QThread,
    QTimer,
    Qt,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QKeySequence,
    QPainter,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import rawio
from pipeline import (
    METHOD_LABELS,
    MODES,
    MODE_LABELS,
    PipelineCancelled,
    PipelineConfig,
    PipelineError,
    RunResult,
    resolve_output_path,
    run_pipelines,
)

APP_NAME = "Multi-Image Optimizer"
APP_VERSION = "1.0.0"

#: Output format key -> suffix and file dialog filter.
FORMAT_SUFFIXES = {"dng": ".dng", "tiff": ".tif", "png": ".png"}
FORMAT_FILTERS = {
    "dng": "DNG image (*.dng)",
    "tiff": "TIFF image (*.tif *.tiff)",
    "png": "PNG image (*.png)",
}

_ARROW_PATHS: dict = {}
_ARROW_DIR: Optional[QTemporaryDir] = None


def _arrow_paths() -> tuple:
    """Render the small arrows and return their file paths.

    Qt style sheets can only draw sub-control arrows from an image, so the
    glyphs are painted once at start-up instead of shipping binary assets
    (which also keeps the PyInstaller bundle free of data files).  The
    temporary directory is owned by a module-level object so the files
    outlive the style sheet and are cleaned up when the process exits.
    """
    global _ARROW_DIR
    if not _ARROW_PATHS:
        _ARROW_DIR = QTemporaryDir()
        directory = Path(_ARROW_DIR.path())
        glyphs = {
            "up": ((4.0, 6.0), (12.0, 6.0), (8.0, 1.5)),
            "down": ((4.0, 1.5), (12.0, 1.5), (8.0, 6.0)),
        }
        for name, points in glyphs.items():
            pixmap = QPixmap(16, 8)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#475569"))
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
            painter.end()
            path = directory / f"arrow-{name}.png"
            pixmap.save(str(path))
            _ARROW_PATHS[name] = path.as_posix()
    return _ARROW_PATHS["up"], _ARROW_PATHS["down"]


_STYLESHEET = Template("""
QWidget {
    background: #f5f7fb;
    color: #1f2937;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QScrollArea { background: #f5f7fb; border: none; }
QLabel#title { font-size: 22px; font-weight: 700; color: #0f172a; }
QLabel#subtitle { color: #64748b; }
QLabel#hint { color: #64748b; font-size: 12px; }
QLabel#status { color: #334155; }
QGroupBox {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    margin-top: 16px;
    padding: 14px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #475569;
}
/* min-height keeps the layout from squashing these fields into slivers when
   the window gets small: the scroll area takes over instead. */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 7px 9px;
    min-height: 20px;
    selection-background-color: #bfdbfe;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #2563eb;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #64748b;
    background: #f1f5f9;
    border-color: #e2e8f0;
}
QComboBox { padding-right: 30px; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 26px;
    border: none;
}
QComboBox::down-arrow {
    image: url("$down_arrow");
    width: 9px;
    height: 6px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: #dbeafe;
    selection-color: #1e3a8a;
}
QSpinBox, QDoubleSpinBox { padding-right: 30px; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 24px;
    margin: 2px 2px 0 0;
    background: #f1f5f9;
    border-left: 1px solid #e2e8f0;
    border-top-right-radius: 7px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 24px;
    margin: 0 2px 2px 0;
    background: #f1f5f9;
    border-left: 1px solid #e2e8f0;
    border-bottom-right-radius: 7px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #e2e8f0;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("$up_arrow");
    width: 9px;
    height: 6px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("$down_arrow");
    width: 9px;
    height: 6px;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 7px 14px;
}
QPushButton:hover { background: #eef2f7; }
QPushButton:disabled { color: #94a3b8; background: #f1f5f9; border-color: #e2e8f0; }
QPushButton#runButton {
    background: #2563eb;
    color: #ffffff;
    border: none;
    font-weight: 600;
    padding: 10px 24px;
}
QPushButton#runButton:hover { background: #1d4ed8; }
QPushButton#runButton:disabled { background: #93b4f5; color: #eef2ff; }
QPushButton#cancelButton { padding: 9px 18px; }
QProgressBar {
    border: 1px solid #e2e8f0;
    border-radius: 7px;
    background: #eef2f7;
    height: 16px;
    text-align: center;
    font-size: 11px;
    color: #334155;
}
QProgressBar::chunk { background: #2563eb; border-radius: 6px; }
QPlainTextEdit {
    background: #0f172a;
    color: #d6e2f0;
    border: none;
    border-radius: 8px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
    padding: 8px;
}
QCheckBox { spacing: 6px; }
""")


def build_stylesheet() -> str:
    """Return the application style sheet with the generated arrow glyphs."""
    up_arrow, down_arrow = _arrow_paths()
    return _STYLESHEET.substitute(up_arrow=up_arrow, down_arrow=down_arrow)


class ProcessingWorker(QThread):
    """Background worker executing one pipeline run."""

    progress = Signal(int, str)
    log = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config: PipelineConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Ask the pipeline to stop after the current frame."""
        self._cancel_event.set()

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            results = run_pipelines(
                self._config,
                progress=self._on_progress,
                log=self.log.emit,
                cancel=self._cancel_event.is_set,
            )
        except PipelineCancelled:
            self.cancelled.emit()
        except PipelineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # defensive: never lose the traceback
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.completed.emit(results)

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.emit(int(round(fraction * 100.0)), message)


HELP_HTML = """
<h2>What Multi-Image Optimizer does</h2>
<p>It takes a burst of frames of the same scene, finds how each frame is
shifted or rotated relative to a reference frame, warps it back into place and
combines the frames. Two independent results are available:</p>
<ul>
  <li><b>Noise reduction (stacking)</b> — averaging or median-combining aligned
      frames cancels random sensor noise. Mean is optimal for Gaussian noise and
      keeps the lowest noise floor; median rejects hot pixels, sensor dust and
      moving objects at the cost of a little detail.</li>
  <li><b>Super-resolution (drizzle)</b> — the sub-pixel differences between
      frames are accumulated on a grid 2–6× finer, recovering real detail
      instead of interpolating it.</li>
  <li><b>Both</b> — produces the stack and the super-resolution frame in one
      pass over the frames and writes two files, so one drop or one click yields
      both results. Each frame is registered, decoded and warped only once.</li>
</ul>
<p>Everything is processed linearly in float32 and written as 16 bit data, so a
16 bit source keeps its 16 bit precision.</p>

<h2>Inputs</h2>
<ul>
  <li><b>Folder</b> — every supported frame inside it. Enable <i>Include
      sub-folders</i> to search recursively; <code>output</code>, <code>_*</code>
      and <code>*_aligned</code> folders are skipped.</li>
  <li><b>Selected files</b> — use <i>Select files…</i> to pick individual frames.
      The selection wins over the folder and is what drag &amp; drop of images
      produces.</li>
  <li><b>Formats</b>: DNG, CR2, CR3, NEF, NRW, ARW, SRF, SR2, RAF, RW2, ORF, PEF,
      SRW, RAW, 3FR, ERF, MRW, X3F, plus TIFF, PNG, JPEG and BMP. Mixed folders
      are fine — the format is detected per file.</li>
</ul>
<p>RAW and 16 bit TIFF samples are linear and used as-is. PNG, JPEG, BMP and
8 bit TIFF are display referred, so their sRGB curve is removed on load.</p>

<h2>Aligning</h2>
<p>ORB keypoints with a ratio test, RANSAC for the gross transform, then ECC
refinement for sub-pixel accuracy, with a phase-correlation fallback. Frames
that cannot be registered are skipped and reported in the log; at least two
frames must remain.</p>

<h2>Outputs</h2>
<p>Results are written next to the sources unless you set a path:
<code>&lt;folder&gt;_stack_&lt;tag&gt;.&lt;ext&gt;</code> with the tag naming the
applied method: <code>noise_mean</code>, <code>noise_median</code> or
<code>sr2x</code> (the drizzle factor is part of the name because it changes the
pixel count). Files carrying that <code>_stack_</code> marker are never read back
as input, so a folder can be processed again and again.</p>
<ul>
  <li><b>16-bit DNG</b> — 3-channel LinearRaw with the source colour tags; falls
      back to TIFF with a warning when the sources carry no colour matrix.</li>
  <li><b>16-bit TIFF</b> — linear sRGB, zlib compressed; for software that
      respects linear data.</li>
  <li><b>16-bit PNG</b> — sRGB encoded, opens correctly anywhere; best for
      sharing.</li>
</ul>
<p><i>Export aligned frames</i> adds a<br><code>&lt;output&gt;_aligned/</code> folder
with every registered frame, numbered in processing order, in the same format.
In the combined mode that folder is written once and belongs to both results.
Flick through them to judge whether the registration is right.</p>

<h2>Drag &amp; drop</h2>
<ul>
  <li>Drop a <b>folder or several images onto the .exe</b>: the window opens, the
      selection is applied and <b>both</b> results are produced immediately.</li>
  <li>Drop onto the <b>open window</b>: the folder or file selection is filled
      in, and you still choose the mode and press <i>Run Processing</i>.</li>
</ul>

<h2>Command line</h2>
<p>Any option runs the program headless. A bare folder or file list opens the GUI
and starts at once — that is what a drop does.</p>
<pre>
main.py -i DIR --mode both --output-format png
main.py -i DIR --mode super_resolution --upscale 2 --pixfrac 0.75 -o out.dng
main.py -i DIR --stack-method median --save-aligned
main.py FILE1.dng FILE2.dng FILE3.dng        # selection, opens the GUI
</pre>
<table cellpadding="4" cellspacing="0">
<tr><td><code>-i, --input-dir</code></td><td>Folder with the frames.</td></tr>
<tr><td><code>-o, --output</code></td><td>Output file; defaults next to the sources.</td></tr>
<tr><td><code>--mode</code></td><td><code>noise_reduction</code>,
    <code>super_resolution</code> or <code>both</code>.</td></tr>
<tr><td><code>--stack-method</code></td><td><code>mean</code> or <code>median</code>.</td></tr>
<tr><td><code>--output-format</code></td><td><code>dng</code>, <code>tiff</code>
    or <code>png</code>.</td></tr>
<tr><td><code>--upscale</code></td><td>Drizzle factor, 2–6.</td></tr>
<tr><td><code>--pixfrac</code></td><td>Drizzle drop size, 0 &lt; pixfrac ≤ 1.</td></tr>
<tr><td><code>--reference N</code></td><td>1-based reference frame; defaults to
    the sharpest.</td></tr>
<tr><td><code>--recursive</code></td><td>Search sub-folders.</td></tr>
<tr><td><code>--save-aligned</code></td><td>Also write the aligned frames.</td></tr>
<tr><td><code>--aligned-dir DIR</code></td><td>Where those frames go; implies the
    flag above.</td></tr>
<tr><td><code>--limit N</code></td><td>Only use the first N frames.</td></tr>
<tr><td><code>--max-dim N</code></td><td>Longest edge of the registration
    preview (default 2000).</td></tr>
<tr><td><code>-v / -q</code></td><td>More / less logging.
    <code>--log-file FILE</code> appends to a file.</td></tr>
<tr><td><code>--gui</code></td><td>Open the GUI even with other options.</td></tr>
<tr><td><code>--selftest</code></td><td>Load the bundled libraries and exit; for
    diagnostics.</td></tr>
</table>
<p>Exit code 0 means success, 1 a processing error, 2 invalid start-up input and
130 cancellation.</p>

<h2>Good to know</h2>
<ul>
  <li>Super-resolution needs genuine sub-pixel movement. A burst from a fixed
tripod without sensor shift contains no new information; the result then equals
a good upscale.</li>
  <li>Median stacking stages one scratch file of
<code>frames × width × height × 3 × 2</code> bytes; point the temporary
folder at a fast disk if it is slow.</li>
  <li>Memory use is roughly two float32 buffers at native resolution, plus the
high-resolution accumulators (about 16 bytes per output pixel) for drizzle.</li>
  <li>The registration preview runs at half resolution for RAW input purely for
speed; the final stack always uses full resolution.</li>
</ul>
"""


class HelpDialog(QDialog):
    """Scrollable documentation for the GUI and the command line."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} — Help")
        self.resize(780, 700)

        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(HELP_HTML)
        layout.addWidget(browser)

        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(
        self,
        initial_dir: Optional[str] = None,
        initial_files: Optional[list] = None,
        auto_start: bool = False,
        forced_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._settings = QSettings("MultiImageOptimizer", "MultiImageOptimizer")
        self._worker: Optional[ProcessingWorker] = None
        self._last_output: Optional[Path] = None
        #: Explicitly picked or dropped images; takes precedence over the folder.
        self._selected_files: list[Path] = []

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setMinimumSize(720, 520)
        self.setAcceptDrops(True)

        self._build_ui()
        # PySide6 does not accept signal names as constructor keywords.
        self._help_shortcut = QShortcut(QKeySequence("F1"), self)
        self._help_shortcut.activated.connect(self._show_help)
        self._apply_initial_geometry()
        self._load_settings()
        if forced_mode:
            index = self.mode_combo.findData(forced_mode)
            if index >= 0:
                self.mode_combo.setCurrentIndex(index)
        if initial_files:
            self._set_selection([Path(path) for path in initial_files])
        elif initial_dir:
            self.input_edit.setText(str(initial_dir))
        self._refresh_frames()
        if auto_start:
            # Let Qt paint the window before the worker thread starts.
            QTimer.singleShot(250, self._auto_start)

    # ----------------------------------------------------------------- UI ---
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # The form has a natural minimum height of roughly 860 px.  Without a
        # scroll area a smaller window forces Qt to squeeze combo boxes and
        # spin boxes down to unreadable slivers, so the whole panel scrolls.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("content")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        scroll.setWidget(content)
        outer.addWidget(scroll)
        self._content = content

        title = QLabel(APP_NAME)
        title.setObjectName("title")
        subtitle = QLabel(
            "Align, stack and reconstruct multi-frame captures — "
            "16-bit linear pipeline, no quality loss."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(title)
        header.addStretch(1)
        self.help_button = QPushButton("Help")
        self.help_button.setToolTip("How the program works, and how to use the CLI (F1)")
        self.help_button.clicked.connect(self._show_help)
        header.addWidget(self.help_button)
        layout.addLayout(header)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_settings_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_log_group(), 1)

    def _apply_initial_geometry(self) -> None:
        """Open at the panel's natural size, clamped to the visible screen.

        This keeps the processing controls fully visible on small or
        high-scaling displays instead of leaving the window taller than the
        screen and letting the layout collapse the fields.
        """
        hint = self._content.sizeHint()
        width = max(hint.width(), 980)
        height = hint.height()
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, available.width() - 60)
            height = min(height, available.height() - 48)
        self.resize(
            max(width, self.minimumWidth()),
            max(height, self.minimumHeight()),
        )

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Files")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Folder with the DNG / TIFF / PNG / JPEG frames…")
        self.input_edit.textChanged.connect(self._on_input_changed)
        self.input_edit.textEdited.connect(self._on_input_typed)
        input_button = QPushButton("Browse…")
        input_button.setToolTip("Use every supported frame inside a folder")
        input_button.clicked.connect(self._browse_input)
        self.files_button = QPushButton("Select files…")
        self.files_button.setToolTip(
            "Pick individual frames, e.g. a burst spread over several folders"
        )
        self.files_button.clicked.connect(self._browse_files)

        self.output_edit = QLineEdit()
        self.output_edit.textEdited.connect(self._on_output_edited)
        output_button = QPushButton("Browse…")
        output_button.clicked.connect(self._browse_output)

        self.frames_label = QLabel("No folder or files selected")
        self.frames_label.setObjectName("hint")
        self.frames_label.setWordWrap(True)

        layout.addWidget(QLabel("Input"), 0, 0)
        layout.addWidget(self.input_edit, 0, 1)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addWidget(input_button)
        buttons.addWidget(self.files_button)
        layout.addLayout(buttons, 0, 2)
        layout.addWidget(QLabel("Output file"), 1, 0)
        layout.addWidget(self.output_edit, 1, 1)
        layout.addWidget(output_button, 1, 2)
        layout.addWidget(self.frames_label, 2, 0, 1, 3)
        layout.setColumnStretch(1, 1)
        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("Processing")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)

        self.mode_combo = QComboBox()
        for mode in MODES:
            self.mode_combo.addItem(MODE_LABELS[mode], mode)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.method_combo = QComboBox()
        for method in ("mean", "median"):
            self.method_combo.addItem(METHOD_LABELS[method], method)

        self.format_combo = QComboBox()
        self.format_combo.addItem("16-bit DNG", "dng")
        self.format_combo.addItem("16-bit TIFF (linear)", "tiff")
        self.format_combo.addItem("16-bit PNG (sRGB)", "png")
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)

        self.upscale_spin = QSpinBox()
        self.upscale_spin.setRange(2, 6)
        self.upscale_spin.setValue(2)
        self.upscale_spin.setSuffix("×")

        self.pixfrac_spin = QDoubleSpinBox()
        self.pixfrac_spin.setRange(0.05, 1.0)
        self.pixfrac_spin.setSingleStep(0.05)
        self.pixfrac_spin.setDecimals(2)
        self.pixfrac_spin.setValue(0.75)

        self.reference_combo = QComboBox()
        self.reference_combo.addItem("Auto (sharpest frame)", None)

        self.recursive_check = QCheckBox("Include sub-folders")
        self.aligned_check = QCheckBox("Export aligned frames")
        self.aligned_check.setToolTip(
            "Write every registered frame to <output>_aligned/ so the alignment "
            "can be checked frame by frame."
        )

        layout.addWidget(QLabel("Mode"), 0, 0)
        layout.addWidget(self.mode_combo, 0, 1)
        layout.addWidget(QLabel("Stack method"), 0, 2)
        layout.addWidget(self.method_combo, 0, 3)
        layout.addWidget(QLabel("Output format"), 1, 0)
        layout.addWidget(self.format_combo, 1, 1)
        layout.addWidget(QLabel("Upscale factor"), 1, 2)
        layout.addWidget(self.upscale_spin, 1, 3)
        layout.addWidget(QLabel("Drizzle pixfrac"), 2, 0)
        layout.addWidget(self.pixfrac_spin, 2, 1)
        layout.addWidget(QLabel("Reference frame"), 2, 2)
        layout.addWidget(self.reference_combo, 2, 3)
        layout.addWidget(self.recursive_check, 3, 0, 1, 2)
        layout.addWidget(self.aligned_check, 3, 2, 1, 2)

        hint = QLabel(
            "Mean is best for Gaussian sensor noise, median removes hot pixels and "
            "moving subjects. Super-resolution needs small sub-pixel camera shifts."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint, 4, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        row = QHBoxLayout()
        self.run_button = QPushButton("Run Processing")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._start_processing)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_processing)

        self.open_button = QPushButton("Open output folder")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_output_folder)

        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        row.addStretch(1)
        row.addWidget(self.open_button)
        layout.addLayout(row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("status")
        self._status_state = ""
        layout.addWidget(self.status_label)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Log")
        layout = QVBoxLayout(group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setFont(QFont("Cascadia Mono", 9))
        # The log should soak up leftover height instead of inflating the
        # panel's preferred size and forcing a scroll bar on every start-up.
        self.log_view.setMinimumHeight(130)
        self.log_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        layout.addWidget(self.log_view)
        return group

    # ------------------------------------------------------------ settings ---
    def _load_settings(self) -> None:
        self.input_edit.setText(str(self._settings.value("input_dir", "") or ""))
        mode = str(self._settings.value("mode", "noise_reduction"))
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(mode)))
        method = str(self._settings.value("method", "mean"))
        self.method_combo.setCurrentIndex(max(0, self.method_combo.findData(method)))
        fmt = str(self._settings.value("format", "dng"))
        self.format_combo.setCurrentIndex(max(0, self.format_combo.findData(fmt)))
        self.upscale_spin.setValue(int(self._settings.value("upscale", 2)))
        self.pixfrac_spin.setValue(float(self._settings.value("pixfrac", 0.75)))
        self.recursive_check.setChecked(
            str(self._settings.value("recursive", "false")).lower() == "true"
        )
        self.aligned_check.setChecked(
            str(self._settings.value("export_aligned", "false")).lower() == "true"
        )
        self._on_mode_changed()
        self._on_format_changed()

    def _save_settings(self) -> None:
        self._settings.setValue("input_dir", self.input_edit.text())
        self._settings.setValue("mode", self.mode_combo.currentData())
        self._settings.setValue("method", self.method_combo.currentData())
        self._settings.setValue("format", self.format_combo.currentData())
        self._settings.setValue("upscale", self.upscale_spin.value())
        self._settings.setValue("pixfrac", self.pixfrac_spin.value())
        self._settings.setValue(
            "recursive", "true" if self.recursive_check.isChecked() else "false"
        )
        self._settings.setValue(
            "export_aligned", "true" if self.aligned_check.isChecked() else "false"
        )

    # --------------------------------------------------------------- paths ---
    def _browse_input(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select the folder containing the frames", self.input_edit.text() or ""
        )
        if folder:
            self._selected_files = []
            self.input_edit.setText(folder)
            self._refresh_frames()

    def _browse_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(
            self,
            "Select the frames to combine",
            self.input_edit.text().strip(),
            "Images (*.dng *.cr2 *.cr3 *.nef *.nrw *.arw *.raf *.rw2 *.orf *.pef *.srw "
            "*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if names:
            self._set_selection([Path(name) for name in names])

    def _set_selection(self, files: list) -> None:
        """Use an explicit file list instead of every frame in a folder."""
        picked = [path for path in files if path.is_file()]
        if not picked:
            return
        self._selected_files = picked
        # The folder line still drives the default output location.
        self.input_edit.setText(str(sorted(picked)[0].parent))
        self._refresh_frames()

    def _on_input_typed(self, _text: str) -> None:
        """Typing a path means "use this folder", so a selection is dropped."""
        if self._selected_files:
            self._selected_files = []
            self._refresh_frames()

    def _browse_output(self) -> None:
        current = self.output_edit.text() or self._suggested_output()
        fmt = str(self.format_combo.currentData())
        path, _ = QFileDialog.getSaveFileName(
            self, "Choose the output file", str(current), FORMAT_FILTERS.get(fmt, "")
        )
        if path:
            self.output_edit.setText(path)

    def _suggested_output(self) -> Path:
        folder = self.input_edit.text().strip()
        if not folder:
            suffix = FORMAT_SUFFIXES.get(str(self.format_combo.currentData()), ".tif")
            return Path("output") / f"stack{suffix}"
        return rawio.default_output_path(
            folder,
            str(self.mode_combo.currentData()),
            str(self.method_combo.currentData()),
            str(self.format_combo.currentData()),
            self.upscale_spin.value(),
        )

    def _on_output_edited(self, _text: str) -> None:
        # Once the user types a path we stop overwriting it automatically.
        self.output_edit.setProperty("user_edited", True)

    def _set_auto_output(self) -> None:
        if self.output_edit.property("user_edited"):
            return
        if self.mode_combo.currentData() == "both":
            # Two files are produced, so a single suggested name would mislead.
            self.output_edit.clear()
            self.output_edit.setPlaceholderText(
                "Empty: two files next to the sources, …_stack_noise_mean/sr2x.<ext>"
            )
            return
        self.output_edit.setPlaceholderText(
            "Empty: <input>/<folder>_stack_<method>.<ext>"
        )
        if self.input_edit.text().strip():
            self.output_edit.setText(str(self._suggested_output()))
        else:
            self.output_edit.clear()

    def _on_input_changed(self) -> None:
        self._set_auto_output()
        self._refresh_frames()

    def _on_mode_changed(self) -> None:
        mode = str(self.mode_combo.currentData())
        needs_drizzle = mode in {"super_resolution", "both"}
        needs_stack = mode in {"noise_reduction", "both"}
        self.upscale_spin.setEnabled(needs_drizzle)
        self.pixfrac_spin.setEnabled(needs_drizzle)
        self.method_combo.setEnabled(needs_stack)
        self._set_auto_output()

    def _on_format_changed(self) -> None:
        if self.output_edit.property("user_edited") and self.output_edit.text().strip():
            path = Path(self.output_edit.text())
            suffix = FORMAT_SUFFIXES.get(str(self.format_combo.currentData()), ".tif")
            self.output_edit.setText(str(path.with_suffix(suffix)))
        else:
            self.output_edit.setProperty("user_edited", False)
            self._set_auto_output()

    def _frames_in_scope(self) -> list:
        """Return the frames the next run would use, in natural order."""
        if self._selected_files:
            return sorted(
                (path for path in self._selected_files if path.is_file()),
                key=lambda path: rawio.natural_key(path.name),
            )
        folder = self.input_edit.text().strip()
        if not folder:
            return []
        return rawio.discover_frames(folder, self.recursive_check.isChecked())

    def _refresh_frames(self) -> None:
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItem("Auto (sharpest frame)", None)

        try:
            frames = self._frames_in_scope()
            error = ""
        except Exception as exc:
            frames, error = [], str(exc)

        for index, frame in enumerate(frames):
            self.reference_combo.addItem(f"{index + 1}. {frame.name}", index)
        self.reference_combo.blockSignals(False)

        if self._selected_files:
            scope = f"{len(self._selected_files)} selected file(s)"
        else:
            scope = self.input_edit.text().strip()

        if error:
            self.frames_label.setText(f"Cannot read folder: {error}")
        elif not scope:
            self.frames_label.setText("No folder or files selected")
        elif len(frames) < 2:
            self.frames_label.setText(
                f"{len(frames)} supported frame(s) in {scope} — at least two are required."
            )
        else:
            self.frames_label.setText(f"{len(frames)} frames ready to process ({scope}).")

    # ------------------------------------------------------------- running ---
    def _collect_config(self) -> Optional[PipelineConfig]:
        files = [str(path) for path in self._selected_files]
        folder = self.input_edit.text().strip()
        if not folder and files:
            folder = str(Path(files[0]).parent)
        if not folder:
            QMessageBox.warning(
                self, "Missing input", "Select a folder or individual frames first."
            )
            return None
        if not Path(folder).is_dir():
            QMessageBox.warning(self, "Missing input", f"The folder does not exist:\n{folder}")
            return None

        return PipelineConfig(
            input_dir=folder,
            input_files=files,
            output_path=self.output_edit.text().strip(),
            mode=str(self.mode_combo.currentData()),
            stack_method=str(self.method_combo.currentData()),
            output_format=str(self.format_combo.currentData()),
            upscale=self.upscale_spin.value(),
            pixfrac=self.pixfrac_spin.value(),
            reference_index=self.reference_combo.currentData(),
            recursive=self.recursive_check.isChecked(),
            export_aligned=self.aligned_check.isChecked(),
        )

    def _auto_start(self) -> None:
        """Start processing a folder that was dropped onto the executable.

        The run is only started when the folder really holds a stack; otherwise
        the status label explains why nothing happened instead of popping up a
        dialog the user did not ask for.
        """
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            frames = self._frames_in_scope()
        except Exception:
            frames = []
        if len(frames) < 2:
            message = (
                f"Automatic start skipped: {len(frames)} supported frame(s) found "
                "in the dropped item (at least two are required)."
            )
            self._append_log(message)
            self._set_status(message)
            return
        self._start_processing()

    def _start_processing(self) -> None:
        config = self._collect_config()
        if config is None:
            return

        self.log_view.clear()
        self._append_log(
            f"Mode: {MODE_LABELS[config.mode]} | method: {config.stack_method} | "
            f"format: {config.output_format.upper()}"
        )
        if config.input_files:
            self._append_log(
                f"Input: {len(config.input_files)} selected files in {config.input_dir}"
            )
        else:
            self._append_log(f"Input: {config.input_dir}")
        if config.mode == "both":
            self._append_log("Output: one noise-reduced stack and one super-resolution frame")
        else:
            self._append_log(f"Output: {resolve_output_path(config)}")

        self._save_settings()
        self.progress_bar.setValue(0)
        self._set_status("Starting…")
        self._set_running(True)

        self._worker = ProcessingWorker(config, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._append_log)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _cancel_processing(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._set_status("Cancelling after the current frame…")
            self.cancel_button.setEnabled(False)

    def _set_running(self, running: bool) -> None:
        for widget in (
            self.input_edit,
            self.files_button,
            self.output_edit,
            self.mode_combo,
            self.method_combo,
            self.format_combo,
            self.upscale_spin,
            self.pixfrac_spin,
            self.reference_combo,
            self.recursive_check,
            self.run_button,
        ):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        if running:
            self.open_button.setEnabled(False)

    def _set_status(self, message: str, state: str = "") -> None:
        """Update the status line; *state* colours it (``success``/``error``)."""
        self.status_label.setText(message)
        if state == self._status_state:
            return
        self._status_state = state
        colour = {"success": "#15803d", "error": "#b91c1c"}.get(state)
        # A widget style sheet always wins over the application one, whereas a
        # dynamic property selector needs a repolish that is easy to miss.
        self.status_label.setStyleSheet(
            f"color: {colour}; font-weight: 600;" if colour else ""
        )

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self._set_status(message)

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _on_completed(self, results: list) -> None:
        outputs: list = [result for result in results if isinstance(result, RunResult)]
        if not outputs:
            return
        self._last_output = outputs[-1].output_path
        self.progress_bar.setValue(100)
        self.open_button.setEnabled(True)
        self._append_log("")
        # ``elapsed`` counts from the start of the run, so two results of a
        # combined run must not be summed; the last one holds the total.
        elapsed = max(result.elapsed for result in outputs)
        for result in outputs:
            self._append_log(
                f"Done: {result.output_path} "
                f"({result.width}x{result.height}, {result.used_frames}/{result.frame_count} "
                f"frames, {result.elapsed:.1f}s)"
            )
            for warning in result.warnings:
                self._append_log(f"Warning: {warning}")

        # Success is reported in the window itself: coloured status line plus
        # the per-file lines in the log, no modal dialog to dismiss. The text
        # stays short because the label lives in a fixed width panel.
        written = "image written" if len(outputs) == 1 else "both images written"
        self._set_status(f"Finished — {written} in {elapsed:.1f} s", "success")

    def _on_failed(self, message: str) -> None:
        self.progress_bar.setValue(0)
        self._append_log(f"ERROR: {message}")
        self._set_status(f"Failed — {message}", "error")
        QMessageBox.critical(self, "Processing failed", message)

    def _on_cancelled(self) -> None:
        self._set_status("Cancelled")
        self._append_log("Processing cancelled.")
        self.progress_bar.setValue(0)

    def _on_worker_finished(self) -> None:
        self._set_running(False)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _open_output_folder(self) -> None:
        if self._last_output is None:
            return
        folder = self._last_output.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _show_help(self) -> None:
        """Open the built-in documentation."""
        HelpDialog(self).exec()

    # ---------------------------------------------------------------- misc ---
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 (Qt API)
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir() or (path.is_file() and rawio.is_supported(path)):
                event.acceptProposedAction()
                return

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 (Qt API)
        paths = [
            Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()
        ]
        folders = [path for path in paths if path.is_dir()]
        files = [path for path in paths if path.is_file() and rawio.is_supported(path)]

        if files and not folders:
            self._set_selection(files)
            event.acceptProposedAction()
            return
        if folders:
            self._selected_files = []
            self.input_edit.setText(str(folders[0]))
            self._refresh_frames()
            event.acceptProposedAction()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._worker is not None and self._worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Processing in progress",
                "Cancel the running job and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(5000)
        self._save_settings()
        super().closeEvent(event)


def run_gui(
    argv: Optional[list] = None,
    initial_dir: Optional[str] = None,
    initial_files: Optional[list] = None,
    auto_start: bool = False,
    forced_mode: Optional[str] = None,
) -> int:
    """Create the application and run the Qt event loop.

    Args:
        argv: Argument list passed to ``QApplication``.
        initial_dir: Folder to select on start (dropped folder or ``--gui -i``).
        initial_files: Images to select on start (dropped files).
        auto_start: Begin processing as soon as the window is shown.
        forced_mode: Mode to select regardless of the saved setting.
    """
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())

    window = MainWindow(
        initial_dir=initial_dir,
        initial_files=initial_files,
        auto_start=auto_start,
        forced_mode=forced_mode,
    )
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover - convenience for `python gui.py`
    raise SystemExit(run_gui())
