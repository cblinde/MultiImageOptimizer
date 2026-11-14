# Multi-Image Optimizer

Multi-frame RAW super-resolution and noise reduction for 360° and standard
cameras. The engine aligns a folder of DNG (or TIFF/PNG/JPEG) frames with
sub-pixel precision and combines them into 16-bit images: a noise-reduced
stack, a drizzle-reconstructed higher-resolution frame, or both at once.

It runs headless from the command line or as a desktop GUI when started without
arguments, and can be packaged into a single `MultiImageOptimizer.exe`.

## Features

- **Pipeline A — Noise reduction:** ORB + RANSAC registration refined by ECC
  (sub-pixel), followed by a weighted **mean** (optimal for Gaussian sensor
  noise) or **median** (rejects hot pixels and moving objects).
- **Pipeline B — Super-resolution:** the same registration, then a
  flux-conserving **drizzle** accumulation on a grid 2–6× finer. Sparse drizzle
  coverage is blended with a Lanczos upscale of the native-resolution mean.
- **16-bit end to end:** frames are decoded linearly (`gamma=(1,1)`, no auto
  brightening, no white balance), accumulated in float32 and written as 16-bit
  DNG, 16-bit TIFF or 16-bit PNG.
- **Robust input:** DNG/CR2/NEF/ARW/RAF/ORF/... via LibRaw, plus TIFF/PNG/JPEG.
  Corrupt frames and frames with too few matches are skipped with a clear
  message instead of aborting the run. Mixed folders work too: the format is
  detected per file.
- **Combined mode:** `--mode both` (and every drop) produces the stacked image
  *and* the super-resolution frame, each named after the method that made it.
  It is one pass over the frames: they are registered, decoded and warped once
  and both results accumulate from that pass, so neither the alignment nor the
  decoding/IO runs twice.
- **Drop and go:** drag a folder or a multi-selection of images onto the
  executable and both results are produced immediately; no console needed.
- **Built-in help:** press **Help** (or F1) for what the program does, how the
  modes differ and the full CLI reference.
- **Results next to the sources:** the default output path is inside the input
  folder, and can be changed freely. Optionally every registered frame can be
  exported separately for inspection.

## Install

Requires Python 3.12, the interpreter CI and the build kit use.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r src/requirements.txt
```

On Windows, `msvc-runtime` supplies `MSVCP140.dll`/`VCRUNTIME140.dll` inside the
virtual environment, so rawpy works without installing the Visual C++
redistributable system-wide. The application registers that directory with the
Windows loader automatically.

## GUI

```bash
python src/main.py
```

or double-click `MultiImageOptimizer.exe`. Choose the input — a folder with
**Browse…**, individual images with **Select files…**, or just drop them on the
window — pick mode, stacking method, output format and (optionally) an output
file, then press **Run Processing**. Processing happens in a `QThread`, so the
window stays responsive; progress and the per-frame alignment log are shown
live, and a running job can be cancelled. **Help** (or F1) explains the engine
and every option.

The mode selector offers *Noise Reduction*, *Super-Resolution* and *Both*; it
enables the drizzle controls whenever super-resolution is involved.
Output defaults to `<input>/<folder>_stack_<tag>.dng|tif|png`, i.e. right next
to your source frames; pick another path with **Browse** or by typing one. The
`<tag>` names the method (`noise_mean`, `noise_median`, `sr2x`), so the combined
mode writes two files that cannot be confused. Files with that `_stack_` marker
are never read back as input, so a folder can be processed repeatedly without
stacking an earlier result.

### Drag & drop

Drop a folder — or a multi-selection of image files — onto
`MultiImageOptimizer.exe` in Explorer: the GUI opens, applies that input and
**immediately produces both results** (noise-reduced stack and
super-resolution frame) with the remaining settings from the last session.
Dragging onto the open window only fills in the input, so you can still adjust
the options before pressing **Run Processing**.

This works because Windows passes every dropped item as a plain command line
argument. A drop always selects the combined mode, so one gesture yields both
images. Mixing folders and files in one drop, dropping several folders, or
combining a drop with `-i` is rejected with a message. Dropping onto a shortcut
works too, as long as it points at the executable itself.

### Aligned frames export

Tick **Export aligned frames** (or pass `--save-aligned`) to keep every frame
after registration: each one is warped onto the reference frame and written to
`<output>_aligned/` in the same format as the result, numbered in processing
order (`01_IMG_0042.dng`). Nothing is computed twice — the engine hands the
already warped frame to the writer. In combined mode the export happens once
and both results point at the same folder, since both work from those frames.

This is the quickest way to judge whether the registration worked: in a
timeline or viewer the exported frames should be pixel-identical, and flicking
through them shows exactly what moved. The fold is skipped during recursive
discovery, so it never contaminates a later run. Cost: one full-size file per
frame on disk, so on a 20-frame 50 MP stack expect a few gigabytes.

## CLI

Any argument launches headless mode:

```bash
# Noise reduction, median stack, 16-bit DNG output
python src/main.py -i "D:\shoots\room" --mode noise_reduction --stack-method median \
    --output-format dng -o "D:\shoots\room\clean.dng"

# 2x drizzle super-resolution from a 12-frame burst
python src/main.py -i "D:\shoots\room" --mode super_resolution --upscale 2 \
    --pixfrac 0.75 --output-format tiff -o "D:\shoots\room\room_2x.tif"

# Same stack as a 16-bit PNG for viewing/sharing, from JPEG or PNG sources too
python src/main.py -i "D:\shoots\room" --stack-method mean \
    --output-format png -o "D:\shoots\room\reduced.png"

# Both results in one run: stack plus 2x drizzle frame
python src/main.py -i "D:\shoots\room" --mode both --output-format png

# Default output next to the sources, plus the aligned frames for inspection
python src/main.py -i "D:\shoots\room" --save-aligned

# Explicit files instead of a folder (opens the GUI and starts both pipelines)
python src/main.py D:\shoot\IMG_0001.dng D:\shoot\IMG_0002.dng D:\shoot\IMG_0003.dng
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-i, --input-dir` | — | Folder containing the frames (required unless files are given). |
| `FOLDER`, `FILE...` | — | Bare paths, as passed by drag & drop: opens the GUI, selects them and starts both pipelines. |
| `-o, --output` | `<input>/<folder>_stack_<tag>.<ext>` | Output file path; with `--mode both` the tag is appended to the stem. |
| `--mode` | `noise_reduction` | `noise_reduction`, `super_resolution` or `both`. |
| `--stack-method` | `mean` | `mean` or `median` (noise reduction only). |
| `--output-format` | `dng` | `dng`, `tiff` or `png`; DNG falls back to TIFF when it is not possible. |
| `--upscale` | `2` | Linear resolution multiplier, 2–6. |
| `--pixfrac` | `0.75` | Drizzle drop size as a fraction of an input pixel. |
| `--reference` | sharpest | 1-based frame index used as the alignment reference. |
| `--max-dim` | `2000` | Longest edge of the registration preview. |
| `--recursive` | off | Search sub-folders, skipping `output/`, `_*` and `*_aligned` folders. |
| `--save-aligned` | off | Also write every aligned frame to `<output>_aligned/`. |
| `--aligned-dir` | — | Folder for those frames; implies `--save-aligned`. |
| `--limit N` | all | Only use the first N frames (quick tests). |
| `--temp-dir` | system temp | Scratch location for the median stack. |
| `--gui` | off | Force the GUI even when other arguments are present. |
| `--selftest` | off | Load the bundled native libraries (Qt, LibRaw, OpenCV) and exit; for diagnostics. |
| `-v / -q` | info | More/less logging. `--log-file` appends to a file. |

The exit code is `0` on success, `1` on a processing error, `2` for an invalid
start-up or drag-and-drop input and `130` when cancelled.

## How it works

1. **Discovery:** supported files are collected in natural sort order
   (`IMG_2` before `IMG_10`), either from a folder or from an explicit file
   selection such as a drag & drop of individual images.
2. **Previews:** each frame is demosaiced linearly (RAW at half size for speed)
   and converted to a percentile-stretched 8-bit luminance preview. The
   sharpest frame (variance of Laplacian) becomes the reference.
3. **Registration:** ORB keypoints with a Hamming ratio test, `estimateAffinePartial2D`
   with RANSAC, then ECC refinement for sub-pixel translation/rotation. A
   phase-correlation fallback handles textureless or strongly displaced frames.
   Implausible transforms (scale ≠ 1, >25° rotation, >35 % diagonal shift) are
   rejected.
4. **Combination:** frames are warped with bicubic interpolation, weighted by
   their valid coverage, then averaged/median-combined, or drizzled. The median
   path stages the aligned stack in a memory-mapped scratch file so large
   stacks do not exhaust RAM.
5. **Output:** float32 linear data is quantised to 16 bits. Camera RGB is
   converted to sRGB using the DNG colour matrix and `AsShotNeutral` for TIFF
   (linear) and PNG (sRGB encoded); DNG output remains camera RGB with the
   original colour tags. The filename carries the method: `noise_mean`,
   `noise_median`, `sr2x`, `sr3x`, … Combined mode writes one file per method.

## Output notes

- **DNG:** written as a 3-channel `LinearRaw` DNG (demosaiced), copying the
  source colour matrix, neutral and camera model. The file is verified by
  reopening it with LibRaw. If the input has no colour matrix (for example a
  plain TIFF), a valid DNG cannot be built and a 16-bit TIFF is written instead
  with a warning.
- **PNG:** 16-bit RGB, sRGB encoded, so it opens correctly in any viewer,
  editor or browser without a gamma surprise. Best choice for handing a result
  to someone else; it carries no DNG colour metadata.
- **TIFF:** 16-bit RGB, zlib-compressed, tagged as linear sRGB. Open it in
  software that respects linear data, or apply a display transform.
- A DNG or PNG produced by this tool can be used as an input frame again
after renaming it so the name no longer contains the `_stack_` marker;
discovery deliberately skips tool outputs.
- Reading is format aware: RAW and 16-bit TIFF samples are linear and used
  as-is, while PNG/JPEG/BMP and 8-bit TIFF are treated as gamma-encoded sRGB and
  linearised on load. That keeps stacking in linear light either way, at the
  cost of the 8-bit precision of JPEG sources.

## Build the standalone EXE

```bash
pip install "pyinstaller>=6.10,<7"
pyinstaller --noconfirm src/build.spec
```

On Windows without a prepared Python, double-click `BUILD.bat` instead: it
provisions a portable interpreter inside the folder and runs the same spec (see
*Building the exe locally* under Releases).

The result is `dist/MultiImageOptimizer.exe` (single file, no console window).
The spec bundles LibRaw (`raw_r.dll`), the OpenMP runtime and the MSVC runtime
DLLs, so the target machine needs no Python installation. Start it without
arguments for the GUI, pass CLI arguments as usual, or drop a folder/selection
of images on it to run both pipelines at once.

## Tests

```bash
python -m unittest discover -s tests -v
```

The synthetic suite verifies sub-pixel translation recovery, rejection of
featureless frames, mean-stack noise reduction, median outlier rejection,
drizzle resolution gain over bicubic upscaling, 16-bit TIFF/PNG/DNG round-trips,
gamma handling of 8-bit sources, the end-to-end pipeline (including its error
paths), the combined mode, explicit file selections, the aligned-frame export and
the drag-and-drop entry point routing.

## Project layout

```
BUILD.bat      one-click Windows build (portable Python, no setup)
README.md      this guide
LICENSE        project licence (MIT)
THIRD_PARTY_NOTICES.md  licences of every bundled component
src/           application source and build recipe
               main.py, gui.py, pipeline.py, aligner.py, stacker.py, rawio.py,
               build.spec, build.ps1, requirements.txt
tests/         synthetic verification suite
.github/       GitHub Actions: Build (push/PR) and Release workflows
```

## Troubleshooting

- **`ImportError: DLL load failed while importing _rawpy`** — the MSVC runtime
  is missing. `pip install msvc-runtime` (already in `src/requirements.txt`) or
  install the Visual C++ 2015–2022 redistributable.
- **Which bundled library failed?** — run `MultiImageOptimizer.exe --selftest`.
  It imports NumPy, OpenCV, tifffile, rawpy and PySide6 inside the bundled build
  and names every library that cannot load.
- **"Could not register at least two frames"** — the frames do not overlap,
  the scene is textureless (sky, blank wall) or motion exceeds 35 % of the
  diagonal. Capture with overlapping detail or use a different reference.
- **Super-resolution looks soft** — drizzle needs genuine sub-pixel dithering.
  Frames taken from a fixed tripod without sensor shift contain no new
  information; the result then equals a good upscale. Try `--pixfrac 0.6`.
- **Median stack is slow / disk-heavy** — it writes one scratch file of
  `frames × width × height × 3 × 2` bytes. Point `--temp-dir` at a fast disk.
- **Processing is memory-hungry** — mean stacking keeps two float32 buffers at
  native resolution; super-resolution adds the high-resolution accumulators
  (about 16 bytes per output pixel).

## Workflows

Two workflows live in `.github/workflows`:

| Workflow | Trigger | Jobs |
| --- | --- | --- |
| **Build** (`build.yml`) | push to `main`, pull requests, and `workflow_call` from Release | `test`, `build`, `build-kit` |
| **Release** (`release.yml`) | push to `main`, tag `v*.*.*`, or **Actions → Release → Run workflow** | `prepare` decides, calls Build, then publishes |

- **`test`** byte-compiles `src` and `tests` and runs the unittest suite on
  `windows-latest`.
- **`build`** builds `MultiImageOptimizer.exe` with PyInstaller and smoke-tests
  `--version`, the bundled native libraries, a synthetic DNG stack and an
  offscreen GUI start. The executable is uploaded as the **`windows-x64`**
  artifact, downloadable from the workflow run under **Artifacts**.
- **`build-kit`** creates `MultiImageOptimizer-build-kit.zip`, extracts it and
  runs `BUILD.bat --no-pause` exactly like a user, including the portable Python
  download. It checks the resulting exe, licence files and zip, then runs the
  kit-built exe with `--version` and `--selftest`. Pull requests run this
  duplicate Windows build only when they touch `BUILD.bat`, `src/build.ps1`,
  `src/build.spec`, `src/requirements.txt` or
  `.github/scripts/make_build_kit.py`; pushes to `main` and every Release always
  run it. On failure it prints the last 150 lines of the
  kit's `build-log.txt` and uploads that file as the **`kit-build-log`** artifact
  for the run.

On a push to `main`, `prepare` reads `APP_VERSION` and the release only runs
when `v<APP_VERSION>` does not exist yet; otherwise the run ends after that
check and no Windows build is paid for. Tag pushes and manual runs always
release. Build and publish happen in the same run, because a tag created with
`GITHUB_TOKEN` would not trigger a second workflow.

The Release workflow calls Build first, so a failing test, exe or kit blocks
publishing. It then checks the tag against `APP_VERSION`, uploads the exe, both
zips, the licence files and `SHA256SUMS` to a draft release, and publishes it
only after every upload succeeded; the assets then appear under **Assets**.

## Releases

### Cutting a release

1. Set `APP_VERSION` in `src/main.py` to the release version and commit it. That is
   the only definition: the GUI imports it, so `--version` and the window title
   always agree. The tag must be `v` + `APP_VERSION` (for example `v1.0.1` for
   `1.0.1`); the release job compares the two and fails instead of publishing a
   mismatched executable.
2. Push that commit to `main`. If `v<APP_VERSION>` does not exist yet, the
   Release workflow builds and publishes it automatically; later pushes with
   the same version do nothing. To release explicitly, push the tag instead:

   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

   Alternatively, run **Actions → Release → Run workflow** and enter the tag;
   it is created from the selected commit if it does not exist yet.
3. The same run calls the reusable Build workflow on `windows-latest`: the test
   suite, the PyInstaller executable with its smoke tests (`--version`, the
   bundled-library selftest, a headless DNG stack and an offscreen GUI start),
   and the extracted build kit run through `BUILD.bat`. Any failure blocks the
   release, and a tag created here does not start a second run.
4. An existing tag must point at the commit being built; if it points at any
   other commit, the run fails before anything is uploaded. A missing tag is
   created at the built commit.
5. Assets are uploaded to a **draft** release, and it is published only after
   every upload succeeded. From **Releases → Multi-Image Optimizer v1.0.0 →
   Assets** download `MultiImageOptimizer.exe` on its own, or the
   `MultiImageOptimizer-windows-x64.zip` archive, which also contains `LICENSE`
   and `THIRD_PARTY_NOTICES.md`. `MultiImageOptimizer-build-kit.zip` rebuilds
   the exe locally, and `SHA256SUMS` covers every asset.

If an automatic run fails after its tag was created, rerun it with
**Actions → Release → Run workflow** for that tag; later pushes with the same
`APP_VERSION` do nothing because the tag already exists.

### Building the exe locally (Windows)

Releases also ship `MultiImageOptimizer-build-kit.zip`, which contains the
source, `src/build.spec`, the licence files and the build scripts. To rebuild the
executable on your own machine:

1. Extract the archive to a short, writable path such as `C:\mio-build`.
2. Double-click `BUILD.bat` and wait. For unattended runs (scripts or CI),
   `BUILD.bat --no-pause` skips the final key press.

CI extracts this exact archive on a Windows runner and executes `BUILD.bat`
with a timeout, so a broken kit fails before a release is published.

The first run downloads the official portable Python 3.12 into `.python` and
the build dependencies into it at `src/requirements.txt` versions, so there is no
system Python install, no registry change, no admin prompt and no need for
developer tools. It then runs PyInstaller and leaves you with:

- `out\MultiImageOptimizer.exe` - the self-contained executable, with
  `LICENSE` and `THIRD_PARTY_NOTICES.md` beside it
- `MultiImageOptimizer-windows-x64.zip` - those three files zipped
- `build-log.txt` - the complete output, for when something fails

A 64-bit Windows 10/11 machine and an internet connection are required on the
first run (roughly 200 MB of downloads); later runs reuse `.python` and only
rebuild. Delete `.python`, `.downloads`, `build`, `dist` and `out` to start
from scratch.

## Limitations

- This project only combines existing frames: it does not synthesise or generate
  new image content, and no image-generation component is part of the source
  tree or the build.
- Output is demosaiced RGB, not a mosaic; it is not a drop-in replacement for
  the camera's original raw file.
- Alignment models translation, rotation and small scale changes (partial
  affine + Euclidean ECC). Strong perspective changes are not corrected.
- Very large super-resolution outputs (over 250 megapixels) are refused to
  avoid exhausting memory.

## License

MultiImageOptimizer is released under the MIT License; see [LICENSE](LICENSE).
Its GUI uses Qt for Python (PySide6) and LibRaw under the LGPL, alongside other
permissively licensed libraries; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
lists every bundled component and reproduces the LGPL-3.0, GPL-3.0, LGPL-2.1,
Apache-2.0 and GPL-2.0 texts plus the BSD and MIT licences. Release archives
ship both files next to the executable.
