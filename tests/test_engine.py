"""Synthetic end-to-end checks for the alignment, stacking and RAW I/O code.

The tests build deterministic 16-bit TIFF stacks whose ground truth is known,
so the numerical behaviour of the engine can be verified without real camera
files.  Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import tifffile

import aligner
import pipeline
import rawio
import stacker


def make_scene(height: int, width: int, seed: int = 7) -> np.ndarray:
    """Return a colourful, high-frequency test chart in ``[0, 1]``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 0.35 + 0.2 * np.sin(xx / 9.0) * np.cos(yy / 13.0)
    rgb = np.stack([base, base * 0.9 + 0.05, base * 0.8 + 0.1], axis=-1)

    for _ in range(14):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(4, 18))
        colour = rng.uniform(0.05, 0.95, size=3)
        cv2.circle(rgb, centre, radius, tuple(float(c) for c in colour), -1, cv2.LINE_AA)
    for _ in range(10):
        p0 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        colour = rng.uniform(0.05, 0.95, size=3)
        cv2.line(rgb, p0, p1, tuple(float(c) for c in colour), 2, cv2.LINE_AA)

    # cv2.putText only accepts 8-bit canvases, so composite it back in.
    text = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        text, "MIO 42", (12, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (242, 230, 51), 2, cv2.LINE_AA,
    )
    mask = text.max(axis=2) > 0
    rgb[mask] = text[mask].astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate *image* by ``(dx, dy)`` pixels with cubic interpolation."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )


def write_frames(folder: Path, frames: np.ndarray) -> None:
    """Write an ``(N, H, W, 3)`` float stack as 16-bit TIFF frames."""
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        data = np.rint(np.clip(frame, 0.0, 1.0) * 65535.0).astype(np.uint16)
        tifffile.imwrite(str(folder / f"frame_{index:02d}.tif"), data, photometric="rgb")


def to_srgb8(linear: np.ndarray) -> np.ndarray:
    """Encode linear ``[0, 1]`` RGB as an 8 bit sRGB image (as a camera would)."""
    data = np.clip(linear, 0.0, 1.0).astype(np.float64)
    encoded = np.where(
        data <= 0.0031308, data * 12.92, 1.055 * np.power(data, 1.0 / 2.4) - 0.055
    )
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def register(frames: np.ndarray) -> list[np.ndarray]:
    """Register an in-memory stack against frame 0 and return its matrices."""
    previews = [aligner.prepare_gray(frame, max_dim=2000) for frame in frames]
    results = aligner.align_frames(previews, reference_index=0)
    return [result.matrix for result in results]


class AlignmentTests(unittest.TestCase):
    """Registration must recover known sub-pixel translations."""

    def test_subpixel_translation_is_recovered(self) -> None:
        scene = make_scene(360, 420)
        reference = aligner.prepare_gray(scene, max_dim=2000)

        true_shift = (1.35, -0.72)  # moving content is shifted right/up
        moving_scene = shift(scene, *true_shift)
        moving = aligner.prepare_gray(moving_scene, max_dim=2000)

        results = aligner.align_frames([reference, moving], reference_index=0)
        matrix = results[1].matrix

        # matrix maps moving -> reference, so its translation is -true_shift.
        dx, dy = matrix[0, 2], matrix[1, 2]
        self.assertLess(abs(dx + true_shift[0]), 0.15, f"dx={dx}")
        self.assertLess(abs(dy + true_shift[1]), 0.15, f"dy={dy}")
        scale, angle = aligner.decompose(matrix)
        self.assertAlmostEqual(scale, 1.0, places=3)
        self.assertAlmostEqual(angle, 0.0, places=1)

    def test_featureless_frame_is_rejected(self) -> None:
        blank = np.zeros((256, 256, 3), dtype=np.float32)
        with self.assertRaises(aligner.AlignmentError):
            aligner.align_frames([aligner.prepare_gray(blank)])


class StackingTests(unittest.TestCase):
    """Mean stacking must suppress noise, median must reject outliers."""

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="mio_test_"))
        self.addCleanup(shutil.rmtree, self.temp, True)

    def test_mean_stack_reduces_noise(self) -> None:
        scene = make_scene(240, 300)
        rng = np.random.default_rng(3)
        shifts = [(0.0, 0.0), (0.45, -0.3), (-0.6, 0.55), (0.85, 0.2),
                  (-0.25, -0.7), (0.5, 0.65), (-0.9, 0.1), (0.15, -0.45)]
        frames = []
        for dx, dy in shifts:
            noisy = shift(scene, dx, dy)
            noisy = noisy + rng.normal(0.0, 0.03, noisy.shape).astype(np.float32)
            frames.append(noisy)
        folder = self.temp / "mean"
        write_frames(folder, frames)

        paths = [str(Path(folder) / f"frame_{i:02d}.tif") for i in range(len(frames))]
        matrices = register(np.asarray(frames))
        result, stats = stacker.stack_noise_reduction(paths, matrices, method="mean")

        single_rmse = rmse(frames[0], scene)
        stacked_rmse = rmse(result, scene)
        self.assertEqual(result.shape, scene.shape)
        # Ideal averaging of 8 frames gives sqrt(8) ~ 2.8x, but cubic
        # resampling and sub-pixel interpolation add a fixed error floor, so
        # require a still substantial 1.5x improvement.
        self.assertLess(stacked_rmse, single_rmse / 1.5,
                        f"single={single_rmse:.4f} stacked={stacked_rmse:.4f}")
        self.assertAlmostEqual(stats.coverage, 1.0, places=2)

    def test_median_rejects_hot_pixels(self) -> None:
        scene = make_scene(180, 220)
        stack = np.repeat(scene[None], 5, axis=0)
        stack[2, 40:60, 70:90] = 1.0  # transient bright blob in one frame
        folder = self.temp / "median"
        write_frames(folder, stack)

        paths = [str(Path(folder) / f"frame_{i:02d}.tif") for i in range(5)]
        matrices = [np.eye(3) for _ in paths]
        result, _ = stacker.stack_noise_reduction(paths, matrices, method="median")
        # The transient blob must be gone: the pixel matches the clean scene.
        self.assertLess(abs(float(result[50, 80].mean()) - float(scene[50, 80].mean())), 0.02)
        self.assertLess(rmse(result, scene), 0.01)

    def test_drizzle_beats_bicubic_upscale(self) -> None:
        high = make_scene(420, 420, seed=11)
        rng = np.random.default_rng(5)
        offsets = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (1.5, 0.5), (0.5, 1.5)]
        frames = []
        for dx, dy in offsets:
            shifted = shift(high, dx, dy)
            down = cv2.resize(shifted, (210, 210), interpolation=cv2.INTER_AREA)
            down += rng.normal(0.0, 0.004, down.shape).astype(np.float32)
            frames.append(down)
        folder = self.temp / "sr"
        write_frames(folder, frames)

        paths = [str(Path(folder) / f"frame_{i:02d}.tif") for i in range(len(frames))]
        matrices = register(np.asarray(frames))
        sr, stats = stacker.stack_super_resolution(paths, matrices, upscale=2, pixfrac=0.8)

        base = cv2.resize(frames[0], (420, 420), interpolation=cv2.INTER_CUBIC)
        sr_error = rmse(sr, high)
        base_error = rmse(base, high)
        self.assertEqual(sr.shape, (420, 420, 3))
        self.assertEqual(sr.shape[0], stats.output_shape[0])
        self.assertLess(sr_error, base_error,
                        f"drizzle={sr_error:.4f} bicubic={base_error:.4f}")


class RawIoAndPipelineTests(unittest.TestCase):
    """DNG serialisation and the full pipeline must work end to end."""

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="mio_pipe_"))
        self.addCleanup(shutil.rmtree, self.temp, True)

    def test_tiff_roundtrip_is_16bit(self) -> None:
        scene = make_scene(120, 160)
        destination = self.temp / "out.tif"
        written = rawio.save_image(destination, scene, "tiff")
        data = tifffile.imread(str(written))
        self.assertEqual(data.dtype, np.uint16)
        self.assertEqual(data.shape, scene.shape)
        loaded = rawio.load_frame(written)
        self.assertLess(rmse(loaded, scene), 1e-4)

    def test_png_output_is_16bit_srgb(self) -> None:
        linear = np.full((32, 48, 3), 0.214, dtype=np.float32)
        written = rawio.save_image(self.temp / "display", linear, "png")
        self.assertEqual(written.suffix, ".png")

        data = cv2.imread(str(written), cv2.IMREAD_UNCHANGED)
        self.assertEqual(data.dtype, np.uint16)
        self.assertEqual(data.shape, (32, 48, 3))
        # Linear 0.214 must be gamma encoded to roughly mid grey, not written raw.
        self.assertAlmostEqual(float(data[0, 0, 0]) / 65535.0, 0.5, delta=0.01)

        # Reading it back must return the original linear values.
        self.assertLess(rmse(rawio.load_frame(written), linear), 1e-3)

    def test_eight_bit_sources_are_linearised(self) -> None:
        srgb = np.full((16, 16, 3), 128, dtype=np.uint8)
        png = self.temp / "mid.png"
        jpeg = self.temp / "mid.jpg"
        cv2.imwrite(str(png), srgb[..., ::-1])
        cv2.imwrite(str(jpeg), srgb[..., ::-1])

        expected = ((128 / 255 + 0.055) / 1.055) ** 2.4
        for path in (png, jpeg):
            loaded = rawio.load_frame(path)
            self.assertEqual(loaded.shape, (16, 16, 3))
            self.assertAlmostEqual(float(loaded.mean()), expected, delta=1e-3)

    def test_pipeline_accepts_png_sources_and_writes_png(self) -> None:
        scene = make_scene(160, 200)
        folder = self.temp / "raster"
        folder.mkdir(parents=True, exist_ok=True)
        shifts = [(0.0, 0.0), (0.6, -0.3), (-0.5, 0.4), (0.8, 0.7)]
        for index, (dx, dy) in enumerate(shifts):
            frame = to_srgb8(shift(scene, dx, dy))
            cv2.imwrite(str(folder / f"shot_{index:02d}.png"), frame[..., ::-1])

        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            output_path=str(self.temp / "raster_out.png"),
            mode="noise_reduction",
            stack_method="mean",
            output_format="png",
            max_dim=800,
        )
        result = pipeline.run_pipeline(config)
        self.assertEqual(result.output_path.suffix, ".png")
        self.assertEqual(result.used_frames, len(shifts))
        self.assertEqual((result.width, result.height), (200, 160))
        # Residual is dominated by cubic resampling of the shifted frames and
        # by 8 bit quantisation of the source, not by the stacking step.
        self.assertLess(rmse(rawio.load_frame(result.output_path), scene), 0.03)

    def test_pipeline_accepts_jpeg_sources(self) -> None:
        scene = make_scene(120, 150)
        folder = self.temp / "jpeg"
        folder.mkdir(parents=True, exist_ok=True)
        shifts = [(0.0, 0.0), (0.5, -0.4), (-0.6, 0.5)]
        for index, (dx, dy) in enumerate(shifts):
            frame = to_srgb8(shift(scene, dx, dy))
            cv2.imwrite(
                str(folder / f"shot_{index:02d}.jpg"),
                frame[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            output_path=str(self.temp / "jpeg_out.tif"),
            mode="noise_reduction",
            stack_method="median",
            output_format="tiff",
            max_dim=600,
        )
        result = pipeline.run_pipeline(config)
        self.assertEqual(result.output_path.suffix, ".tif")
        self.assertEqual(result.used_frames, len(shifts))
        self.assertEqual((result.width, result.height), (150, 120))
        # JPEG compression dominates the residual, so only sanity check it.
        self.assertLess(rmse(rawio.load_frame(result.output_path), scene), 0.1)

    def test_dng_writer_roundtrips_or_falls_back(self) -> None:
        scene = make_scene(96, 128)
        metadata = rawio.FrameMetadata(
            width=128,
            height=96,
            is_raw=True,
            camera_model="Synthetic",
            color_matrix=np.eye(3),
            as_shot_neutral=np.array([0.5, 1.0, 0.7]),
        )
        written = rawio.save_image(self.temp / "dng_out", scene, "dng", metadata)
        self.assertIn(written.suffix.lower(), {".dng", ".tif"})
        if written.suffix.lower() == ".dng":
            # A usable DNG must be readable by LibRaw at the right size.
            self.assertIsNotNone(rawio.rawpy)
            loaded = rawio.load_frame(written)
            self.assertEqual(loaded.shape, scene.shape)

            # Metadata must survive a round-trip, otherwise a second pass
            # could not write DNG at all (the observed CLI fallback bug).
            reread = rawio.read_metadata(written)
            self.assertIsNotNone(reread.color_matrix)
            self.assertEqual(reread.camera_model, "Synthetic")
            self.assertEqual(float(np.asarray(reread.as_shot_neutral)[0]), 0.5)
            second = rawio.save_image(self.temp / "second", scene, "dng", reread)
            self.assertEqual(second.suffix, ".dng")

    def test_pipeline_end_to_end(self) -> None:
        scene = make_scene(200, 260)
        rng = np.random.default_rng(9)
        shifts = [(0.0, 0.0), (0.5, -0.4), (-0.7, 0.6), (0.9, 0.3)]
        frames = []
        for dx, dy in shifts:
            noisy = shift(scene, dx, dy)
            noisy += rng.normal(0.0, 0.02, noisy.shape).astype(np.float32)
            frames.append(noisy)
        folder = self.temp / "frames"
        write_frames(folder, frames)

        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            output_path=str(self.temp / "stacked.tif"),
            mode="noise_reduction",
            stack_method="mean",
            output_format="tiff",
            max_dim=1000,
        )
        messages: list[str] = []
        result = pipeline.run_pipeline(config, log=messages.append)
        self.assertTrue(result.output_path.exists())
        self.assertEqual(result.used_frames, len(frames))
        self.assertEqual((result.width, result.height), (260, 200))
        self.assertLess(rmse(rawio.load_frame(result.output_path), scene), 0.02)
        self.assertTrue(any("Reference frame" in message for message in messages))

    def test_default_output_lives_next_to_the_sources(self) -> None:
        folder = self.temp / "stack_a"
        folder.mkdir(parents=True, exist_ok=True)
        path = rawio.default_output_path(str(folder), "noise_reduction", "mean", "png")
        self.assertEqual(path.parent, folder)
        self.assertEqual(path.suffix, ".png")
        self.assertTrue(rawio.is_tool_output(path))

    def test_previous_output_is_not_stacked_again(self) -> None:
        folder = self.temp / "reuse"
        write_frames(folder, np.stack([make_scene(64, 64), shift(make_scene(64, 64), 0.4, 0.3)]))
        marker = rawio.save_image(folder / "reuse_stack_mean", make_scene(64, 64), "tiff")
        self.assertEqual(marker.parent, folder)

        frames = rawio.discover_frames(folder)
        self.assertEqual(len(frames), 2)
        self.assertNotIn(marker, frames)

        # A run with a different method must also ignore the older result.
        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            output_path=str(self.temp / "reuse_out.tif"),
            stack_method="median",
            output_format="tiff",
            max_dim=600,
        )
        result = pipeline.run_pipeline(config)
        self.assertEqual(result.used_frames, 2)
        self.assertEqual(result.frame_count, 2)

    def test_export_aligned_frames(self) -> None:
        scene = make_scene(120, 150)
        folder = self.temp / "aligned_src"
        shifts = [(0.6, -0.4), (-0.5, 0.5), (0.3, 0.2), (0.0, 0.0)]
        write_frames(folder, np.stack([shift(scene, dx, dy) for dx, dy in shifts]))

        output = self.temp / "result.tif"
        messages: list[str] = []
        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            output_path=str(output),
            output_format="tiff",
            export_aligned=True,
            max_dim=600,
        )
        result = pipeline.run_pipeline(config, log=messages.append)

        self.assertEqual(result.aligned_dir, self.temp / "result_aligned")
        self.assertEqual(result.aligned_count, len(shifts))
        written = sorted(result.aligned_dir.iterdir())
        self.assertEqual(len(written), len(shifts))
        self.assertEqual(written[0].name, "01_frame_00.tif")

        # Frame 4 is unshifted and is the reference, so its aligned copy must
        # reproduce the scene almost exactly.
        reference = rawio.load_frame(result.aligned_dir / "04_frame_03.tif")
        self.assertEqual(reference.shape, scene.shape)
        self.assertLess(rmse(reference, scene), 2e-3)
        self.assertTrue(any("aligned frames" in message.lower() for message in messages))

    def test_aligned_folder_is_skipped_when_recursing(self) -> None:
        folder = self.temp / "recursive"
        write_frames(folder, np.stack([make_scene(64, 64), shift(make_scene(64, 64), 0.5, 0.2)]))
        aligned = folder / "recursive_stack_mean_aligned"
        aligned.mkdir()
        tifffile.imwrite(str(aligned / "01_extra.tif"), np.zeros((64, 64, 3), np.uint16))

        frames = rawio.discover_frames(folder, recursive=True)
        self.assertEqual(len(frames), 2)

    def test_combined_mode_writes_both_results(self) -> None:
        folder = self.temp / "both_src"
        scene = make_scene(96, 120)
        write_frames(
            folder, np.stack([scene, shift(scene, 0.6, -0.4), shift(scene, -0.5, 0.5)])
        )
        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            mode="both",
            stack_method="median",
            output_format="png",
            upscale=2,
            max_dim=600,
        )
        results = pipeline.run_pipelines(config)

        self.assertEqual(
            [result.stats.mode for result in results],
            ["noise_reduction", "super_resolution"],
        )
        self.assertEqual(
            [result.output_path.name for result in results],
            ["both_src_stack_noise_median.png", "both_src_stack_sr2x.png"],
        )
        self.assertEqual((results[0].width, results[0].height), (120, 96))
        self.assertEqual((results[1].width, results[1].height), (240, 192))
        for result in results:
            self.assertTrue(result.output_path.exists())

    def test_combined_mode_registers_once(self) -> None:
        """Both pipelines reuse one registration and share a monotonic bar."""
        folder = self.temp / "shared"
        scene = make_scene(96, 120)
        write_frames(
            folder, np.stack([scene, shift(scene, 0.6, -0.4), shift(scene, -0.5, 0.5)])
        )
        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            mode="both",
            output_format="tiff",
            upscale=2,
            max_dim=600,
        )
        fractions: list[float] = []
        with mock.patch.object(
            aligner, "align_frames", wraps=aligner.align_frames
        ) as registration:
            results = pipeline.run_pipelines(
                config, progress=lambda fraction, _message: fractions.append(fraction)
            )

        self.assertEqual(registration.call_count, 1)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result.output_path.exists())
            self.assertEqual(result.used_frames, 3)
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)

    def test_combined_mode_decodes_each_frame_once(self) -> None:
        """The combined pass reads every frame a single time."""
        folder = self.temp / "decode_once"
        scene = make_scene(96, 120)
        write_frames(
            folder, np.stack([scene, shift(scene, 0.6, -0.4), shift(scene, -0.5, 0.5)])
        )
        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            mode="both",
            output_format="tiff",
            max_dim=600,
        )
        reads: list[str] = []
        original = stacker.load_frame

        def counting(path, *args, **kwargs):
            reads.append(str(path))
            return original(path, *args, **kwargs)

        with mock.patch.object(stacker, "load_frame", side_effect=counting):
            results = pipeline.run_pipelines(config)

        self.assertEqual(len(reads), 3)
        self.assertEqual(len(set(reads)), 3)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result.output_path.exists())

    def test_combined_engine_matches_separate_engines(self) -> None:
        """One shared pass must produce the same pixels as running both."""
        folder = self.temp / "equivalence"
        scene = make_scene(80, 100)
        rng = np.random.default_rng(11)
        frames = np.stack(
            [
                scene + rng.normal(0.0, 0.01, scene.shape).astype(np.float32),
                shift(scene, 0.5, 0.35) + rng.normal(0.0, 0.01, scene.shape).astype(np.float32),
                shift(scene, -0.4, 0.6) + rng.normal(0.0, 0.01, scene.shape).astype(np.float32),
            ]
        )
        write_frames(folder, frames)
        paths = [str(path) for path in sorted(folder.glob("*.tif"))]
        matrices = register(frames)

        for method in ("mean", "median"):
            stack_image, _ = stacker.stack_noise_reduction(paths, matrices, method=method)
            super_image, _ = stacker.stack_super_resolution(
                paths, matrices, upscale=2, pixfrac=0.75
            )
            combined_stack, _, combined_super, _ = stacker.stack_combined(
                paths, matrices, method=method, upscale=2, pixfrac=0.75
            )
            self.assertEqual(combined_stack.shape, stack_image.shape)
            self.assertEqual(combined_super.shape, super_image.shape)
            self.assertLess(
                float(np.abs(combined_stack - stack_image).max()), 1e-6,
                f"{method} stack differs",
            )
            self.assertLess(
                float(np.abs(combined_super - super_image).max()), 1e-6,
                f"{method} drizzle differs",
            )

    def test_combined_mode_tags_an_explicit_output_path(self) -> None:
        folder = self.temp / "explicit"
        scene = make_scene(64, 64)
        write_frames(folder, np.stack([scene, shift(scene, 0.4, 0.3)]))
        config = pipeline.PipelineConfig(
            input_dir=str(folder),
            mode="both",
            output_format="tiff",
            output_path=str(self.temp / "mine.tif"),
            max_dim=400,
        )
        results = pipeline.run_pipelines(config)
        self.assertEqual(
            sorted(result.output_path.name for result in results),
            ["mine_noise_mean.tif", "mine_sr2x.tif"],
        )

    def test_file_selection_replaces_folder_discovery(self) -> None:
        folder = self.temp / "picked"
        scene = make_scene(64, 64)
        write_frames(
            folder, np.stack([scene, shift(scene, 0.5, 0.2), shift(scene, -0.4, 0.4)])
        )
        chosen = [str(folder / "frame_00.tif"), str(folder / "frame_02.tif")]
        config = pipeline.PipelineConfig(
            input_dir="",
            input_files=chosen,
            output_path=str(self.temp / "sel.tif"),
            max_dim=400,
        )
        result = pipeline.run_pipeline(config)
        self.assertEqual(result.used_frames, 2)
        self.assertEqual(result.frame_count, 2)
        self.assertTrue(result.output_path.exists())

    def test_file_selection_rejects_foreign_files(self) -> None:
        bad = self.temp / "notes.txt"
        bad.write_text("x", encoding="utf-8")
        config = pipeline.PipelineConfig(input_files=[str(bad), str(bad)])
        with self.assertRaises(pipeline.PipelineError) as ctx:
            pipeline.run_pipeline(config)
        self.assertIn("not supported", str(ctx.exception))

    def test_pipeline_reports_bad_folder(self) -> None:
        config = pipeline.PipelineConfig(input_dir=str(self.temp / "missing"))
        with self.assertRaises(pipeline.PipelineError):
            pipeline.run_pipeline(config)

    def test_combined_mode_validates_drizzle_settings(self) -> None:
        """Combined mode drizzles, so it must enforce the drizzle limits."""
        for config in (
            pipeline.PipelineConfig(input_dir=str(self.temp), mode="both", upscale=9),
            pipeline.PipelineConfig(input_dir=str(self.temp), mode="both", pixfrac=1.5),
        ):
            with self.assertRaises(pipeline.PipelineError):
                config.validate()

    def test_single_frame_is_rejected(self) -> None:
        folder = self.temp / "single"
        write_frames(folder, make_scene(80, 80)[None])
        config = pipeline.PipelineConfig(input_dir=str(folder), output_format="tiff")
        with self.assertRaises(pipeline.PipelineError) as ctx:
            pipeline.run_pipeline(config)
        self.assertIn("at least two", str(ctx.exception))


class EntryPointTests(unittest.TestCase):
    """A folder dropped onto the executable must start a run in the GUI."""

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="mio_entry_"))
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.folder = self.temp / "frames"
        scene = make_scene(64, 64)
        write_frames(self.folder, np.stack([scene, shift(scene, 0.5, 0.4)]))

    def test_dropped_folder_opens_gui_and_autostarts_both(self) -> None:
        import main

        with mock.patch.object(main, "launch_gui", return_value=0) as launch:
            self.assertEqual(main.main([str(self.folder)]), 0)
        launch.assert_called_once_with(
            initial_dir=str(self.folder), auto_start=True, forced_mode="both"
        )

    def test_dropped_images_open_gui_with_selection(self) -> None:
        import main

        files = sorted(str(path) for path in self.folder.glob("*.tif"))
        with mock.patch.object(main, "launch_gui", return_value=0) as launch:
            self.assertEqual(main.main(files), 0)
        kwargs = launch.call_args.kwargs
        self.assertEqual(sorted(kwargs["initial_files"]), files)
        self.assertTrue(kwargs["auto_start"])
        self.assertEqual(kwargs["forced_mode"], "both")
        self.assertEqual(kwargs["initial_dir"], str(self.folder))

    def test_unsupported_dropped_file_is_rejected(self) -> None:
        import main

        text_file = self.temp / "notes.txt"
        text_file.write_text("not an image", encoding="utf-8")
        with mock.patch.object(main, "launch_gui") as launch, mock.patch.object(
            main, "_show_fatal_dialog"
        ):
            self.assertEqual(main.main([str(text_file)]), 2)
        launch.assert_not_called()

    def test_missing_dropped_item_is_rejected(self) -> None:
        import main

        with mock.patch.object(main, "launch_gui") as launch, mock.patch.object(
            main, "_show_fatal_dialog"
        ):
            self.assertEqual(main.main([str(self.temp / "gone")]), 2)
        launch.assert_not_called()

    def test_input_dir_conflicts_with_dropped_folder(self) -> None:
        import main

        with mock.patch.object(main, "launch_gui") as launch, mock.patch.object(
            main, "_show_fatal_dialog"
        ):
            self.assertEqual(main.main(["-i", str(self.folder), str(self.folder)]), 2)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
