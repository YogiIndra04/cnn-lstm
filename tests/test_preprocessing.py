"""
Unit test untuk preprocessing video.

Video sintetis digunakan sebagai fixture — bukan bagian dataset penelitian.
Test ini hanya memverifikasi mekanisme program, bukan performa model.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Tambahkan root proyek ke path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess_dataset import (
    preprocess_single_frame,
    process_video,
    sample_frame_indices,
    validate_sequence,
)
from configs.model_config import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES


# ---------------------------------------------------------------------------
# Helper: buat video sintetis untuk test
# ---------------------------------------------------------------------------

def create_synthetic_video(
    tmp_dir: Path,
    filename: str,
    n_frames: int = 60,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
) -> Path:
    """
    Buat video MP4 sintetis (warna acak) untuk keperluan testing.
    BUKAN bagian dataset penelitian.
    """
    path = tmp_dir / filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    rng = np.random.default_rng(seed=42)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Test: sample_frame_indices
# ---------------------------------------------------------------------------

class TestSampleFrameIndices:
    def test_exact_count(self):
        indices = sample_frame_indices(60, 30)
        assert len(indices) == 30

    def test_monotonically_non_decreasing(self):
        indices = sample_frame_indices(60, 30)
        assert np.all(indices[1:] >= indices[:-1]), "Indeks harus berurutan (temporal)"

    def test_starts_from_zero(self):
        indices = sample_frame_indices(60, 30)
        assert indices[0] == 0

    def test_ends_at_last_frame(self):
        indices = sample_frame_indices(60, 30)
        assert indices[-1] == 59

    def test_short_video_repeat(self):
        """Video 10 frame, target 30 — indeks berulang tapi jumlah tetap 30."""
        indices = sample_frame_indices(10, 30)
        assert len(indices) == 30
        assert indices.min() >= 0
        assert indices.max() <= 9

    def test_single_frame_video(self):
        indices = sample_frame_indices(1, 30)
        assert len(indices) == 30
        assert all(i == 0 for i in indices)

    def test_exact_match_frames(self):
        """Video 30 frame, target 30 — setiap indeks unik."""
        indices = sample_frame_indices(30, 30)
        assert len(indices) == 30

    def test_deterministic(self):
        a = sample_frame_indices(60, 30)
        b = sample_frame_indices(60, 30)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Test: preprocess_single_frame
# ---------------------------------------------------------------------------

class TestPreprocessSingleFrame:
    def test_output_shape(self):
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        out = preprocess_single_frame(frame_bgr, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert out.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    def test_output_dtype(self):
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        out = preprocess_single_frame(frame_bgr, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert out.dtype == np.uint8

    def test_bgr_to_rgb_conversion(self):
        """BGR merah murni harus menjadi biru di channel terakhir setelah konversi."""
        # Frame BGr merah murni: B=0, G=0, R=255
        frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_bgr[:, :, 2] = 255  # channel R (index 2) dalam BGR
        out = preprocess_single_frame(frame_bgr, 4, 4)
        # Setelah BGR→RGB: index 0 = R = 255, index 2 = B = 0
        assert out[0, 0, 0] == 255, "Channel R di RGB harus 255"
        assert out[0, 0, 2] == 0, "Channel B di RGB harus 0"

    def test_grayscale_converted(self):
        """Frame grayscale harus dikonversi ke 3 kanal."""
        frame_gray = np.full((100, 100), 128, dtype=np.uint8)
        out = preprocess_single_frame(frame_gray, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert out.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    def test_alpha_channel_removed(self):
        """Frame BGRA harus dikonversi ke BGR dahulu."""
        frame_bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        frame_bgra[:, :, 2] = 200  # R dalam BGR
        out = preprocess_single_frame(frame_bgra, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert out.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    def test_value_range(self):
        frame_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = preprocess_single_frame(frame_bgr, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert out.min() >= 0
        assert out.max() <= 255


# ---------------------------------------------------------------------------
# Test: validate_sequence
# ---------------------------------------------------------------------------

class TestValidateSequence:
    def test_valid_sequence(self):
        seq = np.ones((NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        validate_sequence(seq, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)  # tidak raise

    def test_wrong_num_frames(self):
        seq = np.ones((NUM_FRAMES + 1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        with pytest.raises(AssertionError, match="Bentuk tensor salah"):
            validate_sequence(seq, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_wrong_resolution(self):
        seq = np.ones((NUM_FRAMES, 64, 64, 3), dtype=np.uint8)
        with pytest.raises(AssertionError, match="Bentuk tensor salah"):
            validate_sequence(seq, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_wrong_dtype(self):
        seq = np.ones((NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        with pytest.raises(AssertionError, match="dtype salah"):
            validate_sequence(seq, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_all_zeros_rejected(self):
        seq = np.zeros((NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        with pytest.raises(AssertionError, match="seluruhnya nol"):
            validate_sequence(seq, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)


# ---------------------------------------------------------------------------
# Test: process_video (menggunakan video sintetis)
# ---------------------------------------------------------------------------

class TestProcessVideo:
    def test_valid_video_output_shape(self, tmp_path):
        vpath = create_synthetic_video(tmp_path, "test_valid.mp4", n_frames=60)
        seq, info = process_video(vpath, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert seq.shape == (NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        assert seq.dtype == np.uint8
        assert info["status"] == "ok"

    def test_short_video_handled(self, tmp_path):
        """Video lebih pendek dari target — frame diulang, bukan error."""
        vpath = create_synthetic_video(tmp_path, "test_short.mp4", n_frames=10)
        seq, info = process_video(vpath, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert seq.shape == (NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        assert info["repeated"] is True

    def test_long_video_sampled(self, tmp_path):
        """Video lebih panjang dari target — disampling 30 frame."""
        vpath = create_synthetic_video(tmp_path, "test_long.mp4", n_frames=300)
        seq, info = process_video(vpath, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert seq.shape == (NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        assert info["original_frames"] == 300

    def test_file_not_found(self, tmp_path):
        with pytest.raises(RuntimeError, match="Tidak dapat membuka"):
            process_video(tmp_path / "nonexistent.mp4", NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_frame_sequence_not_empty(self, tmp_path):
        vpath = create_synthetic_video(tmp_path, "test_content.mp4", n_frames=60)
        seq, _ = process_video(vpath, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert not np.all(seq == 0), "Sequence tidak boleh seluruhnya nol"


