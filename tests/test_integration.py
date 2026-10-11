"""
Integration test: pipeline penuh video → model → prediksi → label.

Menggunakan video sintetis dan model dummy — BUKAN data penelitian.
Test ini hanya memverifikasi alur program, bukan akurasi model.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model_config import CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FRAMES
from model_architecture import build_model
from predict_video import predict_video, preprocess_video_for_inference


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_synthetic_video(path: Path, n_frames: int = 60) -> Path:
    """Buat video sintetis. BUKAN dataset penelitian."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30, (320, 240))
    rng = np.random.default_rng(42)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def make_label_map(tmp_path: Path, class_names: list[str]) -> Path:
    lbl2idx = {lbl: i for i, lbl in enumerate(class_names)}
    idx2lbl = {str(i): lbl for lbl, i in lbl2idx.items()}
    label_map = {
        "label_to_idx": lbl2idx,
        "idx_to_label": idx2lbl,
        "classes": class_names,
        "n_classes": len(class_names),
        "preprocessing": {
            "num_frames": NUM_FRAMES,
            "img_height": IMAGE_HEIGHT,
            "img_width": IMAGE_WIDTH,
            "channels": CHANNELS,
            "color_mode": "RGB",
            "normalization": "divide_by_255_float32",
            "input_shape_per_sample": [NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS],
        },
    }
    path = tmp_path / "label_map.json"
    path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Test: preprocess_video_for_inference
# ---------------------------------------------------------------------------

class TestPreprocessVideoForInference:
    def test_output_shape(self, tmp_path):
        vpath = make_synthetic_video(tmp_path / "test.mp4")
        x = preprocess_video_for_inference(vpath)
        assert x.shape == (1, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

    def test_output_dtype(self, tmp_path):
        vpath = make_synthetic_video(tmp_path / "test.mp4")
        x = preprocess_video_for_inference(vpath)
        assert x.dtype == np.float32

    def test_value_range(self, tmp_path):
        vpath = make_synthetic_video(tmp_path / "test.mp4")
        x = preprocess_video_for_inference(vpath)
        assert x.min() >= 0.0
        assert x.max() <= 1.0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocess_video_for_inference(tmp_path / "missing.mp4")

    def test_unsupported_extension(self, tmp_path):
        fake = tmp_path / "video.xyz"
        fake.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Format video tidak didukung"):
            preprocess_video_for_inference(fake)

    def test_deterministic(self, tmp_path):
        """Preprocessing yang sama pada video yang sama harus menghasilkan tensor identik."""
        vpath = make_synthetic_video(tmp_path / "det.mp4")
        x1 = preprocess_video_for_inference(vpath)
        x2 = preprocess_video_for_inference(vpath)
        np.testing.assert_array_equal(x1, x2)


# ---------------------------------------------------------------------------
# Test: predict_video (pipeline penuh)
# ---------------------------------------------------------------------------

class TestPredictVideo:
    def _make_compiled_model(self, n_classes: int):
        import tensorflow as tf
        model = build_model(n_classes=n_classes)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )
        return model

    def test_basic_prediction(self, tmp_path):
        classes = ["kamu_makan", "menelepon", "terima_kasih"]
        vpath = make_synthetic_video(tmp_path / "test.mp4")
        model = self._make_compiled_model(len(classes))
        idx2lbl = {i: lbl for i, lbl in enumerate(classes)}
        result = predict_video(vpath, model, idx2lbl)
        assert result["status"] == "ok"
        assert result["predicted_label"] in classes
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top_k_count(self, tmp_path):
        classes = ["a", "b", "c", "d", "e"]
        vpath = make_synthetic_video(tmp_path / "test.mp4")
        model = self._make_compiled_model(len(classes))
        idx2lbl = {i: lbl for i, lbl in enumerate(classes)}
        result = predict_video(vpath, model, idx2lbl, top_k=3)
        assert len(result["top_k"]) == 3

    def test_probabilities_sum_to_one(self, tmp_path):
        classes = ["kamu_makan", "menelepon", "terima_kasih"]
        vpath = make_synthetic_video(tmp_path / "test.mp4")
        model = self._make_compiled_model(len(classes))
        idx2lbl = {i: lbl for i, lbl in enumerate(classes)}
        result = predict_video(vpath, model, idx2lbl, top_k=3)
        prob_sum = sum(e["probability"] for e in result["top_k"])
        assert abs(prob_sum - 1.0) < 0.01 or len(result["top_k"]) < len(classes)

    def test_incompatible_label_map_raises(self, tmp_path):
        """Label map 3 kelas + model 5 kelas → error."""
        vpath = make_synthetic_video(tmp_path / "test.mp4")
        model = self._make_compiled_model(5)
        idx2lbl_wrong = {0: "a", 1: "b", 2: "c"}  # 3 kelas
        with pytest.raises(ValueError, match="tidak cocok dengan"):
            predict_video(vpath, model, idx2lbl_wrong)

    def test_missing_video_raises(self, tmp_path):
        classes = ["a", "b"]
        model = self._make_compiled_model(len(classes))
        idx2lbl = {0: "a", 1: "b"}
        with pytest.raises(FileNotFoundError):
            predict_video(tmp_path / "missing.mp4", model, idx2lbl)


# ---------------------------------------------------------------------------
# Test: model save & load → predict
# ---------------------------------------------------------------------------

class TestModelSaveLoadPredict:
    def test_save_load_and_predict(self, tmp_path):
        """Model disimpan lalu dimuat kembali harus menghasilkan prediksi yang sama."""
        import tensorflow as tf
        classes = ["kamu_makan", "menelepon", "terima_kasih"]
        n = len(classes)
        model = build_model(n_classes=n)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )
        save_path = tmp_path / "model.keras"
        model.save(str(save_path))
        loaded = tf.keras.models.load_model(str(save_path))

        vpath = make_synthetic_video(tmp_path / "test.mp4")
        idx2lbl = {i: lbl for i, lbl in enumerate(classes)}

        r1 = predict_video(vpath, model, idx2lbl)
        r2 = predict_video(vpath, loaded, idx2lbl)

        assert r1["predicted_label"] == r2["predicted_label"]
        assert abs(r1["confidence"] - r2["confidence"]) < 1e-4
