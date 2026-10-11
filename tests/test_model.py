"""
Unit test arsitektur model CNN-LSTM.

Menggunakan data dummy untuk memverifikasi mekanisme teknis (shape, kompilasi,
forward pass, simpan/muat). Bukan pengujian performa model.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model_config import (
    CHANNELS,
    CNN_FILTERS,
    DROPOUT_RATE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LSTM_UNITS,
    NUM_FRAMES,
)
from model_architecture import build_model, verify_model_shapes


# ---------------------------------------------------------------------------
# Test: build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_valid_build(self):
        model = build_model(n_classes=3)
        assert model is not None

    def test_invalid_n_classes(self):
        with pytest.raises(ValueError, match="n_classes harus >= 2"):
            build_model(n_classes=1)

    def test_zero_n_classes(self):
        with pytest.raises(ValueError):
            build_model(n_classes=0)

    def test_input_shape(self):
        model = build_model(n_classes=5)
        expected = (None, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
        assert model.input_shape == expected, (
            f"Input shape: {model.input_shape} ≠ {expected}"
        )

    def test_output_shape_3_classes(self):
        model = build_model(n_classes=3)
        assert model.output_shape == (None, 3)

    def test_output_shape_10_classes(self):
        model = build_model(n_classes=10)
        assert model.output_shape == (None, 10)

    def test_lstm_units(self):
        model = build_model(n_classes=5)
        lstm = model.get_layer("lstm")
        assert lstm.units == LSTM_UNITS

    def test_gap_layer_exists(self):
        """td_gap layer harus ada (TimeDistributed(GlobalAveragePooling2D))."""
        model = build_model(n_classes=5)
        gap = model.get_layer("td_gap")
        assert gap is not None
        # Verifikasi melalui dummy forward pass
        import tensorflow as tf
        x = tf.zeros((1, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        out = model(x, training=False)
        assert out.shape[-1] == 5  # output kelas benar

    def test_dropout_rate(self):
        model = build_model(n_classes=5)
        dropout = model.get_layer("dropout")
        assert abs(dropout.rate - DROPOUT_RATE) < 1e-6, (
            f"Dropout rate: {dropout.rate} ≠ {DROPOUT_RATE}"
        )

    def test_model_name(self):
        model = build_model(n_classes=5)
        assert model.name == "cnn2d_lstm_kolok"

    def test_time_distributed_layers_exist(self):
        """TimeDistributed harus ada untuk CNN yang diterapkan pada frame."""
        model = build_model(n_classes=5)
        layer_types = [type(l).__name__ for l in model.layers]
        assert "TimeDistributed" in layer_types, (
            "TimeDistributed tidak ditemukan — CNN tidak diterapkan per-frame"
        )

    def test_softmax_output(self):
        """Probabilitas output harus dalam [0, 1] dan berjumlah ≈ 1."""
        import tensorflow as tf
        model = build_model(n_classes=5)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )
        dummy_input = np.random.rand(2, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS).astype(np.float32)
        probs = model.predict(dummy_input, verbose=0)
        assert probs.shape == (2, 5)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_forward_pass_batch(self):
        """Model harus dapat memproses batch tanpa error."""
        import tensorflow as tf
        model = build_model(n_classes=3)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )
        x = np.random.rand(4, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS).astype(np.float32)
        out = model(x, training=False)
        assert out.shape == (4, 3)

    def test_save_and_load(self, tmp_path):
        """Model harus dapat disimpan dan dimuat kembali."""
        import tensorflow as tf
        model = build_model(n_classes=3)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )
        save_path = tmp_path / "test_model.keras"
        model.save(str(save_path))
        loaded = tf.keras.models.load_model(str(save_path))
        assert loaded.input_shape == model.input_shape
        assert loaded.output_shape == model.output_shape


# ---------------------------------------------------------------------------
# Test: verify_model_shapes
# ---------------------------------------------------------------------------

class TestVerifyModelShapes:
    def test_valid_shapes_pass(self):
        import tensorflow as tf
        model = build_model(n_classes=5)
        # Build model dengan dummy forward pass agar layer shapes tersedia
        model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy())
        verify_model_shapes(model, 5)  # tidak raise

    def test_wrong_n_classes_raises(self):
        model = build_model(n_classes=3)
        with pytest.raises(AssertionError, match="Output shape model salah"):
            verify_model_shapes(model, 5)
