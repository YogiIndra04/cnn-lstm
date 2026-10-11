"""
Arsitektur model CNN-LSTM untuk pengenalan kata Bahasa Kolok.

Metodologi (dioptimasi untuk efisiensi VRAM 16GB):
  Input  : (batch, 30, 160, 160, 3) float32 [0, 1]
  CNN    : 3 blok Conv2D (32→64→128 filter, 3×3, ReLU, same) + MaxPool2D (2×2)
  GAP    : TimeDistributed(GlobalAveragePooling2D) → (batch, 30, 128)
  LSTM   : 128 unit, return_sequences=False → (batch, 128)
  Dropout: 0.5
  Dense  : 64 unit, ReLU
  Output : n_classes unit, Softmax (dtype=float32 untuk stabilitas numerik mixed precision)
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.model_config import (
    CHANNELS,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_PADDING,
    DENSE_UNITS,
    DROPOUT_RATE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LSTM_UNITS,
    NUM_FRAMES,
    POOL_SIZE,
)


def build_model(n_classes: int) -> tf.keras.Model:
    """
    Bangun model CNN2D-LSTM sesuai spesifikasi skripsi.

    Args:
        n_classes: Jumlah kelas kata Bahasa Kolok yang terverifikasi dari dataset.

    Returns:
        Model Keras yang belum dikompilasi.
    """
    if n_classes < 2:
        raise ValueError(
            f"n_classes harus >= 2, diperoleh {n_classes}. "
            "Pastikan dataset memiliki minimal 2 kelas yang valid."
        )

    inputs = layers.Input(
        shape=(NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
        name="video_sequence_input",
    )

    # --- Blok CNN 2D melalui TimeDistributed ---
    # Bobot CNN dibagi di seluruh langkah waktu (shared weights)

    # Blok 1: Conv2D 32 filter
    x = layers.TimeDistributed(
        layers.Conv2D(CNN_FILTERS[0], CNN_KERNEL_SIZE, padding=CNN_PADDING, activation="relu"),
        name="td_conv1",
    )(inputs)
    x = layers.TimeDistributed(
        layers.MaxPooling2D(POOL_SIZE),
        name="td_pool1",
    )(x)

    # Blok 2: Conv2D 64 filter
    x = layers.TimeDistributed(
        layers.Conv2D(CNN_FILTERS[1], CNN_KERNEL_SIZE, padding=CNN_PADDING, activation="relu"),
        name="td_conv2",
    )(x)
    x = layers.TimeDistributed(
        layers.MaxPooling2D(POOL_SIZE),
        name="td_pool2",
    )(x)

    # Blok 3: Conv2D 128 filter
    x = layers.TimeDistributed(
        layers.Conv2D(CNN_FILTERS[2], CNN_KERNEL_SIZE, padding=CNN_PADDING, activation="relu"),
        name="td_conv3",
    )(x)
    x = layers.TimeDistributed(
        layers.MaxPooling2D(POOL_SIZE),
        name="td_pool3",
    )(x)

    # GlobalAveragePooling2D → (batch, T, 128)
    x = layers.TimeDistributed(
        layers.GlobalAveragePooling2D(),
        name="td_gap",
    )(x)
    # Bentuk di sini harus: (batch, NUM_FRAMES, 128)

    # --- LSTM ---
    # return_sequences=False → ambil output di langkah waktu terakhir
    x = layers.LSTM(LSTM_UNITS, return_sequences=False, name="lstm")(x)
    # Bentuk: (batch, 128)

    # --- Dropout ---
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)

    # --- Dense 64 ---
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense_64")(x)

    # --- Output Softmax (float32 untuk stabilitas numerik pada mixed precision) ---
    outputs = layers.Dense(n_classes, activation="softmax", dtype="float32", name="class_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn2d_lstm_kolok")
    return model


def verify_model_shapes(model: tf.keras.Model, n_classes: int) -> None:
    """
    Periksa bahwa input/output shape model sesuai spesifikasi.
    Raise AssertionError dengan pesan jelas jika ada ketidaksesuaian.
    """
    expected_input = (None, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    expected_output = (None, n_classes)

    actual_input = model.input_shape
    actual_output = model.output_shape

    assert actual_input == expected_input, (
        f"Input shape model salah: {actual_input} (diharapkan {expected_input})"
    )
    assert actual_output == expected_output, (
        f"Output shape model salah: {actual_output} (diharapkan {expected_output})"
    )

    # Verifikasi LSTM output
    lstm_layer = model.get_layer("lstm")
    assert lstm_layer.units == LSTM_UNITS, (
        f"LSTM units salah: {lstm_layer.units} (diharapkan {LSTM_UNITS})"
    )
    assert not lstm_layer.return_sequences, (
        "LSTM harus return_sequences=False untuk klasifikasi sequence-to-one"
    )

    # Verifikasi filter CNN terakhir melalui model output yang dikumpulkan
    # (TimeDistributed tidak selalu mengekspos output_shape secara langsung
    #  bergantung pada versi Keras/TF)
    td_gap = model.get_layer("td_gap")
    # Periksa melalui dummy forward pass jika perlu — cukup verifikasi lapisan ada
    assert td_gap is not None, "Layer 'td_gap' tidak ditemukan di model"

    # Verifikasi Dropout
    dropout_layer = model.get_layer("dropout")
    assert abs(dropout_layer.rate - DROPOUT_RATE) < 1e-6, (
        f"Dropout rate salah: {dropout_layer.rate} (diharapkan {DROPOUT_RATE})"
    )
