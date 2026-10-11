"""
Skrip pengujian bertahap untuk optimasi CNN-LSTM Bahasa Kolok:
1. Pembangunan model
2. Forward pass (batch size 2, 160x160x3, 30 frames)
3. Satu batch training step
4. Smoke test (2 epochs mini-training)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.model_config import (
    BATCH_SIZE,
    CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_FRAMES,
    DATASET_NPZ,
    LABEL_MAP_JSON,
)
from model_architecture import build_model, verify_model_shapes
from train_model import create_tf_dataset, set_seeds


def test_step_1_build_model(n_classes: int = 3) -> tf.keras.Model:
    print("\n" + "=" * 60)
    print("STEP 1: Pengujian Pembangunan Model CNN2D-LSTM")
    print("=" * 60)
    model = build_model(n_classes=n_classes)
    verify_model_shapes(model, n_classes=n_classes)
    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"  [OK] Model berhasil dibangun.")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Total Parameter: {total_params:,}")
    print(f"  Trainable Parameter: {trainable_params:,}")
    return model


def test_step_2_forward_pass(model: tf.keras.Model, batch_size: int = BATCH_SIZE) -> np.ndarray:
    print("\n" + "=" * 60)
    print(f"STEP 2: Pengujian Forward Pass (Batch Size = {batch_size})")
    print("=" * 60)
    dummy_input = tf.random.uniform(
        shape=(batch_size, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )
    t0 = time.time()
    outputs = model(dummy_input, training=False)
    elapsed = time.time() - t0
    probs = outputs.numpy()
    row_sums = probs.sum(axis=1)
    
    print(f"  [OK] Forward pass berhasil dieksekusi dalam {elapsed:.4f} detik.")
    print(f"  Output tensor shape: {outputs.shape}")
    print(f"  Probabilitas output (sample 0): {probs[0]}")
    print(f"  Jumlah probabilitas per baris: {row_sums}")
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilitas output softmax tidak berjumlah 1.0!"
    return probs


def test_step_3_train_one_batch(model: tf.keras.Model, batch_size: int = BATCH_SIZE) -> float:
    print("\n" + "=" * 60)
    print(f"STEP 3: Pengujian Satu Batch Training Step (train_on_batch)")
    print("=" * 60)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # Ambil batch nyata dari dataset jika ada, jika tidak gunakan dummy
    data_path = Path(DATASET_NPZ)
    if data_path.exists():
        data = np.load(data_path)
        X_batch = data["X_train"][:batch_size].astype(np.float32) / 255.0
        y_batch = data["y_train"][:batch_size]
        print(f"  Menggunakan data batch riil dari: {data_path}")
    else:
        X_batch = np.random.uniform(0.0, 1.0, (batch_size, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)).astype(np.float32)
        y_batch = np.array([0, 1][:batch_size], dtype=np.int32)
        print("  Menggunakan data batch dummy sintetis.")

    t0 = time.time()
    metrics = model.train_on_batch(X_batch, y_batch)
    elapsed = time.time() - t0
    
    loss_val = float(metrics[0]) if isinstance(metrics, (list, tuple)) else float(metrics)
    acc_val = float(metrics[1]) if isinstance(metrics, (list, tuple)) and len(metrics) > 1 else None
    
    print(f"  [OK] Satu batch training berhasil dalam {elapsed:.4f} detik.")
    print(f"  Loss: {loss_val:.4f}")
    if acc_val is not None:
        print(f"  Accuracy: {acc_val:.4f}")
    assert not np.isnan(loss_val) and not np.isinf(loss_val), "Loss tidak valid (NaN atau Inf)!"
    return loss_val


def test_step_4_smoke_test(model: tf.keras.Model, batch_size: int = BATCH_SIZE) -> None:
    print("\n" + "=" * 60)
    print(f"STEP 4: Smoke Test Training (2 Epochs dengan Callbacks Lengkap)")
    print("=" * 60)
    data_path = Path(DATASET_NPZ)
    if not data_path.exists():
        print("  [SKIP] Dataset .npz belum ada untuk smoke test.")
        return

    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    train_ds = create_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

    temp_ckpt = Path("models/smoke_test_checkpoint.keras")
    temp_ckpt.parent.mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(temp_ckpt), monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        callbacks=callbacks_list,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"  [OK] Smoke test 2 epoch selesai dalam {elapsed:.2f} detik.")
    print(f"  Epoch 1 loss: {history.history['loss'][0]:.4f}, val_loss: {history.history['val_loss'][0]:.4f}")
    print(f"  Epoch 2 loss: {history.history['loss'][1]:.4f}, val_loss: {history.history['val_loss'][1]:.4f}")
    if temp_ckpt.exists():
        print(f"  [OK] Checkpoint berhasil disimpan ke: {temp_ckpt}")
        temp_ckpt.unlink()  # Bersihkan file smoke test


def main():
    set_seeds(42)
    model = test_step_1_build_model(n_classes=3)
    test_step_2_forward_pass(model, batch_size=2)
    test_step_3_train_one_batch(model, batch_size=2)
    test_step_4_smoke_test(model, batch_size=2)
    print("\n" + "=" * 60)
    print("SELURUH PENGUJIAN TAHAP 1 - 4 BERHASIL (PASSED)!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
