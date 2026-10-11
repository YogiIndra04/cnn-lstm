"""
Training model CNN2D-LSTM untuk pengenalan kata Bahasa Kolok.

Konfigurasi menggunakan parameter dari configs/model_config.py.
Tidak boleh mengubah ketentuan training tanpa dokumentasi yang jelas.

Jalankan dari root proyek:
    python train_model.py [--dataset ...] [--output-model ...] [--batch-size ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.model_config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHANNELS,
    DATASET_NPZ,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    FINAL_MODEL_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL_MAP_JSON,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODELS_DIR,
    NUM_FRAMES,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN_LR,
    REDUCE_LR_MONITOR,
    REDUCE_LR_PATIENCE,
    TRAINING_CONFIG_JSON,
    TRAINING_HISTORY_NPZ,
)
from model_architecture import build_model, verify_model_shapes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Tetapkan random seed untuk Python, NumPy, dan TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    log.info("Random seed ditetapkan: %d", seed)


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------

def create_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """
    Buat tf.data.Dataset dengan normalisasi [0, 1] on-the-fly.
    Menjaga array X tetap uint8 di memori untuk mencegah OOM.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, lbl: (tf.cast(x, tf.float32) / 255.0, lbl),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latih model CNN2D-LSTM pengenalan kata Bahasa Kolok."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NPZ,
        help="Path ke file dataset_sequences.npz.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=BEST_MODEL_PATH,
        help="Path penyimpanan model terbaik (.keras).",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=LABEL_MAP_JSON,
        help="Path ke file label_map.json.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help="Jumlah maksimum epoch pelatihan.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Ukuran batch pelatihan (gunakan batch kecil misal 2 atau 4 di CPU).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate awal optimizer Adam.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seeds(RANDOM_SEED)

    # Periksa ketersediaan akselerator GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info("GPU terdeteksi (%d perangkat): %s", len(gpus), gpus)
    else:
        log.info("GPU tidak tersedia. Training menggunakan CPU.")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset tidak ditemukan di: %s. Jalankan preprocess_dataset.py terlebih dahulu.", dataset_path)
        sys.exit(1)

    label_map_path = Path(args.label_map)
    if not label_map_path.exists():
        log.error("Label map tidak ditemukan di: %s.", label_map_path)
        sys.exit(1)

    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    classes = label_map.get("classes", [])
    n_classes = len(classes)
    if n_classes < 2:
        log.error("Dataset membutuhkan minimal 2 kelas, ditemukan: %d", n_classes)
        sys.exit(1)

    log.info("Dataset valid.")
    log.info("  Kelas (%d): %s", n_classes, classes)

    data = np.load(dataset_path, allow_pickle=True)
    required_keys = {"X_train", "y_train", "X_val", "y_val"}
    if not required_keys.issubset(set(data.files)):
        log.error("Dataset .npz tidak memiliki kunci lengkap %s. Ditemukan: %s", required_keys, list(data.files))
        sys.exit(1)

    X_train = data["X_train"]
    y_train = data["y_train"].astype(np.int32)
    X_val = data["X_val"]
    y_val = data["y_val"].astype(np.int32)
    X_test = data["X_test"] if "X_test" in data.files else None
    y_test = data["y_test"].astype(np.int32) if "y_test" in data.files else None

    test_info = f" | X_test: {X_test.shape}" if X_test is not None else ""
    log.info("  X_train: %s | X_val: %s%s", X_train.shape, X_val.shape, test_info)

    # Validasi bentuk frame
    expected_sample_shape = (NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    if X_train.shape[1:] != expected_sample_shape:
        log.error(
            "Bentuk tensor sampel X_train %s tidak sesuai spesifikasi %s.",
            X_train.shape[1:], expected_sample_shape
        )
        sys.exit(1)

    log.info("Membangun model CNN2D-LSTM...")
    model = build_model(n_classes=n_classes)
    verify_model_shapes(model, n_classes=n_classes)
    log.info("Verifikasi shape model: OK")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    log_csv_path = Path(MODELS_DIR) / "training_log.csv"
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(output_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=REDUCE_LR_MONITOR,
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=1,
        ),
        callbacks.CSVLogger(filename=str(log_csv_path), append=False),
    ]

    train_ds = create_tf_dataset(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size=args.batch_size, shuffle=False)

    log.info("Memulai training...")
    log.info(
        "Config: epochs=%d | batch=%d | lr=%.4f | patience_es=%d | patience_rlr=%d",
        args.epochs, args.batch_size, args.learning_rate, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE
    )

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb_list,
        verbose=1,
    )
    elapsed = time.time() - start_time
    log.info("Training selesai dalam %.2f detik (%.2f menit).", elapsed, elapsed / 60.0)

    # Simpan final model
    final_model_path = Path(FINAL_MODEL_PATH)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_path))
    log.info("Final model tersimpan: %s", final_model_path)

    # Simpan history
    hist_path = Path(TRAINING_HISTORY_NPZ)
    hist_dict = {k: np.array(v) for k, v in history.history.items()}
    np.savez_compressed(hist_path, **hist_dict)
    log.info("Training history tersimpan: %s", hist_path)

    # Simpan konfigurasi training JSON
    config_record = {
        "dataset": str(dataset_path),
        "output_model": str(output_model_path),
        "final_model": str(final_model_path),
        "classes": classes,
        "n_classes": n_classes,
        "epochs_requested": args.epochs,
        "epochs_run": len(history.epoch),
        "batch_size": args.batch_size,
        "initial_learning_rate": args.learning_rate,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "reduce_lr_factor": REDUCE_LR_FACTOR,
        "elapsed_seconds": round(elapsed, 2),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    training_config_path = Path(TRAINING_CONFIG_JSON)
    with training_config_path.open("w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)
    log.info("Konfigurasi pelatihan tersimpan: %s", training_config_path)


if __name__ == "__main__":
    main()
