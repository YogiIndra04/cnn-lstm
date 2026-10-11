"""
Konfigurasi terpusat untuk sistem CNN-LSTM pengenalan kata Bahasa Kolok.

Semua parameter penelitian harus dibaca dari sini.
Jangan mendefinisikan ulang nilai yang sama di file lain.

Keputusan implementasi sementara (⚠️) ditandai dan memerlukan
konfirmasi dari peneliti apabila berbeda dari dokumen skripsi.
"""

# ---------------------------------------------------------------------------
# PREPROCESSING VIDEO
# ---------------------------------------------------------------------------

NUM_FRAMES: int = 30
"""Jumlah frame per video (spesifikasi skripsi)."""

IMAGE_HEIGHT: int = 160
"""Tinggi frame dalam piksel (dioptimasi ke 160x160 untuk efisiensi VRAM 16GB)."""

IMAGE_WIDTH: int = 160
"""Lebar frame dalam piksel (dioptimasi ke 160x160 untuk efisiensi VRAM 16GB)."""

CHANNELS: int = 3
"""Jumlah kanal warna (RGB)."""

COLOR_MODE: str = "RGB"
"""Mode warna input model."""

# ---------------------------------------------------------------------------
# ARSITEKTUR MODEL
# ---------------------------------------------------------------------------

CNN_FILTERS: tuple[int, ...] = (32, 64, 128)
"""Jumlah filter per blok CNN."""

CNN_KERNEL_SIZE: tuple[int, int] = (3, 3)
"""Ukuran kernel konvolusi."""

CNN_PADDING: str = "same"
"""Padding konvolusi."""

POOL_SIZE: tuple[int, int] = (2, 2)
"""Ukuran MaxPooling."""

LSTM_UNITS: int = 128
"""Jumlah unit LSTM."""

DENSE_UNITS: int = 64
"""Jumlah unit Dense sebelum lapisan output."""

DROPOUT_RATE: float = 0.5
"""Dropout setelah LSTM (spesifikasi skripsi)."""

# ---------------------------------------------------------------------------
# KONFIGURASI PELATIHAN
# ---------------------------------------------------------------------------

LEARNING_RATE: float = 1e-3
"""Learning rate awal optimizer Adam."""

BATCH_SIZE: int = 2
"""Ukuran batch pelatihan (default 2 untuk VRAM 16GB, fallback ke 1 jika OOM)."""

USE_MIXED_PRECISION: bool = True
"""Gunakan mixed precision (mixed_float16) pada GPU untuk menghemat VRAM dan mempercepat training."""

MAX_EPOCHS: int = 50
"""Maksimum epoch pelatihan."""

EARLY_STOPPING_PATIENCE: int = 10
"""Patience EarlyStopping (spesifikasi skripsi)."""

EARLY_STOPPING_MONITOR: str = "val_loss"
"""Metrik yang dipantau EarlyStopping."""

REDUCE_LR_PATIENCE: int = 5
"""Patience ReduceLROnPlateau (spesifikasi skripsi)."""

REDUCE_LR_FACTOR: float = 0.5
"""Faktor pengurangan learning rate."""

REDUCE_LR_MONITOR: str = "val_loss"
"""Metrik yang dipantau ReduceLROnPlateau."""

REDUCE_LR_MIN_LR: float = 1e-7
"""Minimum learning rate."""

# ---------------------------------------------------------------------------
# PEMBAGIAN DATASET  ⚠️ Keputusan sementara — belum dikonfirmasi skripsi
# ---------------------------------------------------------------------------

TRAIN_RATIO: float = 0.70
"""Proporsi data training (keputusan sementara 70/15/15)."""

VAL_RATIO: float = 0.15
"""Proporsi data validasi (keputusan sementara 70/15/15)."""

TEST_RATIO: float = 0.15
"""Proporsi data testing (keputusan sementara 70/15/15)."""

RANDOM_SEED: int = 42
"""Seed untuk reproducibility di seluruh proses."""

# ---------------------------------------------------------------------------
# PATH DEFAULT ARTEFAK
# ---------------------------------------------------------------------------

RAW_VIDEO_DIR: str = "data/raw_videos"
PROCESSED_DIR: str = "data/processed"
SPLITS_DIR: str = "data/splits"
METADATA_DIR: str = "data/metadata"
MODELS_DIR: str = "models"
REPORTS_DIR: str = "reports"
LOGS_DIR: str = "logs"

DATASET_NPZ: str = "data/processed/dataset_sequences.npz"
LABEL_MAP_JSON: str = "data/processed/label_map.json"
SPLIT_MANIFEST_JSON: str = "data/splits/split_manifest.json"
BEST_MODEL_PATH: str = "models/best_model.keras"
FINAL_MODEL_PATH: str = "models/final_model.keras"
TRAINING_HISTORY_NPZ: str = "models/training_history.npz"
TRAINING_CONFIG_JSON: str = "models/training_config.json"
EVAL_SUMMARY_JSON: str = "reports/evaluation_summary.json"
EVAL_REPORT_CSV: str = "reports/classification_report.csv"
EVAL_CM_CSV: str = "reports/confusion_matrix.csv"
EVAL_CM_PNG: str = "reports/confusion_matrix.png"
EVAL_PREDICTIONS_CSV: str = "reports/test_predictions.csv"
PREPROCESS_REPORT_JSON: str = "reports/preprocessing_report.json"

# ---------------------------------------------------------------------------
# VALIDASI KONFIGURASI
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """Periksa konsistensi konfigurasi saat modul diimpor."""
    assert NUM_FRAMES > 0, "NUM_FRAMES harus positif"
    assert IMAGE_HEIGHT > 0, "IMAGE_HEIGHT harus positif"
    assert IMAGE_WIDTH > 0, "IMAGE_WIDTH harus positif"
    assert CHANNELS == 3, "CHANNELS harus 3 (RGB)"
    assert COLOR_MODE == "RGB", "COLOR_MODE harus RGB"
    assert 0.0 < DROPOUT_RATE < 1.0, "DROPOUT_RATE harus dalam (0, 1)"
    assert LSTM_UNITS > 0, "LSTM_UNITS harus positif"
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, (
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO harus = 1.0"
    )
    assert EARLY_STOPPING_PATIENCE > 0, "EARLY_STOPPING_PATIENCE harus positif"
    assert REDUCE_LR_PATIENCE > 0, "REDUCE_LR_PATIENCE harus positif"


validate_config()
