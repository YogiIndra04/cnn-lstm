"""
Evaluasi model CNN-LSTM pengenalan kata Bahasa Kolok.

PENTING: Evaluasi HANYA dilakukan pada TEST SET yang independen dari
training dan validation. Tidak menggunakan val set untuk evaluasi final.

Output:
  - reports/evaluation_summary.json
  - reports/classification_report.csv
  - reports/confusion_matrix.csv
  - reports/confusion_matrix.png
  - reports/test_predictions.csv

Jalankan dari root proyek:
    python evaluate_model.py [--model ...] [--dataset ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.model_config import (
    BEST_MODEL_PATH,
    DATASET_NPZ,
    EVAL_CM_CSV,
    EVAL_CM_PNG,
    EVAL_PREDICTIONS_CSV,
    EVAL_REPORT_CSV,
    EVAL_SUMMARY_JSON,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL_MAP_JSON,
    NUM_FRAMES,
    CHANNELS,
)

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
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluasi model CNN2D-LSTM pada test set independen."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=BEST_MODEL_PATH,
        help="Path ke model .keras.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NPZ,
        help="Path ke dataset_sequences.npz.",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=LABEL_MAP_JSON,
        help="Path ke label_map.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Direktori output laporan evaluasi.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------

def main() -> None:
    # Import di sini agar error terjadi lebih awal jika library tidak tersedia
    try:
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import seaborn as sns
    except ImportError as e:
        log.error(
            "Library yang dibutuhkan tidak terpasang: %s. "
            "Jalankan: pip install pandas matplotlib seaborn scikit-learn",
            e,
        )
        sys.exit(1)

    args = parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    label_map_path = Path(args.label_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Validasi file ---
    missing_files = []
    if not dataset_path.exists():
        missing_files.append(str(dataset_path))
    if not model_path.exists():
        missing_files.append(str(model_path))
    if not label_map_path.exists():
        missing_files.append(str(label_map_path))

    if missing_files:
        log.error(
            "File berikut tidak ditemukan:\n  %s\n"
            "Pastikan preprocessing dan training sudah selesai.",
            "\n  ".join(missing_files),
        )
        sys.exit(1)

    # --- Muat label map ---
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx_to_label: dict[int, str] = {
        int(k): v for k, v in label_map["idx_to_label"].items()
    }
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    n_classes = len(class_names)
    log.info("Kelas (%d): %s", n_classes, class_names)

    # --- Muat dataset (hanya test set) ---
    data = np.load(dataset_path, allow_pickle=True)

    if "X_test" not in data.files or "y_test" not in data.files:
        log.error(
            "Kunci 'X_test' atau 'y_test' tidak ada di dataset. "
            "Dataset ini mungkin dibuat oleh preprocess lama yang tidak memiliki test split. "
            "Jalankan ulang preprocess_dataset.py."
        )
        sys.exit(1)

    X_test = data["X_test"].astype(np.float32) / 255.0
    y_test = data["y_test"].astype(np.int32)

    if len(X_test) == 0:
        log.error(
            "Test set kosong. "
            "Periksa dataset dan pembagian train/val/test di preprocess_dataset.py."
        )
        sys.exit(1)

    # Validasi dimensi test set
    if X_test.shape[1:] != (NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS):
        log.error(
            "Dimensi X_test tidak sesuai konfigurasi model. "
            "X_test: %s | Diharapkan: (%s, %d, %d, %d, %d).",
            X_test.shape, "N", NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS,
        )
        sys.exit(1)

    log.info("Test set: %d sampel | shape=%s", len(y_test), X_test.shape)

    # --- Muat model ---
    log.info("Memuat model: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))

    # Verifikasi kompatibilitas model dengan label map
    model_n_classes = model.output_shape[-1]
    if model_n_classes != n_classes:
        log.error(
            "Jumlah kelas model (%d) tidak cocok dengan label map (%d). "
            "Model ini mungkin dilatih dengan dataset berbeda.",
            model_n_classes, n_classes,
        )
        sys.exit(1)

    # --- Prediksi ---
    log.info("Menjalankan prediksi pada test set...")
    probs = model.predict(X_test, verbose=1)  # (N, n_classes)
    y_pred = np.argmax(probs, axis=1)

    # --- Hitung metrik ---
    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )

    log.info("=== Hasil Evaluasi Test Set ===")
    log.info("Accuracy              : %.4f", acc)
    log.info("Precision (macro)     : %.4f", precision_macro)
    log.info("Recall    (macro)     : %.4f", recall_macro)
    log.info("F1-score  (macro)     : %.4f", f1_macro)
    log.info("Precision (weighted)  : %.4f", precision_weighted)
    log.info("Recall    (weighted)  : %.4f", recall_weighted)
    log.info("F1-score  (weighted)  : %.4f", f1_weighted)
    log.info("Sampel test dievaluasi: %d", len(y_test))
    log.info("\nClassification Report:\n%s", report_str)

    # --- Simpan evaluation_summary.json ---
    summary = {
        "status": "evaluation_completed",
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "n_test_samples": int(len(y_test)),
        "n_classes": n_classes,
        "classes": class_names,
        "metrics": {
            "accuracy": round(float(acc), 6),
            "precision_macro": round(float(precision_macro), 6),
            "recall_macro": round(float(recall_macro), 6),
            "f1_macro": round(float(f1_macro), 6),
            "precision_weighted": round(float(precision_weighted), 6),
            "recall_weighted": round(float(recall_weighted), 6),
            "f1_weighted": round(float(f1_weighted), 6),
        },
        "note": (
            "Metrik berasal dari prediksi aktual model pada test set independen. "
            "Bukan dari training/validation set."
        ),
    }
    summary_path = output_dir / Path(EVAL_SUMMARY_JSON).name
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Evaluation summary: %s", summary_path)

    # --- Simpan classification_report.csv ---
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = output_dir / Path(EVAL_REPORT_CSV).name
    report_df.to_csv(report_csv_path, encoding="utf-8")
    log.info("Classification report CSV: %s", report_csv_path)

    # --- Simpan confusion_matrix.csv ---
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = output_dir / Path(EVAL_CM_CSV).name
    cm_df.to_csv(cm_csv_path, encoding="utf-8")
    log.info("Confusion matrix CSV: %s", cm_csv_path)

    # --- Simpan confusion_matrix.png ---
    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes - 1)))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prediksi", fontsize=12)
    ax.set_ylabel("Aktual", fontsize=12)
    plt.tight_layout()
    cm_png_path = output_dir / Path(EVAL_CM_PNG).name
    fig.savefig(str(cm_png_path), dpi=150)
    plt.close(fig)
    log.info("Confusion matrix PNG: %s", cm_png_path)

    # --- Simpan test_predictions.csv ---
    correct = (y_pred == y_test).astype(int)
    pred_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_test)),
            "true_label_idx": y_test,
            "true_label_name": [idx_to_label[i] for i in y_test],
            "pred_label_idx": y_pred,
            "pred_label_name": [idx_to_label[i] for i in y_pred],
            "correct": correct,
            **{f"prob_{cls}": probs[:, i] for i, cls in enumerate(class_names)},
        }
    )
    pred_csv_path = output_dir / Path(EVAL_PREDICTIONS_CSV).name
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8")
    log.info("Test predictions CSV: %s", pred_csv_path)

    log.info("=== Evaluasi selesai ===")


if __name__ == "__main__":
    main()
