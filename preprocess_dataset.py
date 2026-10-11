"""
Preprocessing video dataset Bahasa Kolok untuk model CNN-LSTM.

Alur:
  video MP4 → 30 frame RGB 300×300 → normalisasi float32 [0,1]
  → pembagian train/val/test → dataset_sequences.npz + label_map.json

Jalankan dari root proyek:
    python preprocess_dataset.py [--input-dir ...] [--output-dir ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.model_config import (
    CHANNELS,
    COLOR_MODE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_FRAMES,
    PREPROCESS_REPORT_JSON,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_VIDEO_DIR,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
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

VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Fungsi Pembantu Preprocessing
# ---------------------------------------------------------------------------

def sample_frame_indices(total_frames: int, target_len: int) -> np.ndarray:
    """Sampling target_len indeks frame temporal merata dari total_frames."""
    if total_frames <= 0:
        raise ValueError(f"total_frames harus > 0, diperoleh {total_frames}")
    if total_frames == 1:
        return np.zeros(target_len, dtype=np.int32)
    indices = np.linspace(0, total_frames - 1, target_len)
    return np.round(indices).astype(np.int32)


def preprocess_single_frame(frame_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Konversi format kanal warna ke RGB dan resize ke (width, height). Output uint8 [0, 255]."""
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if (frame_bgr.shape[0] >= height and frame_bgr.shape[1] >= width) else cv2.INTER_LINEAR
    frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=interp)
    return frame_resized.astype(np.uint8)


def validate_sequence(sequence: np.ndarray, num_frames: int, height: int, width: int) -> None:
    """Validasi dimensi, tipe data, dan isi tensor sequence."""
    expected_shape = (num_frames, height, width, 3)
    assert sequence.shape == expected_shape, (
        f"Bentuk tensor salah: {sequence.shape} (diharapkan {expected_shape})"
    )
    assert sequence.dtype == np.uint8, (
        f"dtype salah: {sequence.dtype} (diharapkan uint8)"
    )
    assert not np.all(sequence == 0), (
        "Sequence tidak boleh seluruhnya nol (video kosong atau gagal dibaca)"
    )


def process_video(
    video_path: Path | str,
    num_frames: int = NUM_FRAMES,
    height: int = IMAGE_HEIGHT,
    width: int = IMAGE_WIDTH,
) -> tuple[np.ndarray, dict]:
    """Baca dan proses satu video menjadi sequence (num_frames, height, width, 3) uint8."""
    vpath = Path(video_path)
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {vpath}")

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError(f"Tidak dapat membaca frame dari video: {vpath}")

    repeated = total_frames < num_frames
    indices = sample_frame_indices(total_frames, num_frames)

    processed_frames: list[np.ndarray] = []
    for idx in indices:
        pf = preprocess_single_frame(frames[idx], height, width)
        processed_frames.append(pf)

    seq = np.stack(processed_frames, axis=0).astype(np.uint8)
    validate_sequence(seq, num_frames, height, width)

    info = {
        "status": "ok",
        "repeated": repeated,
        "original_frames": total_frames,
        "reason": "",
        "path": str(vpath),
    }
    return seq, info


def build_label_map(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Bangun mapping label unik secara deterministik (alfabetis)."""
    unique_sorted = sorted(list(set(labels)))
    lbl2idx = {lbl: i for i, lbl in enumerate(unique_sorted)}
    idx2lbl = {i: lbl for i, lbl in enumerate(unique_sorted)}
    return lbl2idx, idx2lbl


def discover_videos(root_dir: Path | str) -> list[tuple[Path, str]]:
    """Pindai direktori root untuk subfolder kelas dan file video valid."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Direktori root tidak ditemukan: {root}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"Tidak ada subfolder kelas di {root}")

    samples: list[tuple[Path, str]] = []
    for cdir in class_dirs:
        class_name = cdir.name
        vfiles = sorted([f for f in cdir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
        if not vfiles:
            log.warning("Kelas '%s' tidak memiliki file video valid.", class_name)
            print(f"Peringatan: Kelas '{class_name}' tidak memiliki video.")
        for vf in vfiles:
            samples.append((vf, class_name))

    return samples


# ---------------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw RGB videos to 300x300 image sequences for CNN2D-LSTM."
    )
    parser.add_argument("--input-dir", type=str, default=RAW_VIDEO_DIR, help="Direktori root raw_videos.")
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DIR, help="Direktori output data/processed.")
    parser.add_argument("--splits-dir", type=str, default=SPLITS_DIR, help="Direktori output data/splits.")
    parser.add_argument("--report-path", type=str, default=PREPROCESS_REPORT_JSON, help="Path laporan preprocessing.")
    parser.add_argument("--dry-run", action="store_true", help="Pindai dataset tanpa memproses frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)
    report_path = Path(args.report_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Memindai dataset di: %s", input_dir)
    samples = discover_videos(input_dir)
    if not samples:
        log.error("Tidak ada video yang ditemukan di: %s", input_dir)
        sys.exit(1)

    all_labels = [lbl for _, lbl in samples]
    lbl2idx, idx2lbl = build_label_map(all_labels)
    classes = sorted(list(lbl2idx.keys()))
    n_classes = len(classes)
    log.info("Ditemukan %d video dari %d kelas: %s", len(samples), n_classes, classes)

    if args.dry_run:
        log.info("Mode --dry-run selesai. Tidak ada pemrosesan video.")
        return

    sequences: list[np.ndarray] = []
    labels_int: list[int] = []
    preprocessing_log: list[dict] = []
    failed_samples: list[dict] = []

    for idx, (vpath, label) in enumerate(samples, start=1):
        try:
            seq, info = process_video(vpath, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
            sequences.append(seq)
            labels_int.append(lbl2idx[label])
            info["label"] = label
            preprocessing_log.append(info)
            log.info(
                "[%3d/%3d] OK  %s | label=%s | frame=%d→%d",
                idx, len(samples), vpath.name, label, info["original_frames"], NUM_FRAMES
            )
        except Exception as e:
            log.error("[%3d/%3d] GAGAL %s | error: %s", idx, len(samples), vpath.name, e)
            failed_info = {"path": str(vpath), "label": label, "error": str(e)}
            failed_samples.append(failed_info)

    if not sequences:
        log.error("Tidak ada video yang berhasil diproses.")
        sys.exit(1)

    X = np.stack(sequences, axis=0)  # uint8 (N, 30, 300, 300, 3)
    y = np.array(labels_int, dtype=np.int32)
    n_samples = len(X)
    log.info("Bentuk X sebelum split: %s dtype=%s", X.shape, X.dtype)

    # Stratified split: Train (70%), Val (15%), Test (15%)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    indices = np.arange(n_samples)

    train_idx, val_test_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    test_share = TEST_RATIO / val_test_ratio
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=test_share,
        random_state=RANDOM_SEED,
        stratify=y[val_test_idx],
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    log.info("Split: train=%d | val=%d | test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Simpan split manifest
    split_manifest = {
        "random_seed": RANDOM_SEED,
        "train_indices": [int(i) for i in train_idx],
        "val_indices": [int(i) for i in val_idx],
        "test_indices": [int(i) for i in test_idx],
    }
    split_manifest_path = splits_dir / "split_manifest.json"
    with split_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    log.info("Manifest split tersimpan: %s", split_manifest_path)

    # Simpan dataset npz
    dataset_path = output_dir / "dataset_sequences.npz"
    np.savez_compressed(
        dataset_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=np.array(classes),
    )
    log.info("Dataset tersimpan: %s", dataset_path)
    log.info(
        "X_train=%s | X_val=%s | X_test=%s | dtype=%s",
        X_train.shape, X_val.shape, X_test.shape, X_train.dtype
    )

    # Simpan label_map.json
    label_map = {
        "label_to_idx": lbl2idx,
        "idx_to_label": {str(k): v for k, v in idx2lbl.items()},
        "classes": classes,
        "n_classes": n_classes,
        "preprocessing": {
            "num_frames": NUM_FRAMES,
            "img_height": IMAGE_HEIGHT,
            "img_width": IMAGE_WIDTH,
            "channels": CHANNELS,
            "color_mode": COLOR_MODE,
            "normalization": "divide_by_255_float32",
            "input_shape_per_sample": [NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS],
        },
    }
    label_map_path = output_dir / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    log.info("Label map tersimpan: %s", label_map_path)

    # Simpan preprocessing_report.json
    report_data = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_videos_found": len(samples),
        "videos_ok": len(sequences),
        "videos_failed": len(failed_samples),
        "n_classes": n_classes,
        "classes": classes,
        "split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "config": {
            "num_frames": NUM_FRAMES,
            "img_height": IMAGE_HEIGHT,
            "img_width": IMAGE_WIDTH,
            "channels": CHANNELS,
            "color_mode": COLOR_MODE,
        },
        "failed_samples": failed_samples,
        "preprocessing_log": preprocessing_log,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    log.info("Laporan preprocessing tersimpan: %s", report_path)
    log.info("=== Preprocessing selesai ===")


if __name__ == "__main__":
    main()
