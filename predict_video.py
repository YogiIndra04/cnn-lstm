"""
Inferensi: prediksi kata Bahasa Kolok dari satu video.

Fungsi `predict_video()` dapat diimpor dan digunakan kembali di modul lain.
Preprocessing identik dengan pipeline training (raw RGB, 300×300, 30 frame).

Jalankan dari root proyek:
    python predict_video.py --video PATH_VIDEO [--model ...] [--label-map ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.model_config import (
    BEST_MODEL_PATH,
    CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL_MAP_JSON,
    NUM_FRAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Preprocessing (identik dengan training)
# ---------------------------------------------------------------------------

def _sample_frame_indices(total_frames: int, target_len: int) -> np.ndarray:
    indices = np.linspace(0, total_frames - 1, target_len)
    return np.round(indices).astype(np.int32)


def _preprocess_frame_bgr(frame_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Konversi BGR→RGB, resize. Mengembalikan float32 [0,1] (H, W, 3)."""
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return frame_resized.astype(np.float32) / 255.0


def preprocess_video_for_inference(
    video_path: str | Path,
    num_frames: int = NUM_FRAMES,
    height: int = IMAGE_HEIGHT,
    width: int = IMAGE_WIDTH,
) -> np.ndarray:
    """
    Baca satu video dan hasilkan tensor siap inferensi.

    Preprocessing identik dengan training:
    - raw RGB (tanpa MediaPipe)
    - 30 frame tersebar merata temporal
    - resize 300×300
    - normalisasi [0, 1]

    Returns:
        np.ndarray shape (1, num_frames, height, width, 3) float32 — batch size 1.

    Raises:
        FileNotFoundError: video tidak ditemukan.
        ValueError: format tidak didukung atau video tidak dapat dibaca.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    if video_path.suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError(
            f"Format video tidak didukung: {video_path.suffix}. "
            f"Format yang didukung: {VALID_EXTENSIONS}"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Tidak dapat membuka video: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is not None and frame.size > 0:
            frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"Video tidak memiliki frame yang dapat dibaca: {video_path}")

    n_raw = len(frames)
    if n_raw < num_frames:
        log.warning(
            "Video hanya memiliki %d frame (target %d). Frame akan diulang.",
            n_raw, num_frames,
        )

    indices = _sample_frame_indices(n_raw, num_frames)
    seq = np.zeros((num_frames, height, width, 3), dtype=np.float32)

    for t, src_idx in enumerate(indices):
        seq[t] = _preprocess_frame_bgr(frames[int(src_idx)], height, width)

    # Validasi
    assert seq.shape == (num_frames, height, width, 3), f"Shape tensor salah: {seq.shape}"
    assert seq.min() >= 0.0 and seq.max() <= 1.0, "Nilai piksel di luar [0, 1]"

    return np.expand_dims(seq, axis=0)  # (1, T, H, W, C)


# ---------------------------------------------------------------------------
# Prediksi
# ---------------------------------------------------------------------------

def predict_video(
    video_path: str | Path,
    model,  # tf.keras.Model — sudah dimuat
    idx_to_label: dict[int, str],
    top_k: int = 3,
) -> dict:
    """
    Prediksi kata Bahasa Kolok dari satu video.

    Args:
        video_path: path ke file video.
        model: model Keras yang sudah dimuat (muat sekali untuk banyak inferensi).
        idx_to_label: mapping indeks kelas → nama kelas (dari label_map.json).
        top_k: jumlah prediksi teratas yang dikembalikan.

    Returns:
        dict dengan kunci:
          - predicted_label: nama kelas dengan probabilitas tertinggi.
          - confidence: probabilitas tertinggi (float 0–1).
          - top_k: list dict {rank, label, probability}.
          - video_path: path video sumber.
          - status: "ok" atau pesan error.
    """
    # Validasi kompatibilitas label map dengan model
    n_model_classes = model.output_shape[-1]
    n_map_classes = len(idx_to_label)
    if n_model_classes != n_map_classes:
        raise ValueError(
            f"Jumlah kelas model ({n_model_classes}) tidak cocok dengan "
            f"label map ({n_map_classes}). Pastikan label map yang digunakan "
            "berasal dari training yang sama dengan model."
        )

    x = preprocess_video_for_inference(str(video_path))
    # x shape: (1, NUM_FRAMES, H, W, 3)

    probs = model.predict(x, verbose=0)[0]  # (n_classes,)

    # Sanity check probabilitas Softmax
    prob_sum = float(np.sum(probs))
    if abs(prob_sum - 1.0) > 0.01:
        log.warning(
            "Jumlah probabilitas Softmax: %.4f (diharapkan ≈ 1.0). "
            "Mungkin ada masalah pada model.",
            prob_sum,
        )

    sorted_idx = np.argsort(probs)[::-1]
    top_k_results = [
        {
            "rank": r + 1,
            "label": idx_to_label[int(sorted_idx[r])],
            "probability": round(float(probs[sorted_idx[r]]), 6),
        }
        for r in range(min(top_k, len(sorted_idx)))
    ]

    return {
        "predicted_label": top_k_results[0]["label"],
        "confidence": top_k_results[0]["probability"],
        "top_k": top_k_results,
        "video_path": str(video_path),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediksi kata Bahasa Kolok dari satu file video."
    )
    parser.add_argument("--video", type=str, required=True, help="Path ke file video.")
    parser.add_argument(
        "--model", type=str, default=BEST_MODEL_PATH, help="Path ke model .keras."
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=LABEL_MAP_JSON,
        help="Path ke label_map.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Jumlah prediksi teratas yang ditampilkan.",
    )
    return parser.parse_args()


def main() -> None:
    import tensorflow as tf

    args = parse_args()

    model_path = Path(args.model)
    label_map_path = Path(args.label_map)

    if not model_path.exists():
        log.error("Model tidak ditemukan: %s", model_path)
        sys.exit(1)
    if not label_map_path.exists():
        log.error("Label map tidak ditemukan: %s. Jalankan preprocessing terlebih dahulu.", label_map_path)
        sys.exit(1)

    # Muat label map
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map_data = json.load(f)
    idx_to_label = {int(k): v for k, v in label_map_data["idx_to_label"].items()}

    # Verifikasi preprocessing config di label map cocok dengan model config
    preproc = label_map_data.get("preprocessing", {})
    if preproc:
        mismatch = []
        if preproc.get("num_frames") != NUM_FRAMES:
            mismatch.append(f"num_frames: label_map={preproc.get('num_frames')}, config={NUM_FRAMES}")
        if preproc.get("img_height") != IMAGE_HEIGHT:
            mismatch.append(f"img_height: label_map={preproc.get('img_height')}, config={IMAGE_HEIGHT}")
        if preproc.get("img_width") != IMAGE_WIDTH:
            mismatch.append(f"img_width: label_map={preproc.get('img_width')}, config={IMAGE_WIDTH}")
        if mismatch:
            log.error(
                "Ketidaksesuaian preprocessing antara label_map.json dan configs/model_config.py:\n  %s\n"
                "Label map ini mungkin dari sesi preprocessing berbeda.",
                "\n  ".join(mismatch),
            )
            sys.exit(1)

    # Muat model (sekali saja)
    log.info("Memuat model: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))

    # Prediksi
    log.info("Memproses video: %s", args.video)
    try:
        result = predict_video(
            video_path=args.video,
            model=model,
            idx_to_label=idx_to_label,
            top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError) as e:
        log.error("Gagal memproses video: %s", e)
        sys.exit(1)

    print("\n=== Hasil Prediksi ===")
    print(f"Video      : {result['video_path']}")
    print(f"Prediksi   : {result['predicted_label']}")
    print(f"Confidence : {result['confidence']:.4f}")
    print(f"\nTop-{args.top_k} prediksi:")
    for entry in result["top_k"]:
        print(f"  {entry['rank']}. {entry['label']:<20} {entry['probability']:.4f}")
    print()


if __name__ == "__main__":
    main()
