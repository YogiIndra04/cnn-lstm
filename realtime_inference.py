"""
Inferensi real-time webcam untuk pengenalan kata Bahasa Kolok menggunakan CNN2D-LSTM.

Input model: raw RGB 300×300 (konsisten dengan preprocessing training).
Tidak menggunakan MediaPipe atau representasi landmark.

Jalankan dari root proyek:
    python realtime_inference.py [--model ...] [--label-map ...]

Tekan Q untuk keluar.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time webcam inference CNN2D-LSTM Bahasa Kolok (raw RGB)."
    )
    parser.add_argument("--model", type=str, default=BEST_MODEL_PATH, help="Path ke file model .keras.")
    parser.add_argument("--label-map", type=str, default=LABEL_MAP_JSON, help="Path ke label_map.json.")
    parser.add_argument("--seq-len", type=int, default=NUM_FRAMES, help="Jumlah frame per sequence window.")
    parser.add_argument("--img-height", type=int, default=IMAGE_HEIGHT, help="Tinggi frame input model.")
    parser.add_argument("--img-width", type=int, default=IMAGE_WIDTH, help="Lebar frame input model.")
    parser.add_argument("--camera-id", type=int, default=0, help="Device ID webcam.")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="Ambang batas kepercayaan minimal.")
    parser.add_argument("--predict-every", type=int, default=5, help="Prediksi setiap N frame saat buffer penuh.")
    parser.add_argument("--min-margin", type=float, default=0.08, help="Batas selisih top-1 vs top-2.")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Rata-rata probabilitas N prediksi terakhir.")
    return parser.parse_args()


def preprocess_frame_for_buffer(frame_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Preprocess satu frame webcam ke format input model: RGB 300x300 float32 [0, 1]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        log.error("Model tidak ditemukan di: %s. Pastikan model sudah dilatih.", model_path)
        sys.exit(1)

    label_map_path = Path(args.label_map)
    if not label_map_path.exists():
        log.error("Label map tidak ditemukan di: %s.", label_map_path)
        sys.exit(1)

    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx_to_label: dict[int, str] = {
        int(k): v for k, v in label_map.get("idx_to_label", {}).items()
    }
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    n_classes = len(class_names)
    log.info("Memuat model dari: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))
    log.info("Model berhasil dimuat. Kelas (%d): %s", n_classes, class_names)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        log.error("Webcam (id=%d) tidak dapat dibuka.", args.camera_id)
        sys.exit(1)

    buffer: deque[np.ndarray] = deque(maxlen=args.seq_len)
    prob_history: deque[np.ndarray] = deque(maxlen=max(1, args.smoothing_window))

    frame_count = 0
    predicted_text = "-"
    predicted_conf = 0.0
    predicted_margin = 0.0
    top2_text = "-"

    window_name = "Bahasa Kolok RGB Real-time Inference"
    log.info("Memulai inferensi real-time webcam. Tekan Q pada jendela webcam untuk keluar.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Gagal membaca frame webcam.")
            break

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
        prev_time = curr_time

        # Preprocess frame untuk buffer RGB
        processed_frame = preprocess_frame_for_buffer(frame, args.img_height, args.img_width)
        buffer.append(processed_frame)
        frame_count += 1

        # Jalankan prediksi setiap `predict_every` frame saat buffer sudah terisi 30 frame
        if len(buffer) == args.seq_len and (frame_count % args.predict_every == 0):
            input_tensor = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)  # (1, 30, 300, 300, 3)
            raw_probs = model.predict(input_tensor, verbose=0)[0]
            prob_history.append(raw_probs)
            smoothed_probs = np.mean(np.stack(prob_history), axis=0)

            sorted_idx = np.argsort(smoothed_probs)[::-1]
            top1_idx = int(sorted_idx[0])
            top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx

            conf1 = float(smoothed_probs[top1_idx])
            conf2 = float(smoothed_probs[top2_idx])
            margin = conf1 - conf2

            top1_lbl = idx_to_label.get(top1_idx, f"Kelas_{top1_idx}")
            top2_lbl = idx_to_label.get(top2_idx, f"Kelas_{top2_idx}")

            predicted_conf = conf1
            predicted_margin = margin
            top2_text = f"{top1_lbl}:{conf1:.2f} | {top2_lbl}:{conf2:.2f}"

            if conf1 < args.confidence_threshold:
                predicted_text = "Tidak yakin"
            elif margin < args.min_margin:
                predicted_text = "Ambigu"
            else:
                predicted_text = top1_lbl

        # Tampilkan informasi pada jendela
        buf_status = f"Buffer: {len(buffer)}/{args.seq_len}"
        cv2.putText(frame, f"Prediksi: {predicted_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {predicted_conf:.2f} (margin: {predicted_margin:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Top-2: {top2_text}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"{buf_status} | FPS: {fps:.1f} | 'Q': Keluar", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Inferensi webcam dihentikan.")


if __name__ == "__main__":
    main()
