"""Real-time webcam inference for Kolok sentence recognition."""

from __future__ import annotations

import argparse
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf

from kolok_utils import create_hands_detector, extract_xy_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time inference using webcam.")
    parser.add_argument("--model", type=str, default="models/cnn_lstm_kolok.keras")
    parser.add_argument("--scaler", type=str, default="data/processed/minmax_scaler.pkl")
    parser.add_argument("--label-encoder", type=str, default="data/processed/label_encoder.pkl")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument(
        "--max-num-hands",
        type=int,
        default=2,
        choices=[1, 2],
        help="Default 2 (otomatis bisa 1 atau 2 tangan per frame).",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--predict-every", type=int, default=3, help="Predict every N frames when buffer full.")
    parser.add_argument("--min-margin", type=float, default=0.08, help="Minimum gap between top-1 and top-2 to avoid ambiguous label.")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Average probabilities over last N predictions.")
    parser.add_argument("--no-hand-reset-frames", type=int, default=15, help="Clear temporal buffers after N consecutive frames without detected hands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = tf.keras.models.load_model(args.model)
    scaler = joblib.load(args.scaler)
    label_encoder = joblib.load(args.label_encoder)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak bisa dibuka.")

    hands = create_hands_detector(max_num_hands=args.max_num_hands)
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    buffer = deque(maxlen=args.seq_len)
    prob_history = deque(maxlen=max(1, args.smoothing_window))
    frame_count = 0
    no_hand_streak = 0
    predicted_text = "-"
    predicted_conf = 0.0
    predicted_margin = 0.0
    top2_text = "-"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        feature = extract_xy_landmarks(results)
        buffer.append(feature)
        frame_count += 1
        hand_detected = bool(results.multi_hand_landmarks)

        if hand_detected:
            no_hand_streak = 0
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            no_hand_streak += 1
            if no_hand_streak >= args.no_hand_reset_frames:
                buffer.clear()
                prob_history.clear()
                predicted_text = "-"
                predicted_conf = 0.0
                predicted_margin = 0.0
                top2_text = "-"
                no_hand_streak = 0

        if hand_detected and len(buffer) == args.seq_len and (frame_count % args.predict_every == 0):
            sequence = np.array(buffer, dtype=np.float32)
            sequence_scaled = scaler.transform(sequence)
            x = np.expand_dims(sequence_scaled, axis=0)
            probs = model.predict(x, verbose=0)[0]
            prob_history.append(probs)
            probs_smoothed = np.mean(np.stack(prob_history), axis=0)

            sorted_idx = np.argsort(probs_smoothed)[::-1]
            class_idx = int(sorted_idx[0])
            second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else class_idx

            conf = float(probs_smoothed[class_idx])
            second_conf = float(probs_smoothed[second_idx])
            margin = conf - second_conf

            predicted_conf = conf
            predicted_margin = margin
            top1_label = str(label_encoder.inverse_transform([class_idx])[0])
            top2_label = str(label_encoder.inverse_transform([second_idx])[0])
            top2_text = f"{top1_label}:{conf:.2f} | {top2_label}:{second_conf:.2f}"

            if conf < args.confidence_threshold:
                predicted_text = "Tidak yakin"
            elif margin < args.min_margin:
                predicted_text = "Ambigu"
            else:
                predicted_text = top1_label

        cv2.putText(frame, f"Prediksi: {predicted_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {predicted_conf:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Margin: {predicted_margin:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(frame, f"Top2: {top2_text}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "Press Q to quit", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        cv2.imshow("Kolok Real-time Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
