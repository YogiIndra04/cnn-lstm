"""Extract MediaPipe hand landmarks from gesture videos and save as NumPy sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from kolok_utils import N_FEATURES, create_hands_detector, extract_xy_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hand landmarks from recorded videos.")
    parser.add_argument("--input-dir", type=str, default="data/raw_videos", help="Directory containing label subfolders with videos.")
    parser.add_argument("--output-dir", type=str, default="data/landmarks", help="Directory for .npy landmark sequences.")
    parser.add_argument(
        "--max-num-hands",
        type=int,
        default=2,
        choices=[1, 2],
        help="Default 2 (otomatis bisa 1 atau 2 tangan per frame).",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def extract_video_landmarks(video_path: Path, hands) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    sequence = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        features = extract_xy_landmarks(results)
        sequence.append(features)

    cap.release()
    if not sequence:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.array(sequence, dtype=np.float32)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
    if not video_paths:
        raise FileNotFoundError(f"Tidak ada video ditemukan di: {input_dir}")

    hands = create_hands_detector(
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    total = len(video_paths)
    for idx, video_path in enumerate(video_paths, start=1):
        label = video_path.parent.name
        seq = extract_video_landmarks(video_path, hands)

        save_dir = output_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{video_path.stem}.npy"
        np.save(out_path, seq)

        print(f"[{idx}/{total}] {video_path.name} -> {out_path} shape={seq.shape}")

    hands.close()
    print("Ekstraksi landmark selesai.")


if __name__ == "__main__":
    main()
