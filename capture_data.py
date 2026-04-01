"""Capture gesture videos per sentence label using webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Kolok gesture videos from webcam.")
    parser.add_argument("--label", type=str, required=True, help="Sentence label, e.g. 'saya makan'.")
    parser.add_argument("--output-dir", type=str, default="data/raw_videos", help="Root output directory.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of clips to record.")
    parser.add_argument("--duration", type=float, default=3.0, help="Clip duration in seconds.")
    parser.add_argument("--fps", type=int, default=20, help="Recording FPS.")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    return "_".join(label.lower().strip().split())


def countdown(window_name: str, frame, seconds: int = 3):
    for sec in range(seconds, 0, -1):
        canvas = frame.copy()
        cv2.putText(canvas, f"Recording starts in {sec}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1000)


def main() -> None:
    args = parse_args()

    label_dir = Path(args.output_dir) / sanitize_label(args.label)
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Webcam tidak bisa dibuka.")

    window_name = "Kolok Data Capture"
    frame_interval = 1.0 / max(1, args.fps)

    print("Instruksi:")
    print("- Tekan SPACE untuk mulai rekam clip.")
    print("- Tekan Q untuk keluar.")

    sample_idx = 1
    while sample_idx <= args.num_samples:
        ok, frame = cap.read()
        if not ok:
            print("Frame gagal dibaca. Menghentikan capture.")
            break

        display = frame.copy()
        cv2.putText(display, f"Label: {args.label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"Sample: {sample_idx}/{args.num_samples}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, "SPACE: record | Q: quit", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key != ord(" "):
            continue

        countdown(window_name, frame, seconds=3)

        video_path = label_dir / f"sample_{sample_idx:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (frame.shape[1], frame.shape[0]))

        start_time = time.time()
        while (time.time() - start_time) < args.duration:
            ok, rec_frame = cap.read()
            if not ok:
                break
            writer.write(rec_frame)

            rec_display = rec_frame.copy()
            remaining = max(0.0, args.duration - (time.time() - start_time))
            cv2.putText(rec_display, f"REC {remaining:0.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.imshow(window_name, rec_display)
            cv2.waitKey(max(1, int(frame_interval * 1000)))

        writer.release()
        print(f"Tersimpan: {video_path}")
        sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
