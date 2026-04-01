"""Shared utilities for the Kolok sentence recognition prototype."""

from __future__ import annotations

from pathlib import Path
from typing import List

import mediapipe as mp
import numpy as np

DEFAULT_LABELS = [
    "saya makan",
    "saya pergi",
    "kamu datang",
    "terima kasih",
    "apa kabar",
]

N_HANDS = 2
N_LANDMARKS = 21
N_FEATURES = N_HANDS * N_LANDMARKS * 2


def create_hands_detector(
    max_num_hands: int = N_HANDS,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create a configured MediaPipe Hands detector."""
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_xy_landmarks(results) -> np.ndarray:
    """Return flattened (x, y) landmarks with fixed shape (84,).

    The first 42 values are the Left hand (if detected), next 42 are Right hand.
    If only one hand is visible, the missing hand slice stays zero.
    """
    output = np.zeros(N_FEATURES, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return output

    handed_entries = []
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:N_HANDS]):
        if results.multi_handedness and idx < len(results.multi_handedness):
            label = results.multi_handedness[idx].classification[0].label
        else:
            label = f"Unknown_{idx}"
        handed_entries.append((label, hand_landmarks))

    # Keep a stable hand order across frames: Left then Right.
    handed_entries.sort(key=lambda item: (0 if item[0] == "Left" else 1, item[0]))

    cursor = 0
    for _, hand_landmarks in handed_entries:
        for lm in hand_landmarks.landmark:
            output[cursor] = lm.x
            output[cursor + 1] = lm.y
            cursor += 2
    return output


def resize_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Resize sequence to fixed length via interpolation indices + zero padding if needed."""
    n_frames = len(sequence)
    if n_frames == 0:
        return np.zeros((target_len, N_FEATURES), dtype=np.float32)

    if n_frames >= target_len:
        idx = np.linspace(0, n_frames - 1, target_len).astype(np.int32)
        return sequence[idx]

    pad_len = target_len - n_frames
    pad = np.zeros((pad_len, sequence.shape[1]), dtype=np.float32)
    return np.vstack([sequence, pad])


def discover_labels_from_videos(raw_video_dir: str | Path) -> List[str]:
    """Discover labels from directory names in raw video folder."""
    base = Path(raw_video_dir)
    labels = sorted([d.name for d in base.iterdir() if d.is_dir()]) if base.exists() else []
    return labels if labels else DEFAULT_LABELS
