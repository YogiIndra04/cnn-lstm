"""Inspect saved ML artifacts (.npz, .pkl, .keras) in this project."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved artifacts and optionally save report to text file.")
    parser.add_argument("--save-txt", type=str, default="", help="Optional path to save inspection report as .txt")
    return parser.parse_args()


def add_line(lines: list[str], text: str = "") -> None:
    lines.append(text)
    print(text)


def inspect_npz(path: Path, lines: list[str]) -> None:
    add_line(lines, f"\n=== NPZ: {path} ===")
    if not path.exists():
        add_line(lines, "File tidak ditemukan.")
        return

    data = np.load(path, allow_pickle=True)
    add_line(lines, f"Keys: {list(data.files)}")
    for key in data.files:
        value = data[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        add_line(lines, f"- {key}: shape={shape}, dtype={dtype}")


def inspect_label_encoder(path: Path, lines: list[str]) -> None:
    add_line(lines, f"\n=== PKL LabelEncoder: {path} ===")
    if not path.exists():
        add_line(lines, "File tidak ditemukan.")
        return

    obj = joblib.load(path)
    classes = getattr(obj, "classes_", None)
    add_line(lines, f"Tipe objek: {type(obj).__name__}")
    add_line(lines, f"Classes: {classes}")


def inspect_scaler(path: Path, lines: list[str]) -> None:
    add_line(lines, f"\n=== PKL Scaler: {path} ===")
    if not path.exists():
        add_line(lines, "File tidak ditemukan.")
        return

    obj = joblib.load(path)
    add_line(lines, f"Tipe objek: {type(obj).__name__}")
    for attr in ("feature_range", "data_min_", "data_max_", "min_", "scale_"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is not None:
            add_line(lines, f"- {attr}: shape={shape}")
        else:
            add_line(lines, f"- {attr}: {value}")


def inspect_model(path: Path, lines: list[str]) -> None:
    add_line(lines, f"\n=== KERAS Model: {path} ===")
    if not path.exists():
        add_line(lines, "File tidak ditemukan.")
        return

    model = tf.keras.models.load_model(path)
    add_line(lines, f"Nama model: {model.name}")
    add_line(lines, f"Input shape: {model.input_shape}")
    add_line(lines, f"Output shape: {model.output_shape}")
    add_line(lines, "Ringkasan layer:")

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    for row in summary_lines:
        add_line(lines, row)


def main() -> None:
    args = parse_args()
    lines: list[str] = []

    inspect_npz(Path("data/processed/dataset_sequences.npz"), lines)
    inspect_scaler(Path("data/processed/minmax_scaler.pkl"), lines)
    inspect_label_encoder(Path("data/processed/label_encoder.pkl"), lines)
    inspect_model(Path("models/cnn_lstm_kolok.keras"), lines)

    if args.save_txt:
        out_path = Path(args.save_txt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nReport tersimpan: {out_path}")


if __name__ == "__main__":
    main()
