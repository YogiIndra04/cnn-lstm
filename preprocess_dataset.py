"""Preprocess landmark sequences: resize, normalize, split, and save dataset artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from kolok_utils import N_FEATURES, resize_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized sequence dataset from landmark arrays.")
    parser.add_argument("--input-dir", type=str, default="data/landmarks", help="Directory with .npy files per label.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to store processed dataset and scalers.")
    parser.add_argument("--seq-len", type=int, default=30, help="Fixed number of frames per sequence.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--augment-swap-hands",
        dest="augment_swap_hands",
        action="store_true",
        default=True,
        help="Augment data by swapping Left/Right hand feature slots.",
    )
    parser.add_argument(
        "--no-augment-swap-hands",
        dest="augment_swap_hands",
        action="store_false",
        help="Disable left/right hand swap augmentation.",
    )
    return parser.parse_args()


def swap_hand_slots(sequence: np.ndarray) -> np.ndarray:
    """Swap Left-hand and Right-hand feature slices (42 + 42)."""
    half = N_FEATURES // 2
    return np.concatenate([sequence[:, half:], sequence[:, :half]], axis=1)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"Tidak ada file .npy di {input_dir}")

    sequences = []
    labels = []

    for file_path in npy_files:
        label = file_path.parent.name
        seq = np.load(file_path)

        if seq.ndim != 2 or seq.shape[1] != N_FEATURES:
            print(f"Lewati {file_path} karena shape tidak valid: {seq.shape}")
            continue

        seq_fixed = resize_sequence(seq, args.seq_len)
        sequences.append(seq_fixed)
        labels.append(label)

        if args.augment_swap_hands:
            sequences.append(swap_hand_slots(seq_fixed))
            labels.append(label)

    if not sequences:
        raise RuntimeError("Tidak ada sequence valid setelah filtering.")

    X = np.array(sequences, dtype=np.float32)
    y_text = np.array(labels)

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    n_samples = len(y)
    n_classes = len(le.classes_)
    class_counts = np.bincount(y)

    if n_samples < 2:
        raise RuntimeError("Dataset terlalu kecil. Minimal butuh 2 sampel untuk split train/val.")

    test_count = max(1, int(np.ceil(n_samples * args.test_size)))
    can_stratify = bool(
        n_classes >= 2
        and test_count >= n_classes
        and np.all(class_counts >= 2)
    )

    if can_stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
    else:
        print(
            "Peringatan: stratified split dinonaktifkan karena dataset kecil/tidak seimbang. "
            "Menjalankan split biasa (tanpa stratify)."
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=None,
        )

    # Hindari data leakage: scaler hanya fit pada train split.
    scaler = MinMaxScaler()
    X_train_2d = X_train.reshape(-1, N_FEATURES)
    X_val_2d = X_val.reshape(-1, N_FEATURES)
    X_train = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val = scaler.transform(X_val_2d).reshape(X_val.shape)

    dataset_path = output_dir / "dataset_sequences.npz"
    np.savez_compressed(
        dataset_path,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        labels=le.classes_,
        seq_len=args.seq_len,
    )

    joblib.dump(scaler, output_dir / "minmax_scaler.pkl")
    joblib.dump(le, output_dir / "label_encoder.pkl")

    print(f"Dataset tersimpan: {dataset_path}")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"Jumlah kelas: {len(le.classes_)} -> {list(le.classes_)}")
    print(f"Augment swap hands: {args.augment_swap_hands}")


if __name__ == "__main__":
    main()
