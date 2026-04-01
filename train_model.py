"""Train a CNN-LSTM model for Kolok sentence classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM model from processed sequence dataset.")
    parser.add_argument("--dataset", type=str, default="data/processed/dataset_sequences.npz")
    parser.add_argument("--output-model", type=str, default="models/cnn_lstm_kolok.keras")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def build_cnn_lstm(seq_len: int, n_features: int, n_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(seq_len, n_features), name="sequence_input")

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="class_output")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_kolok")
    return model


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset, allow_pickle=True)

    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_train = data["y_train"].astype(np.int32)
    y_val = data["y_val"].astype(np.int32)
    labels = data["labels"]

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    n_classes = len(labels)

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

    model = build_cnn_lstm(seq_len, n_features, n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=str(output_model), monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=1,
    )

    final_loss, final_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"Validation loss: {final_loss:.4f}")
    print(f"Validation acc : {final_acc:.4f}")

    hist_path = output_model.with_suffix(".history.npz")
    np.savez_compressed(hist_path, **history.history)
    print(f"Model terbaik tersimpan: {output_model}")
    print(f"History tersimpan      : {hist_path}")


if __name__ == "__main__":
    main()
