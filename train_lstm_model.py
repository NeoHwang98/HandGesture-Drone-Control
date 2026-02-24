import argparse
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train dynamic LSTM gesture model from data/gesture_sequence_data.npz"
    )
    parser.add_argument("--data-npz", default="data/gesture_sequence_data.npz")
    parser.add_argument("--model-out", default="models/dynamic/updated_gesture_lstm1.keras")
    parser.add_argument("--label-encoder-out", default="models/dynamic/updated_label_encoder_lstm1.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--expected-labels",
        nargs="*",
        default=["circle_cw", "circle_ccw", "circle_nil"],
        help="Optional expected dynamic labels to validate against.",
    )
    return parser.parse_args()


def load_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError("NPZ must contain arrays named 'X' and 'y'.")
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"]).astype(str)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F). Got {X.shape}")
    if len(X) != len(y):
        raise ValueError("X and y must have matching sample counts.")
    if X.shape[2] != 42:
        raise ValueError(f"Expected feature size 42 (21 x + 21 y). Got {X.shape[2]}")
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 dynamic classes to train.")
    return X, y


def build_model(timesteps, features, n_classes, args):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, features)),
            tf.keras.layers.LSTM(args.lstm_units),
            tf.keras.layers.Dense(args.dense_units, activation="relu"),
            tf.keras.layers.Dropout(args.dropout),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    data_path = Path(args.data_npz)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    X, y_raw = load_dataset(data_path)
    timesteps, features = X.shape[1], X.shape[2]

    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_int, num_classes=n_classes)

    missing = sorted(set(args.expected_labels) - set(le.classes_))
    if missing:
        print(f"WARNING: Expected labels missing from dataset: {missing}")

    X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
        X,
        y_cat,
        y_int,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_int,
    )

    model = build_model(timesteps, features, n_classes, args)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Classes: {list(le.classes_)}")
    print(f"Input shape: {model.input_shape}, output shape: {model.output_shape}")
    print(f"Epochs run: {len(history.history['loss'])}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.label_encoder_out).parent.mkdir(parents=True, exist_ok=True)

    model.save(args.model_out)
    with open(args.label_encoder_out, "wb") as f:
        pickle.dump(le, f)

    print(f"Saved model: {args.model_out}")
    print(f"Saved label encoder: {args.label_encoder_out}")


if __name__ == "__main__":
    main()
