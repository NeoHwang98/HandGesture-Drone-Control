import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def compute_relative_features(landmarks_42):
    x_coords = landmarks_42[:21]
    y_coords = landmarks_42[21:]
    wrist_x, wrist_y = x_coords[0], y_coords[0]
    features = []
    for i in range(21):
        features.append(x_coords[i] - wrist_x)
        features.append(y_coords[i] - wrist_y)
    return features


def compute_thumb_index_distance(landmarks_42):
    thumb_tip_x, thumb_tip_y = landmarks_42[8], landmarks_42[9]
    index_tip_x, index_tip_y = landmarks_42[16], landmarks_42[17]
    return np.sqrt((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2)


def build_feature_matrix(df):
    x_cols = [f"x{i}" for i in range(21)]
    y_cols = [f"y{i}" for i in range(21)]
    needed = ["label"] + x_cols + y_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X_out = []
    y_out = df["label"].astype(str).to_numpy()
    for row in df.itertuples(index=False):
        x = [float(getattr(row, c)) for c in x_cols]
        y = [float(getattr(row, c)) for c in y_cols]
        landmarks = x + y
        relative = compute_relative_features(landmarks)
        thumb_idx_dist = compute_thumb_index_distance(landmarks)
        X_out.append(relative + [thumb_idx_dist])

    return np.array(X_out, dtype=np.float32), y_out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train static RandomForest gesture model from data/gesture_data.csv"
    )
    parser.add_argument("--data-csv", default="data/gesture_data.csv")
    parser.add_argument("--model-out", default="models/static/gesture_model2.pkl")
    parser.add_argument("--label-encoder-out", default="models/static/label_encoder2.pkl")
    parser.add_argument("--scaler-out", default="models/static/scaler2.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Input dataset is empty.")

    X, y_raw = build_feature_matrix(df)
    if len(np.unique(y_raw)) < 2:
        raise ValueError("At least 2 gesture classes are required to train a classifier.")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classes:", list(le.classes_))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.label_encoder_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scaler_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.model_out, "wb") as f:
        pickle.dump(model, f)
    with open(args.label_encoder_out, "wb") as f:
        pickle.dump(le, f)
    with open(args.scaler_out, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved model: {args.model_out}")
    print(f"Saved label encoder: {args.label_encoder_out}")
    print(f"Saved scaler: {args.scaler_out}")


if __name__ == "__main__":
    main()
