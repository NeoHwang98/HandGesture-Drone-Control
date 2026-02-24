import argparse
import csv
import os
import time

import cv2
import numpy as np
from hand_landmarker_compat import HandLandmarkerCompat

STATIC_GESTURES_DEFAULT = [
    "up1",
    "down1",
    "left1",
    "right1",
    "open_palm1",
    "fist1",
    "close_palm1",
]

DYNAMIC_GESTURES_DEFAULT = ["circle_cw", "circle_ccw", "circle_nil"]


def extract_landmarks(hand_landmarks):
    coords = [xy[0] for xy in hand_landmarks]
    coords.extend(xy[1] for xy in hand_landmarks)
    return coords


def augment_landmarks(landmarks):
    x_coords = np.array(landmarks[:21], dtype=np.float32)
    y_coords = np.array(landmarks[21:], dtype=np.float32)
    cx, cy = x_coords[0], y_coords[0]

    tx = np.random.uniform(-0.05, 0.05)
    ty = np.random.uniform(-0.05, 0.05)
    scale = np.random.uniform(0.9, 1.1)
    angle = np.radians(np.random.uniform(-15, 15))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    new_x, new_y = [], []
    for x, y in zip(x_coords, y_coords):
        nx, ny = x - cx, y - cy
        rx = nx * cos_a - ny * sin_a
        ry = nx * sin_a + ny * cos_a
        sx, sy = rx * scale, ry * scale
        final_x = sx + cx + tx + np.random.normal(0, 0.005)
        final_y = sy + cy + ty + np.random.normal(0, 0.005)
        new_x.append(final_x)
        new_y.append(final_y)
    return new_x + new_y


def init_static_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])


def draw_hud(image, lines, ok_color=(0, 255, 0)):
    y = 30
    for idx, text in enumerate(lines):
        color = ok_color if idx == 0 else (255, 255, 255)
        cv2.putText(
            image,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
        y += 28
    cv2.putText(image, "Press 'q' to quit", (10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)


def show_countdown(cap, window_name, title, seconds):
    end_time = time.time() + seconds
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            continue
        image = cv2.flip(frame, 1)
        remaining = max(0, int(np.ceil(end_time - time.time())))
        draw_hud(
            image,
            [
                f"{title}",
                f"Starting in: {remaining}s",
            ],
            ok_color=(0, 255, 255),
        )
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def collect_static(args):
    init_static_csv(args.static_csv)
    hand_tracker = HandLandmarkerCompat(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")
    window_name = "Gesture Data Collection - Static"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for gesture in args.static_gestures:
            print(f"\nPrepare static gesture: '{gesture}'")
            print(f"Starting in {args.prep_seconds}s...")
            if not show_countdown(cap, window_name, f"Prepare static gesture: {gesture}", args.prep_seconds):
                print("Data collection interrupted by user.")
                return

            count = 0
            total_saved = 0
            print(f"Recording {args.samples_per_gesture} real frames for '{gesture}'")

            while count < args.samples_per_gesture:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                image = cv2.flip(frame, 1)
                hand_landmarks = hand_tracker.detect(image)

                if hand_landmarks:
                    hand_tracker.draw(image, hand_landmarks)
                    landmarks = extract_landmarks(hand_landmarks)

                    with open(args.static_csv, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([gesture] + landmarks)
                        total_saved += 1
                        for _ in range(args.augmentations_per_frame):
                            writer.writerow([gesture] + augment_landmarks(landmarks))
                            total_saved += 1

                    count += 1
                    print(
                        f"[{gesture}] {count}/{args.samples_per_gesture} frames | rows saved: {total_saved}"
                    )
                    time.sleep(args.capture_delay)
                status = "Hand: detected" if hand_landmarks else "Hand: not detected"
                draw_hud(
                    image,
                    [
                        f"Mode: static | Target: {gesture}",
                        f"Progress: {count}/{args.samples_per_gesture} frames",
                        f"Rows saved (this gesture): {total_saved}",
                        status,
                    ],
                    ok_color=(0, 255, 0) if hand_landmarks else (0, 0, 255),
                )
                cv2.imshow(window_name, image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Data collection interrupted by user.")
                    return

            print(f"Finished '{gesture}'. Total rows written for this gesture: {total_saved}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.close()


def collect_dynamic(args):
    hand_tracker = HandLandmarkerCompat(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")
    window_name = "Gesture Data Collection - Dynamic"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    X, y = [], []
    try:
        for label in args.dynamic_gestures:
            print(f"\nPrepare dynamic gesture: '{label}'")
            print(f"Starting in {args.prep_seconds}s...")
            if not show_countdown(cap, window_name, f"Prepare dynamic gesture: {label}", args.prep_seconds):
                print("Data collection interrupted by user.")
                return
            sequences_recorded = 0

            while sequences_recorded < args.sequences_per_label:
                sequence = []
                while len(sequence) < args.sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame.")
                        break

                    image = cv2.flip(frame, 1)
                    hand_landmarks = hand_tracker.detect(image)

                    if hand_landmarks:
                        hand_tracker.draw(image, hand_landmarks)
                        sequence.append(extract_landmarks(hand_landmarks))
                    status = "Hand: detected" if hand_landmarks else "Hand: not detected"
                    draw_hud(
                        image,
                        [
                            f"Mode: dynamic | Target: {label}",
                            f"Frame in sequence: {len(sequence)}/{args.sequence_length}",
                            f"Sequences saved: {sequences_recorded}/{args.sequences_per_label}",
                            status,
                        ],
                        ok_color=(0, 255, 0) if hand_landmarks else (0, 0, 255),
                    )
                    cv2.imshow(window_name, image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Data collection interrupted by user.")
                        return

                if len(sequence) == args.sequence_length:
                    X.append(sequence)
                    y.append(label)
                    sequences_recorded += 1
                    print(f"[{label}] sequence {sequences_recorded}/{args.sequences_per_label}")

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        np.savez_compressed(args.dynamic_npz, X=X, y=y)
        print(f"\nSaved dynamic dataset to '{args.dynamic_npz}'")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect static and dynamic hand-gesture datasets."
    )
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--prep-seconds", type=int, default=5)

    parser.add_argument("--static-csv", default="data/gesture_data.csv")
    parser.add_argument("--samples-per-gesture", type=int, default=400)
    parser.add_argument("--augmentations-per-frame", type=int, default=7)
    parser.add_argument("--capture-delay", type=float, default=0.0)
    parser.add_argument(
        "--static-gestures",
        nargs="+",
        default=STATIC_GESTURES_DEFAULT,
        help="Static class labels to collect.",
    )

    parser.add_argument("--dynamic-npz", default="data/gesture_sequence_data.npz")
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--sequences-per-label", type=int, default=100)
    parser.add_argument(
        "--dynamic-gestures",
        nargs="+",
        default=DYNAMIC_GESTURES_DEFAULT,
        help="Dynamic class labels to collect.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "static":
        collect_static(args)
    else:
        collect_dynamic(args)


if __name__ == "__main__":
    main()
