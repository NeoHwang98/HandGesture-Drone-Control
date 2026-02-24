import argparse
import pickle
import time
from collections import deque

import numpy as np
from tensorflow.keras.models import load_model
from hand_landmarker_compat import HandLandmarkerCompat


CONFIDENCE_THRESHOLD_STATIC = 0.3
CONFIDENCE_THRESHOLD_LSTM = 0.75


def parse_args():
    parser = argparse.ArgumentParser(
        description="Webcam-only inference test for static + dynamic gesture models."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-static", default="models/static/gesture_model2.pkl")
    parser.add_argument("--encoder-static", default="models/static/label_encoder2.pkl")
    parser.add_argument("--scaler-static", default="models/static/scaler2.pkl")
    parser.add_argument("--model-lstm", default="models/dynamic/updated_gesture_lstm1.keras")
    parser.add_argument("--encoder-lstm", default="models/dynamic/updated_label_encoder_lstm1.pkl")
    parser.add_argument("--lstm-threshold", type=float, default=CONFIDENCE_THRESHOLD_LSTM)
    parser.add_argument("--static-threshold", type=float, default=CONFIDENCE_THRESHOLD_STATIC)
    return parser.parse_args()


def compute_relative_features(landmarks):
    x_coords = landmarks[:21]
    y_coords = landmarks[21:]
    wrist_x, wrist_y = x_coords[0], y_coords[0]
    relative_features = []
    for i in range(21):
        relative_features.extend([x_coords[i] - wrist_x, y_coords[i] - wrist_y])
    return relative_features


def compute_thumb_index_distance(landmarks):
    thumb_tip_x, thumb_tip_y = landmarks[8], landmarks[9]
    index_tip_x, index_tip_y = landmarks[16], landmarks[17]
    return np.sqrt((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2)


def delta_math(hand_landmarks_xy):
    wrist = hand_landmarks_xy[0]
    middle_mcp = hand_landmarks_xy[9]
    thumb_tip = hand_landmarks_xy[4]
    palm_x = 0.5 * (wrist[0] + middle_mcp[0])
    palm_y = 0.5 * (wrist[1] + middle_mcp[1])

    tip_indices = [4, 8, 12, 16, 20]
    deltas_x = [abs(hand_landmarks_xy[i][0] - palm_x) for i in tip_indices]
    deltas_y = [abs(hand_landmarks_xy[i][1] - palm_y) for i in tip_indices]

    orientation = 1 if max(deltas_x) > max(deltas_y) else 0
    if orientation == 1 and thumb_tip[0] > palm_x:
        return "right"
    if orientation == 1 and thumb_tip[0] < palm_x:
        return "left"
    if orientation == 0 and thumb_tip[1] > palm_y:
        return "down"
    if orientation == 0 and thumb_tip[1] < palm_y:
        return "up"
    return None


def main():
    args = parse_args()
    try:
        import cv2
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "opencv-python is required for webcam inference. Install with: pip install opencv-python"
        ) from e

    with open(args.encoder_static, "rb") as f:
        le = pickle.load(f)
    with open(args.model_static, "rb") as f:
        model = pickle.load(f)
    with open(args.scaler_static, "rb") as f:
        scaler = pickle.load(f)

    with open(args.encoder_lstm, "rb") as f:
        le_lstm = pickle.load(f)
    model_lstm = load_model(args.model_lstm, compile=False)

    lstm_enabled = (
        isinstance(model_lstm.input_shape, tuple)
        and len(model_lstm.input_shape) == 3
        and model_lstm.input_shape[1:] == (15, 42)
        and model_lstm.output_shape[1] == len(le_lstm.classes_)
    )
    if not lstm_enabled:
        print("WARNING: LSTM model is not compatible with expected (15,42) sequence input.")

    print(f"Static labels: {list(le.classes_)}")
    print(f"LSTM labels: {list(le_lstm.classes_)}")

    hand_tracker = HandLandmarkerCompat(
        max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.25
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    sequence_buffer = deque(maxlen=15)
    lstm_result = None
    confidence_lstm = 0.0
    last_lstm_ts = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            hand_landmarks = hand_tracker.detect(frame)

            static_text = "Static: none"
            static_conf_text = "Conf: 0.00"
            delta_text = "Delta: none"
            lstm_text = "LSTM: none"
            action_text = "Action: none"

            if hand_landmarks:
                hand_tracker.draw(frame, hand_landmarks)

                landmarks = [xy[0] for xy in hand_landmarks] + [xy[1] for xy in hand_landmarks]
                relative_features = compute_relative_features(landmarks)
                thumb_index_distance = compute_thumb_index_distance(landmarks)
                enhanced_features = np.array(relative_features + [thumb_index_distance]).reshape(1, -1)
                enhanced_features_scaled = scaler.transform(enhanced_features)

                pred = model.predict(enhanced_features_scaled)[0]
                pred_proba = model.predict_proba(enhanced_features_scaled)[0]
                static_gesture = le.inverse_transform([pred])[0]
                static_conf = float(np.max(pred_proba))
                static_text = f"Static: {static_gesture}"
                static_conf_text = f"Conf: {static_conf:.2f}"

                delta_gesture = delta_math(hand_landmarks)
                if delta_gesture:
                    delta_text = f"Delta: {delta_gesture}"

                if static_gesture == "open_palm1":
                    sequence_buffer.append(landmarks)
                    time.sleep(0.05)
                    sequence_buffer.append(landmarks)
                else:
                    sequence_buffer.clear()
                    lstm_result = None

                if lstm_enabled and len(sequence_buffer) == 15:
                    sequence = np.array(sequence_buffer).reshape(1, 15, 42)
                    sequence_buffer.clear()
                    pred_lstm = model_lstm.predict(sequence, verbose=0)[0]
                    confidence_lstm = float(np.max(pred_lstm))
                    pred_lstm_label = le_lstm.inverse_transform([int(np.argmax(pred_lstm))])[0]
                    if confidence_lstm >= args.lstm_threshold:
                        lstm_result = str(pred_lstm_label)
                        last_lstm_ts = time.time()

                if lstm_result and time.time() - last_lstm_ts < 1.0:
                    lstm_text = f"LSTM: {lstm_result} ({confidence_lstm:.2f})"
                elif lstm_result:
                    lstm_result = None

                # Emulated action resolution without drone control.
                if static_conf >= args.static_threshold:
                    if static_gesture == "fist1":
                        action_text = "Action: FORWARD"
                    elif static_gesture == "close_palm1":
                        action_text = "Action: BACKWARD"
                    elif static_gesture == "ok1":
                        action_text = "Action: LAND/STOP"
                    elif static_gesture == "open_palm1" and lstm_result == "circle_cw":
                        action_text = "Action: ROTATE LEFT"
                    elif static_gesture == "open_palm1" and lstm_result == "circle_ccw":
                        action_text = "Action: ROTATE RIGHT"
                    elif delta_gesture:
                        action_text = f"Action: MOVE {delta_gesture.upper()}"
            else:
                sequence_buffer.clear()
                lstm_result = None

            cv2.putText(frame, static_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, static_conf_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, delta_text, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(frame, lstm_text, (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, action_text, (10, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 2)
            cv2.imshow("Webcam Gesture Inference Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_tracker.close()


if __name__ == "__main__":
    main()
