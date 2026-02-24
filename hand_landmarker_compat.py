import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

# Pairs of landmark indices for drawing the hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

DEFAULT_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_task_file(path):
    task_path = Path(path)
    if task_path.exists():
        return task_path
    task_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DEFAULT_TASK_URL, str(task_path))
    return task_path


class HandLandmarkerCompat:
    def __init__(
        self,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        task_model_path="assets/hand_landmarker.task",
    ):
        self._use_solutions = hasattr(mp, "solutions")
        self._timestamp_ms = 0

        if self._use_solutions:
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            model_override = os.environ.get("MP_HAND_LANDMARKER_TASK_PATH")
            model_path = model_override if model_override else task_model_path
            model_path = _ensure_task_file(model_path)

            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            vision = mp.tasks.vision
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        if self._use_solutions:
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._hands.process(image_rgb)
            if not results.multi_hand_landmarks:
                return None
            return [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += 33
        if not result.hand_landmarks:
            return None
        return [(lm.x, lm.y) for lm in result.hand_landmarks[0]]

    def draw(self, frame_bgr, landmarks_xy, color=(0, 255, 0)):
        h, w = frame_bgr.shape[:2]
        pts = [(int(x * w), int(y * h)) for (x, y) in landmarks_xy]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(frame_bgr, p, 3, (255, 255, 255), -1)

    def close(self):
        if self._use_solutions:
            self._hands.close()
        else:
            self._landmarker.close()
