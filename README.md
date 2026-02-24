# Vision-Based Gesture Control of Drones

Real-time hand gesture control for a DJI RoboMaster drone using computer vision, classical ML, and sequence modeling.

## Project Overview

This project uses MediaPipe hand landmarks and a hybrid recognition pipeline to control drone motion from hand gestures.

The control stack combines:
- Static gesture classification (pickled model + scaler + label encoder)
- Rule-based 2D directional inference (`delta_math`) for up/down/left/right
- LSTM sequence detection for circular gestures (clockwise/counter-clockwise)

The primary runtime script reads the drone camera feed, detects gestures, and maps them to drone flight commands and onboard LED indicators.

## Demo Video

Watch project demo at https://drive.google.com/file/d/1aaCWzcWyq33mFf73Fruo_OFLyHly0il5/view?usp=drive_link

## Repository Structure

- `Vision based Gesture Control of Drones.py` - Main runtime for drone control and gesture inference
- `hand_gesture_data_collection.py` - Webcam data collection and augmentation utility
- `models/static/gesture_model2.pkl` - Trained static gesture classifier
- `models/static/label_encoder2.pkl` - Label encoder for static model outputs
- `models/static/scaler2.pkl` - Feature scaler used before static model inference
- `models/dynamic/updated_gesture_lstm1.keras` - Trained LSTM model for sequential gestures
- `models/dynamic/updated_label_encoder_lstm1.pkl` - Label encoder for LSTM outputs
- `data/gesture_data.csv` - Static gesture training dataset
- `data/gesture_sequence_data.npz` - Dynamic sequence training dataset
- `docs/EE3180-E028-24S1 (DIP).pdf` - Project report

## How It Works

1. Drone video frames are captured from RoboMaster camera.
2. MediaPipe extracts 21 hand landmarks (x/y).
3. Static model predicts gesture class and confidence.
4. Directional fallback/augmentation uses relative landmark geometry.
5. If open palm is detected, a 15-frame sequence is buffered for LSTM circular-gesture detection.
6. Gesture outputs are converted to drone movement commands and LED feedback.

## Gesture-to-Action Mapping (From Current Code)

- `fist1` -> fly forward
- `close_palm1` -> fly backward
- `ok1` -> stop main loop / initiate landing flow
- Directional hand orientation -> up/down/left/right movement
- `open_palm1` + `circle_cw` (LSTM) -> rotate one direction
- `open_palm1` + `circle_ccw` (LSTM) -> rotate opposite direction

Note: Left/right rotate direction depends on drone SDK yaw sign convention used in the script.

## Requirements

- Python 3.8.19 (recommended for RoboMaster SDK + MediaPipe compatibility)
- DJI RoboMaster drone and SDK connectivity
- Webcam (for data collection script)

Python packages:
- `numpy`
- `opencv-python`
- `mediapipe`
- `keyboard`
- `scikit-learn`
- `tensorflow` (for Keras model loading)
- `robomaster`

Install:

```bash
pip install -r requirements.txt
```

## Run

Start drone control runtime:

```bash
python "Vision based Gesture Control of Drones.py"
```

Run data collection utility:

```bash
python hand_gesture_data_collection.py
```

Collect dynamic sequence data for LSTM training:

```bash
python hand_gesture_data_collection.py --mode dynamic --dynamic-npz data/gesture_sequence_data.npz
```

Retrain static classifier artifacts used by runtime (`models/static/gesture_model2.pkl`, `models/static/label_encoder2.pkl`, `models/static/scaler2.pkl`):

```bash
python train_static_model.py --data-csv data/gesture_data.csv
```

Retrain dynamic LSTM artifacts used by runtime (`models/dynamic/updated_gesture_lstm1.keras`, `models/dynamic/updated_label_encoder_lstm1.pkl`):

```bash
python train_lstm_model.py --data-npz data/gesture_sequence_data.npz
```

Webcam-only inference test (no drone required):

```bash
python webcam_inference_test.py
```

## Notes and Safety

- Operate in a controlled area with clear safety boundaries.
- Keep an emergency stop method available.
- Test gesture inference with low-speed movement first.
- Press `Esc` to exit runtime (as implemented in script).

## Limitations

- The static classifier artifacts (`models/static/gesture_model2.pkl`, `models/static/label_encoder2.pkl`) currently expose 14 labels and do not include `close_palm1` / `ok1`, which are referenced by runtime command branches.
- Some gesture labels and thresholds are hard-coded and may need calibration for different users and lighting conditions.
- The dynamic LSTM artifact has been corrected for shape/class compatibility, but it should still be retrained on your real captured sequence dataset for production use.

## Suggested Next Improvements

- Refactor monolithic runtime into modules (`vision`, `inference`, `control`, `ui`).
- Add command-line flags for thresholds and model paths.
- Add unit tests for feature extraction and directional logic.
- Add a reproducible training pipeline script and dataset schema docs.

