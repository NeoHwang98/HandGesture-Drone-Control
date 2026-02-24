import cv2, time, pickle, keyboard, threading, numpy as np, mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from robomaster import robot


# ---------------------------
# 1. Load the model and other objects
# ---------------------------
with open('models/static/label_encoder2.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/static/gesture_model2.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/static/scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

STATIC_CLASS_LABELS = set(le.classes_)
# ---------------------------
# Loading Long Short-Term Memory objects
# ---------------------------
with open('models/dynamic/updated_label_encoder_lstm1.pkl', 'rb') as f:
    le_lstm = pickle.load(f)

EXPECTED_LSTM_LABELS = {"circle_cw", "circle_ccw", "circle_nil"}
lstm_enabled = False

try:
    model_lstm = load_model('models/dynamic/updated_gesture_lstm1.keras', compile = False)
    input_shape = model_lstm.input_shape
    output_shape = model_lstm.output_shape
    lstm_labels = set(le_lstm.classes_)

    valid_input = isinstance(input_shape, tuple) and len(input_shape) == 3 and input_shape[1:] == (15, 42)
    valid_output = isinstance(output_shape, tuple) and len(output_shape) == 2 and output_shape[1] == len(le_lstm.classes_)
    valid_labels = EXPECTED_LSTM_LABELS.issubset(lstm_labels)

    if valid_input and valid_output and valid_labels:
        lstm_enabled = True
    else:
        print("WARNING: LSTM artifacts are incompatible with expected sequence model contract.")
        print(f"  input_shape={input_shape}, output_shape={output_shape}, labels={sorted(lstm_labels)}")
except Exception as e:
    print(f"WARNING: Failed to load LSTM model: {e}")

if "close_palm1" not in STATIC_CLASS_LABELS:
    print("WARNING: 'close_palm1' not found in static classifier labels. Backward gesture will not trigger.")
if "ok1" not in STATIC_CLASS_LABELS:
    print("WARNING: 'ok1' not found in static classifier labels. Gesture-based landing trigger will not work.")

sequence_buffer = deque(maxlen = 15) # Buffer to hold 15 frames for LSTM
CONFIDENCE_THRESHOLD_LSTM = 0.75  # Adjust this to suit our requirements for the confidence of the circular gestures

# ---------------------------
# 2. Initialize mediapipe hands
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.8, #lowered from 0.7
    min_tracking_confidence = 0.25, #lowered from 0.5
    # static_image_mode = False
)
mp_drawing = mp.solutions.drawing_utils


# ---------------------------
# 3. Drone's LED matrix display
# ---------------------------
matrix_led_arrow_left = "000r000000r000000r000000rrrrrrrrrrrrrrrr0r00000000r00000000r00000" 
matrix_led_arrow_right = "0000r00000000r00000000r0rrrrrrrrrrrrrrrr000000r000000r000000r0000"
matrix_led_smile = "00bbbb000b0000b0b0b00b0bb000000bb0b00b0bb00bb00b0b0000b000bbbb00"
matrix_led_arrow_up = "000bb00000bbbb000b0bb0b0b00bb00b000bb000000bb000000bb000000bb000"
matrix_led_arrow_down = "000bb000000bb000000bb000000bb000b00bb00b0b0bb0b000bbbb00000bb000"
matrix_led_open_palm = "00000000000000000000000000000000000000000000000000000000bbbbbbbb"
matrix_led_fist = "bbbbbbbb00000000000000000000000000000000000000000000000000000000"
matrix_led_open_palm_no = "00rrrr000r0000r0rbbb000rrb0bbbbrrb0bb0brrb0bbbbr0r0000r000rrrr00"
matrix_led_rotate_left = "0bbb000000b000000b0b00b0b000000bb000000bb000000b0b0000b000bbbb00"
matrix_led_rotate_right = "0000bbb00000bb000b00b0b0b000000bb000000bb000000b0b0000b000bbbb00"
matrix_led_ok = "00000000bbb0b00rb0b0b0brb0b0bb0rb0b0bb0rbbb0b0b00000000r00000000"
matrix_led_two = "00pppp000p0000p00p0000p000000p000000p000000p000000p000000pppppp0"
matrix_led_one = "0000p000000pp00000p0p0000000p0000000p0000000p0000000p00000ppppp0"


# ---------------------------
# 4. Variables declaration
# These are considered global variables liao. If u want to modify them in functions,
# need to global them in the function. If read-only dont need to
# If variables are created in functions and want modify/read them in other functions,
# then need to declare global in both functions u created it in
# ---------------------------
classification_gesture = None
classification_confidence = 0.0
delta_math_orientation = None
delta_math_gesture = None
main_loop_stopper = 0
result_gesture_lstm = None
lstm_result = None
confidence_lstm = 0.0
results = None

thread_stopper = threading.Event() #threading event to signal the global stop of all threads


# ---------------------------
# 5. Encode hand coordinates into classfication model's 'dialect'
# ---------------------------
def compute_relative_features(landmarks):
    x_coords = landmarks[:21]
    y_coords = landmarks[21:]
    wrist_x, wrist_y = x_coords[0], y_coords[0]
    relative_features = []
    for i in range(21):
        relative_x = x_coords[i] - wrist_x
        relative_y = y_coords[i] - wrist_y
        relative_features.extend([relative_x, relative_y])
    return relative_features

def compute_thumb_index_distance(landmarks):
    thumb_tip_x, thumb_tip_y = landmarks[8], landmarks[9]
    index_tip_x, index_tip_y = landmarks[16], landmarks[17]
    distance = np.sqrt((thumb_tip_x - index_tip_x)**2 + (thumb_tip_y - index_tip_y)**2)
    return distance


# ---------------------------
# 6. 2D plane algorithm
# Finds out hand is point horizontally or vertically first(delta of x VS delta of y)
# Followed by computing direction by comparing fingers and wrist position
# ---------------------------
def delta_math(hand_landmark):
    global delta_math_orientation, delta_math_gesture

    wrist = hand_landmark.landmark[mp_hands.HandLandmark.WRIST]
    lmiddle = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    palm_x = 0.5*(wrist.x + lmiddle.x)
    palm_y = 0.5*(wrist.y + lmiddle.y)
    thumb = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]

    deltas_x = []
    deltas_y = []

    for landmark in [
        hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP],
        hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmark.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmark.landmark[mp_hands.HandLandmark.PINKY_TIP] ]:

        deltas_x.append(abs(landmark.x - palm_x))
        deltas_y.append(abs(landmark.y - palm_y))

    max_value_x = max(deltas_x) if deltas_x else 0
    max_value_y = max(deltas_y) if deltas_y else 0

    delta_math_orientation = 1 if max_value_x > max_value_y else 0  # 0 is horizontal, 1 is vertical
    if delta_math_orientation == 1 and thumb.x > palm_x:
        delta_math_gesture = "right"
    if delta_math_orientation == 1 and thumb.x < palm_x:
        delta_math_gesture = "left"
    if delta_math_orientation == 0 and thumb.y > palm_y:
        delta_math_gesture = "down"
    if delta_math_orientation == 0 and thumb.y < palm_y:
        delta_math_gesture = "up"


# ---------------------------
# 7. Function to determine hand gesture, via all 3 techniques
# ---------------------------
def detect_gestures():
    global results, delta_math_gesture, delta_math_orientation, classification_gesture, classification_confidence, video_single_frame_flipped, result_gesture_lstm, confidence_lstm, lstm_result, lstm_enabled
    while not thread_stopper.is_set():

        image_rgb = cv2.cvtColor(video_single_frame_flipped, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Mediapipe
        results = hands.process(image_rgb)  # Process the frame to detect hand landmarks

        if results.multi_hand_landmarks: # Checks if any hand landmarks were detected
            for hand_landmark in results.multi_hand_landmarks: # Iterates through each detected hand

                #trigger classifcation code
                landmarks = [lm.x for lm in hand_landmark.landmark] + [lm.y for lm in hand_landmark.landmark] #extract landmarks

                relative_features = compute_relative_features(landmarks)
                thumb_index_distance = compute_thumb_index_distance(landmarks)
                enhanced_features = np.array(relative_features + [thumb_index_distance]).reshape(1, -1)

                enhanced_features_scaled = scaler.transform(enhanced_features)

                prediction = model.predict(enhanced_features_scaled)
                prediction_proba = model.predict_proba(enhanced_features_scaled)
                class_id = prediction[0]
                classification_confidence = np.max(prediction_proba)
                classification_gesture = le.inverse_transform([class_id])[0]

                #trigger 2D plane algorithm code
                delta_math(hand_landmark)

                #trigger lstm code
                if classification_gesture == "open_palm1":
                    sequence_buffer.append(landmarks)     
                    time.sleep(0.05)
                    sequence_buffer.append(landmarks)                                               # Continuously updating the buffer with the next new frame

                if lstm_enabled and len(sequence_buffer) == 15:                                     # Double checks that buffer has 15 frames waiting
                    sequence = np.array(sequence_buffer).reshape(1, 15, 42)                         # Capturing the sequence of 15 frames via buffer
                    sequence_buffer.clear()
                    prediction_lstm = model_lstm.predict(sequence, verbose = 0)                     # Running LTSM prediction
                    confidence_lstm = np.max(prediction_lstm)                                       # Get the highest confidence score
                    predicted_label_lstm = le_lstm.inverse_transform([np.argmax(prediction_lstm)])  # Find out highest confidence related to which gesture
                    result_gesture_lstm = str(predicted_label_lstm[0])                              # Converts the gesture name into string
                    lstm_result = None                                                              # Clears away the movement in case nothing was confident enough 
                    if confidence_lstm > CONFIDENCE_THRESHOLD_LSTM:                                 # Check if it is confident enough to output
                        lstm_result = result_gesture_lstm                                           # lstm_result will be forwarded to drone_actions()  
                elif not lstm_enabled:
                    sequence_buffer.clear()
                    lstm_result = None
                    result_gesture_lstm = None
                    confidence_lstm = 0.0
        else:
            classification_gesture = None                                                           # Resets everything if no hand was detected
            classification_confidence = 0.0
            delta_math_orientation = None
            delta_math_gesture = None
            result_gesture_lstm = None
            confidence_lstm = 0.0


def drone_actions():
    global drone, drone_flight, drone_led
    global main_loop_stopper, results, lstm_result, sequence_buffer

    classification_generalthreshold = 0.3 # general threshold to jump into classification loop ln234
    classification_specialthreshold = 0.8 # lowered from 90% for fist and open palm
    open_palm_threshold = 0.5             # Attempting to make open_palm appear more, instead of 'up' command
    palm_shown = 0
    nothing_detected_counter = 0

    fly_up_count = 0
    max_fly_up_count= 45
    drone_action_initializer = 0

    while not thread_stopper.is_set(): #will run as long as thread stopper is not set
        if main_loop_stopper != 1 and classification_confidence != 0.0 and nothing_detected_counter == 1: 
            nothing_detected_counter = 0

        if drone_action_initializer != 1 and fly_up_count <= max_fly_up_count - 1:
            if results is None or results.multi_hand_landmarks is None:
                if fly_up_count == 0:
                    drone.led.set_mled_graph(matrix_led_arrow_up)
                    drone_led.set_led(r=255, g=0, b=0)
                drone_flight.rc(a=0, b=0, c=20, d=0)
                fly_up_count += 1
                print(fly_up_count)
                time.sleep(0.15)
            else:
                drone_action_initializer = 1
                print("starting in 2s")
                drone.led.set_mled_graph(matrix_led_two)
                drone_led.set_led(r=0, g=255, b=0)
                drone_flight.rc(a=0, b=0, c=0, d=0)
                time.sleep(1)
                print("starting in 1s")
                drone.led.set_mled_graph(matrix_led_one)
                time.sleep(1)
            continue

        if classification_confidence >= classification_generalthreshold and main_loop_stopper != 1:  # Classification Loop
            #a is roll, b is pitch, c is throttle, d is yaw
            if classification_gesture == "open_palm1"  and (lstm_result == "circle_nil" or lstm_result == None): #1st latch to lock open palm
                if palm_shown == 0:
                    print("palm_detected")
                    drone_led.set_led(r=0, g=255, b=0)
                    drone.led.set_mled_graph(matrix_led_smile)
                    palm_shown = 1
                time.sleep(0.1)
                continue #this is so that i skip the 15 frame leak into up gesture detection completely
            if classification_gesture == "open_palm1"  and lstm_result == "circle_cw":
                print(f'Prediction: {lstm_result} (Conf: {confidence_lstm:.2f})')
                lstm_result = None
                drone.led.set_mled_graph(matrix_led_rotate_left)
                drone_led.set_led(r=255, g=0, b=0)
                # drone_flight.rc(a=0, b=0, c=0, d=50) #rotates left, dont use this cus this is for continuous movement
                drone_flight.rotate(angle = -360).wait_for_completed()
                palm_shown = 0
                time.sleep(0.5)
                sequence_buffer.clear()
                lstm_result = None
                continue
            if classification_gesture == "open_palm1"  and lstm_result == "circle_ccw":
                print(f'Prediction: {lstm_result} (Conf: {confidence_lstm:.2f})')
                lstm_result = None
                drone.led.set_mled_graph(matrix_led_rotate_right)
                drone_led.set_led(r=255, g=0, b=0)
                # drone_flight.rc(a=0, b=0, c=0, d=-50) #rotates right, dont use this cus this is for continuous movement
                drone_flight.rotate(angle = 360).wait_for_completed()
                palm_shown = 0
                time.sleep(0.5)
                sequence_buffer.clear()
                lstm_result = None
                continue
            if classification_gesture == "open_palm1" and classification_confidence >= open_palm_threshold: #2nd latch to catch the remaining conditions
                if palm_shown == 0:
                    print("palm_detected")
                    palm_shown = 1
                    drone_led.set_led(r=0, g=255, b=0)
                    drone.led.set_mled_graph(matrix_led_smile)
                time.sleep(0.1)
                continue

            palm_shown = 0
            lstm_result = None

            if classification_gesture == "fist1" and classification_confidence >= classification_specialthreshold:
                print(f"Fist detected with {classification_confidence * 100:.1f}% confidence")
                drone.led.set_mled_graph(matrix_led_fist)
                drone_led.set_led(r=255, g=0, b=0)
                drone_flight.rc(a=0, b=30, c=0, d=0) #flies forward
                time.sleep(0.1)
                continue
            elif "close_palm1" in STATIC_CLASS_LABELS and classification_gesture == "close_palm1" and classification_confidence >= classification_specialthreshold:
                print(f"close palm detected with {classification_confidence * 100:.1f}% confidence")
                drone.led.set_mled_graph(matrix_led_open_palm_no)
                drone_led.set_led(r=255, g=0, b=0)
                drone_flight.rc(a=0, b=-30, c=0, d=0) 
                time.sleep(0.1)
                continue
            elif "ok1" in STATIC_CLASS_LABELS and classification_gesture == "ok1" and classification_confidence >= classification_specialthreshold:
                print('OK detected')
                drone.led.set_mled_graph(matrix_led_ok)
                drone_led.set_led(r=0, g=255, b=0)
                main_loop_stopper = 1
                break
        
        if delta_math_gesture == "up" and classification_gesture != "open_palm1":
            print("Up gesture detected")
            drone.led.set_mled_graph(matrix_led_arrow_up)
            drone_led.set_led(r=255, g=0, b=0)
            drone_flight.rc(a=0, b=0, c=40, d=0) #flies up
            time.sleep(0.1)
            continue
        if delta_math_gesture == "left":
            print("Left gesture detected")
            drone.led.set_mled_graph(matrix_led_arrow_left)
            drone_led.set_led(r=255, g=0, b=0)
            drone_flight.rc(a=30, b=0, c=0, d=0) #flies drone's right
            time.sleep(0.1)
            continue
        elif delta_math_gesture == "right":
            print("Right gesture detected")
            drone.led.set_mled_graph(matrix_led_arrow_right)
            drone_led.set_led(r=255, g=0, b=0)
            drone_flight.rc(a=-30, b=0, c=0, d=0) #flies drone's left
            time.sleep(0.1)
            continue
        elif delta_math_gesture == "down":
            print("Down gesture detected")
            drone.led.set_mled_graph(matrix_led_arrow_down)
            drone_led.set_led(r=255, g=0, b=0)
            drone_flight.rc(a=0, b=0, c=-40, d=0) #flies down
            time.sleep(0.1)
            continue
            
        else:
            if main_loop_stopper != 1 and classification_confidence == 0.0 and nothing_detected_counter == 0: #nothing_detected_counter == 0 and main_loop_stopper != 1:
                nothing_detected_counter = 1
                palm_shown = 0
                print("Nothing detected")
                drone_led.set_led(r=0, g=255, b=0)
                drone_led.set_mled_char_scroll(direction = "l", color = "r", freq = 2.5, display_str = "GHOST")
                drone_flight.rc(a=0, b=0, c=0, d=0)
                time.sleep(0.1)
            time.sleep(0.1)


def main():
    global drone, drone_flight, drone_led
    global video_single_frame_flipped

    drone = robot.Drone()
    drone.initialize()
    drone_flight = drone.flight
    drone_battery = drone.battery
    drone_led = drone.led

    drone_camera = drone.camera
    drone_camera.start_video_stream(display = False)
    drone_camera.set_fps("high")
    drone_camera.set_resolution("medium")
    drone_camera.set_bitrate(6)

    try:
        drone_actions_thread = threading.Thread(target = drone_actions)
        detect_gestures_thread = threading.Thread(target = detect_gestures)

        # video_object = cv2.VideoCapture(0)
        drone_flight.takeoff()
        
        while True:
            video_single_frame = drone_camera.read_cv2_image()
            
            # ret, video_single_frame = video_object.read()
            # if not ret:
            #     print("Failed to grab frame.")
            #     break

            video_single_frame_flipped = cv2.flip(video_single_frame, 1)
            cv2.imshow("Drone Camera", video_single_frame_flipped)

            if detect_gestures_thread is None or not detect_gestures_thread.is_alive():
                detect_gestures_thread.start()
                time.sleep(0.1) #lets dynamic_gestures thread start up reliably first
            
            if drone_actions_thread is None or not drone_actions_thread.is_alive():
                drone_actions_thread.start()

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("PRESS ESC LA AIYO BROOO")
                continue
            if keyboard.is_pressed("esc"):
                print("EXITING FROM MAIN TRY LOOP")
                thread_stopper.set()
                break

            if main_loop_stopper == 1:
                print("EXITING FROM MAIN TRY LOOP")
                thread_stopper.set()
                break

    finally:
        drone_flight.land().wait_for_completed()
        drone_battery_info = drone_battery.get_battery()
        print("Drone battery remaining: {0}".format(drone_battery_info))

        print("STOPPING ALL THREADS...")
        thread_stopper.set()
        if drone_actions_thread.is_alive():
            drone_actions_thread.join()
        if detect_gestures_thread.is_alive():
            detect_gestures_thread.join()
        print("STOPPED ALL THREADS SUCCESSFULLY")
        
        print("STOPPING CV2...")
        # video_object.release()
        cv2.destroyAllWindows()
        print("STOPPED CV2 SUCCESSFULLY")

        print("STOPPING DRONE CAMERA")
        drone_camera.stop_video_stream()
        print("STOPPED DRONE CAMERA SUCCESSFULLY")

        print("STOPPING MEDIAPIPE...")
        hands.close() 
        print("STOPPED MEDIAPIPE SUCCESSFULLY")

        print("STOPPING DRONE CONNECTION...")
        drone.close()
        print("STOPPED DRONE CONNECTION SUCCESSFULLY")

if __name__ == "__main__":
    main()
