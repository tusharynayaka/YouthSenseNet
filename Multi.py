import cv2
import numpy as np
import time
import csv
import os
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
import mediapipe as mp

# ==== CONFIG & CONSTANTS ====
MODEL_PATH = "my_85percent_model.h5"
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
IMAGE_SIZE = 48
LOG_FILENAME = "attentiveness_log.csv"
LOG_INTERVAL_SECONDS = 10.0 # Log status every 10 seconds
CONDITIONAL_PERSISTENCE_SECONDS = 3.0 # NEW: Pitch AND EAR must be low for 3 seconds

# ==== INDIRECT DISENGAGEMENT CONSTANTS (Pitch AND EAR) ====
# EXTREME PITCH THRESHOLD: Head bent down excessively (e.g., face parallel to chest/desk).
HEAD_PITCH_THRESHOLD = -160 
# EAR THRESHOLD: Eyes closed or very narrow (used for both Drowsiness and the combined check).
EYE_AR_THRESHOLD = 0.185 
EYE_AR_CONSEC_FRAMES = 30 # For sustained drowsiness detection (Approx. 1-2 seconds)

# ==== GLOBAL CONSTANTS & STATE ====
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
SEVERE_NEGATIVE_EMOTIONS = ['angry', 'disgust', 'fear'] 

# Dictionary to store tracking information for each student
# Key: tracking_id (int) -> Value: { 'drowsy_counter': int, 'cond_unatt_start_time': float, 'last_log_time': float, 'center': tuple }
student_trackers = {} 

# Simple ID Counter for assigning new students
next_student_id = 1 
last_log_time = time.time()

# Initialize MediaPipe Face Mesh (Multi-Face tracking enabled)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=5, # Allow tracking up to 5 students
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Indices for the eyes in the MediaPipe Face Mesh model (approximate)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380] 
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------

def init_csv_logger():
    """Initializes the CSV file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILENAME):
        with open(LOG_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Student_ID', 'Status', 'Reason', 'Pitch', 'EAR', 'Emotion'])

def log_to_csv(student_data):
    """Logs the attentiveness data for a single student to the CSV."""
    with open(LOG_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(student_data)

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR) using 6 points. """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(landmarks, frame_w, frame_h):
    """Calculates head pose (Pitch only) using PnP with MediaPipe landmarks."""
    # 2D and 3D points for PnP
    image_points = np.array([
        [landmarks[4].x, landmarks[4].y], [landmarks[152].x, landmarks[152].y],
        [landmarks[226].x, landmarks[226].y], [landmarks[446].x, landmarks[446].y],
        [landmarks[61].x, landmarks[61].y], [landmarks[291].x, landmarks[291].y]
    ], dtype="double") * [frame_w, frame_h]

    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    
    pitch = euler_angles[0, 0] 
    
    return int(pitch), image_points

def get_bbox_from_landmarks(landmarks, frame_w, frame_h):
    """Calculates a simple bounding box from all detected landmarks."""
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    x_min = int(min(x_coords) * frame_w)
    y_min = int(min(y_coords) * frame_h)
    x_max = int(max(x_coords) * frame_w)
    y_max = int(max(y_coords) * frame_h)
    
    return x_min, y_min, x_max - x_min, y_max - y_min

# Simple function to assign a persistent ID based on proximity to existing students
def assign_student_id(center_x, center_y):
    global student_trackers, next_student_id

    # Try to match to an existing student based on proximity (100 pixels threshold for simplicity)
    for student_id, tracker_data in student_trackers.items():
        if 'center' in tracker_data:
            dist_sq = (center_x - tracker_data['center'][0])**2 + (center_y - tracker_data['center'][1])**2
            if dist_sq < 100**2: 
                return student_id
    
    # If no match, assign a new ID and initialize tracking data
    new_id = next_student_id
    student_trackers[new_id] = {
        'drowsy_counter': 0, 
        'cond_unatt_start_time': None, # New field for conditional persistence
        'last_log_time': time.time() - LOG_INTERVAL_SECONDS * 2
    } 
    next_student_id += 1
    return new_id


# ----------------------------------------------------
# LIVE DETECTION LOOP
# ----------------------------------------------------
def live_detection(model):
    global student_trackers, last_log_time, next_student_id

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("Error: camera not available.")
        return

    init_csv_logger()
    print(f"Multi-Student Monitor (3s Persistence) running. Logging to {LOG_FILENAME} every {LOG_INTERVAL_SECONDS}s. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        present_student_ids = []
        
        # --- Start Face Processing ---
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                landmarks = face_landmarks.landmark
                x, y, w, h = get_bbox_from_landmarks(landmarks, FRAME_W, FRAME_H)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 1. Assign/Retrieve Persistent Student ID
                student_id = assign_student_id(center_x, center_y)
                present_student_ids.append(student_id)

                # Get tracker data and update center for next frame's association
                tracker = student_trackers[student_id]
                tracker['center'] = (center_x, center_y)

                l_eye_pts = np.array([[int(landmarks[i].x * FRAME_W), int(landmarks[i].y * FRAME_H)] for i in LEFT_EYE_INDICES])
                r_eye_pts = np.array([[int(landmarks[i].x * FRAME_W), int(landmarks[i].y * FRAME_H)] for i in RIGHT_EYE_INDICES])

                # --- 1. EMOTION PREDICTION ---
                roi = gray_frame[y:y + h, x:x + w]
                predicted_emotion = "N/A"
                
                if roi.shape[0] >= IMAGE_SIZE and roi.shape[1] >= IMAGE_SIZE:
                    roi_resized = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
                    roi_input = roi_resized.astype("float32") / 255.0
                    roi_input = np.expand_dims(roi_input, axis=[0, -1]) 
                    
                    preds = model.predict(roi_input, verbose=0)[0]
                    predicted_emotion = emotion_labels[np.argmax(preds)]
                    emotion_conf = preds[np.argmax(preds)]
                else:
                    emotion_conf = 0.0
                
                # --- 2. DROWSINESS CHECK (EAR) ---
                leftEAR = eye_aspect_ratio(l_eye_pts)
                rightEAR = eye_aspect_ratio(r_eye_pts)
                ear = (leftEAR + rightEAR) / 2.0
                
                is_low_ear = (ear < EYE_AR_THRESHOLD)
                
                if is_low_ear:
                    tracker['drowsy_counter'] += 1
                else:
                    tracker['drowsy_counter'] = 0

                is_drowsy_sustained = (tracker['drowsy_counter'] >= EYE_AR_CONSEC_FRAMES)

                # --- 3. PITCH CHECK (Head Pose) ---
                pitch, image_points = get_head_pose(landmarks, FRAME_W, FRAME_H)
                is_extreme_pitch = (pitch < HEAD_PITCH_THRESHOLD)
                
                # --- 4. CONDITIONAL PERSISTENCE LOGIC (The core update) ---
                
                is_currently_conditional_low = is_extreme_pitch and is_low_ear

                if is_currently_conditional_low:
                    # If conditions are met and timer hasn't started, start the timer
                    if tracker['cond_unatt_start_time'] is None:
                        tracker['cond_unatt_start_time'] = current_time
                else:
                    # If conditions are broken, reset the timer
                    tracker['cond_unatt_start_time'] = None

                # Check if the combined condition has persisted for the required duration
                is_cond_unattentive_sustained = (
                    tracker['cond_unatt_start_time'] is not None and 
                    (current_time - tracker['cond_unatt_start_time']) >= CONDITIONAL_PERSISTENCE_SECONDS
                )
                
                # --- 5. FINAL ATTENTIVENESS LOGIC ---
                
                is_severe_emotion = predicted_emotion in SEVERE_NEGATIVE_EMOTIONS
                
                # UNATTENTIVE TRIGGERS: 
                # 1. Pitch AND Low EAR (Sustained for 3s)
                # 2. Sustained Drowsiness (EAR only, for nodding off)
                # 3. Severe Emotion
                is_unattentive = is_cond_unattentive_sustained or is_drowsy_sustained or is_severe_emotion

                # --- 6. VISUAL OUTPUT & LOGGING DATA PREP ---
                
                if is_unattentive:
                    reasons = []
                    
                    if is_cond_unattentive_sustained: 
                        reasons.append(f"Conditional (Sustained > {CONDITIONAL_PERSISTENCE_SECONDS}s)") 
                    
                    # Use elif here to ensure we prioritize the conditional flag if both are met
                    elif is_drowsy_sustained: 
                        reasons.append(f"Drowsy(Frames:{tracker['drowsy_counter']})")
                    
                    if is_severe_emotion: 
                        reasons.append(f"SevereEmotion({predicted_emotion})")
                    
                    reason_str = "; ".join(reasons)
                    overall_status = "UNATTENTIVE"
                    status_color = (0, 0, 255) # RED
                else:
                    overall_status = "ATTENTIVE"
                    reason_str = "N/A"
                    status_color = (0, 255, 0) # GREEN

                # Data for CSV Log
                log_data = [
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
                    student_id,
                    overall_status,
                    reason_str,
                    f"{pitch}°",
                    f"{ear:.2f}",
                    predicted_emotion
                ]

                # --- 7. LOGGING TO CSV ---
                if current_time - tracker['last_log_time'] >= LOG_INTERVAL_SECONDS:
                    log_to_csv(log_data)
                    tracker['last_log_time'] = current_time

                # --- 8. DRAW OUTPUT ---
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
                
                # Display Student ID and Status
                cv2.putText(
                    frame,
                    f"S{student_id}: {overall_status}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
                )
                
                # Display detailed metrics
                cv2.putText(frame, f"E: {predicted_emotion} ({emotion_conf:.0%})", 
                            (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show remaining time if conditions are met
                time_remaining = 0.0
                if is_currently_conditional_low and tracker['cond_unatt_start_time'] is not None:
                     time_elapsed = current_time - tracker['cond_unatt_start_time']
                     time_remaining = max(0.0, CONDITIONAL_PERSISTENCE_SECONDS - time_elapsed)

                cv2.putText(frame, f"P: {pitch}°, EAR: {ear:.2f} (T- {time_remaining:.1f}s)", 
                            (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Cleanup trackers for students who left the frame (simple decay)
        for s_id in list(student_trackers.keys()):
            if s_id not in present_student_ids and 'center' in student_trackers[s_id]:
                # Remove center data to allow them to be re-tracked when they return
                del student_trackers[s_id]['center']


        # --- 9. OVERALL SYSTEM STATUS DISPLAY ---
        
        summary_text = f"Students: {len(present_student_ids)} | Next Log in: {LOG_INTERVAL_SECONDS - (current_time - (last_log_time if next_student_id == 1 else student_trackers[present_student_ids[0]].get('last_log_time', 0))):.1f}s"
        cv2.putText(frame, summary_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Multi-Student Attentiveness Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print(f"Closed. Data logged to {LOG_FILENAME}")

if __name__ == "__main__":
    try:
        model = load_model(MODEL_PATH)
        live_detection(model)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Model file not found at {e.filename}.")
        print("Ensure 'my_85percent_model.h5' is in the script directory.")
    except ImportError:
        print("FATAL ERROR: Required library not installed.")
        print("Please run: pip install tensorflow opencv-python mediapipe scipy")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")