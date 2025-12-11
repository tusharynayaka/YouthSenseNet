import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
import mediapipe as mp

# ==== CONFIG ====
MODEL_PATH = "my_85percent_model.h5"
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
IMAGE_SIZE = 48

# ==== INDIRECT DISENGAGEMENT CONSTANTS (Pitch AND EAR) ====
# EXTREME PITCH THRESHOLD: Head bent down excessively (e.g., face parallel to chest/desk).
HEAD_PITCH_THRESHOLD = -160 

# EAR THRESHOLD: Eyes closed or very narrow (used for both Drowsiness and the combined check).
EYE_AR_THRESHOLD = 0.185 
EYE_AR_CONSEC_FRAMES = 30 # For sustained drowsiness detection

# ==== GLOBAL CONSTANTS & STATE ====
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
COUNTER = 0 # Global frame counter for sustained drowsiness

# Define the set of severe negative emotions (Sadness is excluded)
SEVERE_NEGATIVE_EMOTIONS = ['angry', 'disgust', 'fear'] 

# Initialize MediaPipe Face Mesh (Set max_num_faces=1 for single student)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, 
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
    
    # We only care about Pitch (Up/Down)
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

# ----------------------------------------------------
# LIVE DETECTION LOOP
# ----------------------------------------------------
def live_detection(model):
    global COUNTER 

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("Error: camera not available.")
        return

    print("Single-Student Attentiveness Monitor (Conditional Logic) running. Press 'q' to quit.")

    overall_status = "Scanning..."
    status_color = (100, 100, 100)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Start Face Processing ---
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0] 
                
            landmarks = face_landmarks.landmark
            x, y, w, h = get_bbox_from_landmarks(landmarks, FRAME_W, FRAME_H)
            
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
            
            is_drowsy_sustained = False
            if ear < EYE_AR_THRESHOLD:
                COUNTER += 1 
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    is_drowsy_sustained = True
            else:
                COUNTER = 0
            
            # Instantaneous low EAR check (for the combined condition)
            is_low_ear = (ear < EYE_AR_THRESHOLD)

            # --- 3. PITCH CHECK (Head Pose) ---
            pitch, image_points = get_head_pose(landmarks, FRAME_W, FRAME_H)
            
            # Instantaneous low Pitch check (for the combined condition)
            is_extreme_pitch = (pitch < HEAD_PITCH_THRESHOLD)
            
            # --- 4. CUSTOM ATTENTIVENESS LOGIC ---
            
            is_severe_emotion = predicted_emotion in SEVERE_NEGATIVE_EMOTIONS
            
            # NEW CONDITIONAL LOGIC: Phone Suspect OR Sustained Drowsiness OR Severe Emotion
            is_phone_suspect_AND_low_ear = is_extreme_pitch and is_low_ear 

            # UNATTENTIVE TRIGGERS: 
            # 1. Pitch AND Low EAR (instantaneous for hidden device)
            # 2. Sustained Drowsiness (for sleeping/nodding off)
            # 3. Severe Emotion
            is_unattentive = is_phone_suspect_AND_low_ear or is_drowsy_sustained or is_severe_emotion

            if is_unattentive:
                reasons = []
                # Flag the highest priority reason first
                if is_phone_suspect_AND_low_ear: 
                    reasons.append(f"Conditional UNATTENTIVE (Pitch: {pitch}° AND Low EAR: {ear:.2f})") 
                
                # Note: is_drowsy_sustained will only be met if is_low_ear has been true for 30 frames
                # If both are true, the conditional message above is clearer.
                elif is_drowsy_sustained: 
                    reasons.append(f"Sustained Drowsiness (EAR: {ear:.2f}, Frames: {COUNTER})")
                
                if is_severe_emotion: 
                    reasons.append(f"Severe Emotion ({predicted_emotion})")
                
                # Fallback in case of multiple flags
                reason_str = "; ".join(reasons)
                
                overall_status = f"UNATTENTIVE ({reason_str})"
                status_color = (0, 0, 255) # RED
            else:
                overall_status = "ATTENTIVE"
                status_color = (0, 255, 0) # GREEN

            # --- 5. DRAW OUTPUT ---
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
            
            cv2.putText(
                frame,
                f"STUDENT STATUS: {overall_status}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
            )
            
            cv2.putText(frame, f"Emotion: {predicted_emotion} ({emotion_conf:.0%})", 
                        (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Head: {pitch}° Pitch, EAR: {ear:.2f} (Counter: {COUNTER})", 
                        (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # --- 6. OVERALL SYSTEM STATUS DISPLAY ---
        
        cv2.putText(frame, f"System Status: {overall_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Single-Student Attentiveness Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Closed.")

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