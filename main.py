import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time
from tensorflow.keras.models import load_model

# ==== CONFIG ====
MODEL_PATH = "my_85percent_model.h5"   # change to your .h5 path
CAMERA_INDEX = 0                  # try 1,2 if 0 does not work
FRAME_W, FRAME_H = 640, 480
IMAGE_SIZE = 48                   # 48x48 for FER-style models

# ==== GLOBAL STATE FOR DASHBOARD ====
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_probs_history = deque(maxlen=50)
current_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
current_gray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

# ----------------------------------------------------
# DASHBOARD (matplotlib)
# ----------------------------------------------------
def create_dashboard():
    global current_frame, current_gray, emotion_probs_history

    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Emotion Detection Dashboard', fontsize=16, fontweight='bold')

    # Panel 1: Live Color Feed
    ax1.set_title('Live Feed')
    ax1.axis('off')
    img1 = ax1.imshow(current_frame[..., ::-1])  # BGR->RGB

    # Panel 2: Emotion Probabilities
    ax2.set_title('Emotion Probabilities')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Probability')
    bars = ax2.bar(
        emotion_labels,
        [0] * len(emotion_labels),
        color=['red', 'brown', 'orange', 'green', 'blue', 'purple', 'gray']
    )
    ax2.set_xticklabels(emotion_labels, rotation=45, ha='right')

    # Panel 3: Gray feed
    ax3.set_title('B&W Feed + Detection')
    ax3.axis('off')
    img3 = ax3.imshow(current_gray, cmap='gray')

    # Hide unused 4th subplot
    _.axis('off')

    def update(_frame):
        # Update color image
        img1.set_data(current_frame[..., ::-1])

        # Update gray image
        img3.set_data(current_gray)

        # Update emotion bars
        if emotion_probs_history:
            latest = emotion_probs_history[-1]
            for bar, prob in zip(bars, latest):
                bar.set_height(prob)
                bar.set_color('green' if prob > 0.3 else 'lightcoral')

        return [img1, img3, *bars]

    FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------
# LIVE EMOTION DETECTION LOOP
# ----------------------------------------------------
def live_detection(model):
    global current_frame, current_gray, emotion_probs_history

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("Error: camera not available")
        return

    # Start dashboard window
    import threading
    dashboard_thread = threading.Thread(target=create_dashboard, daemon=True)
    dashboard_thread.start()
    time.sleep(1)  # give matplotlib time to show

    print("Dashboard running. Press 'q' in OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # Update globals for dashboard
        current_frame = frame.copy()
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            current_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(40, 40)
        )

        frame_out = frame.copy()

        for (x, y, w, h) in faces:
            # Crop and preprocess ROI
            roi = current_gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=[0, -1])

            # Predict
            preds = model.predict(roi, verbose=0)[0]

            # Optional small neutral correction if you see sad bias:
            # preds[4] *= 0.9   # sad
            # preds[6] *= 1.1   # neutral
            # preds = preds / preds.sum()

            emotion_probs_history.append(preds.tolist())

            idx = np.argmax(preds)
            emotion = emotion_labels[idx]
            conf = preds[idx]

            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)

            cv2.rectangle(frame_out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame_out,
                f"{emotion}: {conf:.0%}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Show plain OpenCV window as backup
        cv2.imshow("Webcam Emotion Detection", frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    live_detection(model)
