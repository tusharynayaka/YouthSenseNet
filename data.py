"""
🚀 COMPLETE: Train from YOUR images + 4-Panel Live Dashboard
Folder structure: images/angry/ disgust/ fear/ happy/ sad/ surprise/ neutral/
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time

# === YOUR PATH - UPDATE LINE 15 ===
DATA_FOLDER = r"C:\\Users\\tusha.TUSHAR\\OneDrive\\Desktop\\Tushar\\Code\\Images"
SAVE_MODEL_PATH = "my_85percent_model.h5"

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Global variables for dashboard
emotion_probs_history = deque(maxlen=50)
current_frame = None
current_gray = None

# === 1. TRAINING FUNCTION ===
def train_model():
    """Train high-accuracy model from YOUR images"""
    print(f"🚀 Training from: {DATA_FOLDER}")
    
    # Check dataset first
    total_images = 0
    for emotion in EMOTIONS:
        folder = os.path.join(DATA_FOLDER, emotion)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  📁 {emotion}: {count} images")
            total_images += count
        else:
            print(f"  ❌ {emotion} folder missing!")
    
    print(f"\n📊 TOTAL: {total_images} images")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    train_gen = datagen.flow_from_directory(
        DATA_FOLDER, target_size=(48, 48), color_mode='grayscale',
        batch_size=32, class_mode='categorical', subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        DATA_FOLDER, target_size=(48, 48), color_mode='grayscale',
        batch_size=32, class_mode='categorical', subset='validation'
    )
    
    # High-accuracy model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
        
        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(), MaxPooling2D(2,2), Dropout(0.3),
        
        Flatten(),
        Dense(512, activation='relu'), Dropout(0.5),
        Dense(256, activation='relu'), Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(train_gen, validation_data=val_gen, epochs=50, verbose=1)
    
    model.save(SAVE_MODEL_PATH)
    print(f"✅ SAVED: {SAVE_MODEL_PATH} - Final Val Accuracy: {max(history.history['val_accuracy']):.1%}")
    return model

# === 2. 4-PANEL DASHBOARD ===
def create_dashboard(model):
    """4-Panel live dashboard"""
    global current_frame, current_gray, emotion_probs_history
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 Emotion Detection Dashboard - YOUR Trained Model', fontsize=18, fontweight='bold')
    
    # Panel 1: Live Color Feed
    ax1.set_title('🟢 Live Color Feed', fontweight='bold', pad=10)
    ax1.axis('off')
    img1 = ax1.imshow(np.zeros((480, 640, 3)))
    
    # Panel 2: Emotion Progress Bars
    ax2.set_title('📊 Live Emotion Probabilities', fontweight='bold', pad=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Probability')
    bars = ax2.bar(EMOTIONS, [0]*7, color='lightcoral', alpha=0.8)
    ax2.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    
    # Panel 3: Emotion Evolution Scatter
    ax3.set_title('📈 Emotion Confidence Over Time', fontweight='bold', pad=10)
    ax3.set_xlabel('Frames (Last 50)')
    ax3.set_ylabel('Max Probability')
    line, = ax3.plot([], [], 'b-', linewidth=3, label='Trend')
    scatter = ax3.scatter([], [], c='red', s=40, alpha=0.7, label='History')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: B&W Feed + Detection
    ax4.set_title('⚫ Grayscale + Face Detection', fontweight='bold', pad=10)
    ax4.axis('off')
    img4 = ax4.imshow(np.zeros((480, 640)))
    
    def update(frame_num):
        global current_frame, current_gray
        
        if current_frame is None:
            return img1, img4, *bars, line, scatter
        
        # Panel 1: Update live feed
        img1.set_data(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
        
        # Panel 4: Update B&W
        img4.set_data(current_gray)
        
        # Panel 2: Update progress bars
        if emotion_probs_history:
            probs = list(emotion_probs_history)[-1]
            for bar, prob in zip(bars, probs):
                bar.set_height(prob)
                bar.set_color('limegreen' if prob > 0.3 else 'lightcoral')
            ax2.set_title(f'📊 Emotion Probabilities (Max: {max(probs):.1%})', fontweight='bold')
        
        # Panel 3: Update scatter plot
        if len(emotion_probs_history) > 1:
            x = list(range(len(emotion_probs_history)))
            y = [max(p) for p in emotion_probs_history]
            line.set_data(x, y)
            scatter.set_offsets(np.column_stack([x, y]))
            ax3.set_xlim(0, max(50, len(x)))
            ax3.set_ylim(0, 1)
        
        return [img1, img4] + bars + [line, scatter]
    
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show(block=False)
    
    try:
        plt.pause(0.01)
    except:
        pass

# === 3. LIVE DETECTION WITH DASHBOARD ===
def live_detection_dashboard(model):
    """Main live detection with 4-panel dashboard"""
    global current_frame, current_gray, emotion_probs_history
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Start dashboard thread
    dashboard_thread = threading.Thread(target=create_dashboard, args=(model,), daemon=True)
    dashboard_thread.start()
    
    print("🎨 4-PANEL DASHBOARD ACTIVE! Close matplotlib window + press 'q' to quit.")
    time.sleep(2)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        current_frame = frame.copy()
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(current_gray, 1.3, 5, minSize=(40,40))
        
        for (x, y, w, h) in faces:
            # Preprocess
            face_roi = cv2.resize(current_gray[y:y+h, x:x+w], (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=[0, -1])
            
            # Predict ALL 7 emotions
            pred = model.predict(face_roi, verbose=0)[0]
            emotion_probs_history.append(pred.tolist())
            
            # Dominant emotion
            emotion_idx = np.argmax(pred)
            emotion = EMOTIONS[emotion_idx]
            confidence = pred[emotion_idx]
            
            # Draw on frame
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Top 3 emotions
            top3 = np.argsort(pred)[-3:][::-1]
            for i, idx in enumerate(top3):
                emo_name = EMOTIONS[idx]
                prob = pred[idx]
                y_pos = y + h + 25 + i * 25
                cv2.putText(frame, f"{emo_name}: {prob:.0%}", (x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Live Feed + Dashboard Active', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚀 Emotion Detection: Training + 4-Panel Dashboard")
    print(f"📁 Dataset: {DATA_FOLDER}")
    
    # STEP 1: Train model
    if os.path.exists(SAVE_MODEL_PATH):
        print("✅ Loading existing model...")
        model = load_model(SAVE_MODEL_PATH)
    else:
        print("\n🔥 TRAINING NEW MODEL...")
        model = train_model()
        input("\n✅ Training complete! Press Enter for 4-Panel Dashboard...")
    
    # STEP 2: Start 4-Panel Dashboard
    print("\n🎨 LAUNCHING 4-PANEL DASHBOARD...")
    live_detection_dashboard(model)
