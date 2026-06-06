import cv2
from deepface import DeepFace
from collections import deque, Counter

# Initialize webcam
cap = cv2.VideoCapture(0)

# Buffer for temporal smoothing (tracks last 10 frames)
emotion_buffer = deque(maxlen=10)

# Emotion color mapping (BGR format)
COLOR_MAP = {
    "happy": (128, 255, 0),       # Green
    "sad": (255, 100, 0),        # Blue
    "angry": (0, 0, 255),        # Red
    "surprise": (255, 255, 0),   # Cyan
    "neutral": (220, 220, 220)   # Slate White
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Optional: Flip frame for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    try:
        # Step 1: Analyze frame using an efficient backend (enforcing detection)
        # To boost FPS, you can run this every 2 or 3 frames instead of every single frame
        predictions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        
        # Filter out valid face predictions
        valid_faces = [face for face in predictions if face['face_confidence'] > 0.4]

        if not valid_faces:
            # --- EDGE CASE: NO FACE DETECTED ---
            cv2.putText(frame, "SEARCHING FOR FACE...", (int(w*0.3), int(h*0.5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)
        else:
            # --- EDGE CASE: MULTIPLE FACES (Sort by face area size, largest first) ---
            valid_faces.sort(key=lambda x: x['region']['w'] * x['region']['h'], reverse=True)

            for idx, face in enumerate(valid_faces):
                region = face['region']
                x, y, fw, fh = region['x'], region['y'], region['w'], region['h']
                
                if idx == 0:
                    # PRIMARY USER
                    raw_emotion = face['dominant_emotion']
                    # Keep only requested classes
                    if raw_emotion not in COLOR_MAP:
                        raw_emotion = "neutral"
                        
                    # Smooth predictions over time
                    emotion_buffer.append(raw_emotion)
                    smoothed_emotion = Counter(emotion_buffer).most_common(1)[0][0]
                    
                    color = COLOR_MAP.get(smoothed_emotion, (255, 255, 255))
                    
                    # Draw a nice bounding box for primary user
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                    cv2.putText(frame, f"You: {smoothed_emotion.upper()}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                else:
                    # BACKGROUND USERS (Muted presentation)
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (150, 150, 150), 1)
                    cv2.putText(frame, "Guest", (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
                                
    except Exception as e:
        # Fallback to keep app running smoothly if DeepFace hiccups
        pass

    # Display the final frame
    cv2.imshow("Emotion Mirror", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()