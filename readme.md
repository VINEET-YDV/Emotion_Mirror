# Emotion Mirror 🪞

A real-time Computer Vision application that detects faces via webcam and predicts current emotions (Happy, Sad, Angry, Surprised, or Neutral) using a polished, user-friendly interface. 

This project was built focusing on system design, robust edge-case handling, and smooth User Experience (UX) rather than just raw model inference.

## 🌟 Key Features & Design Decisions

To make the experience feel like a polished product rather than a raw script, I implemented the following solutions for common edge cases:

*   **Temporal Smoothing (Anti-Jitter):** Raw model predictions often fluctuate frame-by-frame. I implemented a rolling mode filter using `collections.deque` to buffer the last 10 frames. The app displays the most frequent emotion in the buffer, resulting in a stable, jitter-free UI.
*   **Multiple Face Handling:** When multiple people enter the frame, the app calculates the bounding box area to identify the person closest to the camera. The closest person is assigned as the "Primary User" with full emotion tracking, while others are gracefully labeled as "Guests" with muted UI elements.
*   **"Scanning" Fallback State:** Instead of crashing or showing a raw camera feed when no face is detected, the app shifts into a clean "Searching for face..." fallback UI state.
*   **Color-Coded Feedback:** Bounding boxes dynamically change color based on the predicted emotion (e.g., Green for Happy, Red for Angry) to provide intuitive visual feedback.

## 🛠️ Tech Stack

*   **Python 3.10**
*   **OpenCV (`cv2`):** For real-time video capture and UI overlays.
*   **DeepFace:** For robust facial detection and emotion classification.


## 🚀 Setup & Installation

### Prerequisites
* Python 3.9 - 3.11 recommended
* A working integrated or USB webcam

### Environment Configuration
Navigate to your project directory and create a clean isolated Python virtual environment:

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

#Install Dependencies
pip install opencv-python deepface tf-keras

#Execute the main script via your terminal:
python main.py
