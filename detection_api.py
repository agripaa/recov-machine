# Install FastAPI and Uvicorn if you haven't already
# pip install fastapi uvicorn

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
import cv2
import torch
import os
import time
from ultralytics import YOLO

app = FastAPI()

# Model Loading
fall_model = YOLO('best_masday_colab.pt')
safety_model = YOLO('best.pt')
capture_folder = './result/captures'
os.makedirs(capture_folder, exist_ok=True)

# In-Memory Storage for Detection Results
detection_results = []

# Data Model for JSON
class Detection(BaseModel):
    type: str
    date: str
    capture: str  # path to image capture

@app.get("/detection_results")
async def get_detections():
    """Endpoint for Laravel to fetch detection data."""
    return detection_results

def process_frame(model, frame, detection_type):
    """Process frame with specified model and capture result if detected."""
    results = model(frame)
    for detection in results[0].boxes:
        if detection.conf > 0.5:
            # Capture image
            timestamp = int(time.time())
            capture_path = os.path.join(capture_folder, f"{detection_type}_{timestamp}.png")
            cv2.imwrite(capture_path, frame)

            # Append to results
            detection_data = {
                "type": detection_type,
                "date": datetime.now().isoformat(),
                "capture": capture_path
            }
            detection_results.append(detection_data)
            return detection_data
    return None

def detect_fall_and_safety(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process Fall Detection
        fall_detection = process_frame(fall_model, frame, "Fall Detection")
        if fall_detection:
            print("Fall detected and stored:", fall_detection)

        # Process Safety Check
        safety_detection = process_frame(safety_model, frame, "Safety Check")
        if safety_detection:
            print("Safety check detected and stored:", safety_detection)

        # Display the frame (optional for debugging)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = './images/8964793-uhd_3840_2160_25fps.mp4'
    detect_fall_and_safety(video_path)
