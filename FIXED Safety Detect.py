import cv2
import torch
from ultralytics import YOLO
import os
import time
import requests
from datetime import datetime

# Inisialisasi kamera menggunakan OpenCV
cap = cv2.VideoCapture(0)  # Menggunakan kamera pertama (index 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fps = 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load model YOLO
safety_model = YOLO('safety_check_best.pt')
fall_model = YOLO('fall_detect_best.pt')

# Tentukan format video output
output_path = './result/combined_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Folder untuk menyimpan capture
capture_folder = './result/captures'
os.makedirs(capture_folder, exist_ok=True)

# URL API endpoint
api_url = "http://localhost:8000/dashboard/records/store"

def send_data_to_api(detection_type, capture_path):
    with open(capture_path, "rb") as image_file:
        files = {"capture": image_file}
        data = {
            "type": detection_type,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "capture": os.path.basename(capture_path)
        }
        response = requests.post(api_url, data=data, files=files)
        print(f"API Response: {response.status_code} - {response.text}")

# Fungsi untuk deteksi safety
def safety_detection(frame):
    safety_results = safety_model(frame)
    for result in safety_results[0].boxes:
        if result.cls.item() not in [0, 1, 2, 3, 4]:  # Abaikan kelas tertentu
            class_name = safety_model.names[int(result.cls.item())]
            confidence = result.conf.item()
            coordinates = result.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coordinates)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 128, 0), 2)

            # Capture and send data for safety check
            timestamp = int(time.time())
            capture_path = os.path.join(capture_folder, f"safety_detected_{timestamp}.png")
            cv2.imwrite(capture_path, frame)
            send_data_to_api("Safety Detection", capture_path)

# Fungsi untuk deteksi fall
def fall_detection(frame):
    fall_results = fall_model(frame)
    fall_detected = False  # Status awal untuk deteksi jatuh

    for detection in fall_results[0].boxes:
        if detection.conf > 0.5:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].numpy())
            label = fall_model.names[int(detection.cls.item())]

            if label == "Fall Detected":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Fall Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                fall_detected = True

                # Capture dan kirim data untuk deteksi jatuh
                timestamp = int(time.time())
                capture_path = os.path.join(capture_folder, f"fall_detected_{timestamp}.png")
                cv2.imwrite(capture_path, frame)
                send_data_to_api("Fall Detected", capture_path)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "No Fall Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if not fall_detected:
        cv2.putText(frame, "No Fall Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Proses video frame by frame
frame_skip = 2
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    if frame_count % frame_skip == 0:
        safety_detection(frame)
        fall_detection(frame)

    # Simpan hasil ke video output
    out.write(frame)
    cv2.imshow('Combined Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Bersihkan resource
cap.release()
out.release()
cv2.destroyAllWindows()
