from picamera2 import Picamera2
import cv2
import onnxruntime as ort
import os
import time
import requests
from datetime import datetime
import numpy as np

# Inisialisasi kamera menggunakan Picamera2
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"format": 'RGB888', "size": (320, 240)})
picam2.configure(camera_config)
picam2.start()

picam2.set_controls({
    "AwbEnable": True,  # Atur White Balance otomatis
    "AeEnable": True,   # Atur Eksposur otomatis
})

fps = 30  # FPS lebih realistis untuk perangkat keras terbatas
width, height = 640, 480

# Load model ONNX menggunakan ONNX Runtime
safety_model_path = "safety_check_best.onnx"
fall_model_path = "fall_detect_best.onnx"

safety_session = ort.InferenceSession(safety_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
fall_session = ort.InferenceSession(fall_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Tentukan format video output
output_path = './result/combined_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Folder untuk menyimpan capture
capture_folder = './result/captures'
os.makedirs(capture_folder, exist_ok=True)

# URL API endpoint
api_url = "http://localhost:8000/dashboard/records/store"

# Definisikan nama kelas berdasarkan ID di model
custom_class_names = {
    0: "Shoes",
    1: "Glasses",
    2: "Gloves",
    3: "Helmet",
    4: "Vest"
}

def get_class_name(class_id):
    return custom_class_names.get(class_id, "Unknown")

# Fungsi untuk mengirim data ke API
def send_data_to_api(detection_type, capture_path):
    try:
        with open(capture_path, "rb") as image_file:
            files = {"capture": image_file}
            data = {
                "type": detection_type,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "capture": os.path.basename(capture_path)
            }
            response = requests.post(api_url, data=data, files=files)
            print(f"API Response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"API Error: {e}")

# Fungsi untuk memproses inferensi menggunakan ONNX Runtime
def run_inference_onnx(session, frame):
    # Preprocess frame (resize dan normalisasi)
    input_frame = cv2.resize(frame, (640, 640))
    input_frame = np.transpose(input_frame, (2, 0, 1))  # HWC ke CHW
    input_frame = input_frame.astype(np.float32) / 255.0  # Normalisasi
    input_frame = np.expand_dims(input_frame, axis=0)  # Tambahkan batch dimension

    # Jalankan inferensi
    outputs = session.run(None, {"images": input_frame})
    return outputs

# Fungsi untuk deteksi safety
def safety_detection(frame):
    outputs = run_inference_onnx(safety_session, frame)
    detections = outputs[0]  # Output deteksi
    
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        if confidence > 0.5:  # Threshold confidence
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = get_class_name(int(class_id))

            # Tampilkan hasil deteksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Fungsi untuk deteksi jatuh
def fall_detection(frame):
    outputs = run_inference_onnx(fall_session, frame)
    detections = outputs[0]  # Output deteksi

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        if confidence > 0.5:  # Threshold confidence
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = "Fall Detected"

            # Tampilkan hasil deteksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Proses video
frame_count = 0
while True:
    frame = picam2.capture_array()

    # Jalankan deteksi hanya pada setiap 3 frame
    if frame_count % 3 == 0:
        safety_detection(frame)
        fall_detection(frame)

    # Simpan hasil ke video output
    out.write(frame)
    cv2.imshow('Combined Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Bersihkan resource
out.release()
cv2.destroyAllWindows()
picam2.close()
