import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load kedua model YOLO
safety_model = YOLO('best.pt')
fall_model = YOLO('best_masday_colab.pt')

# Path ke video input atau kamera
video_path = './images/8964793-uhd_3840_2160_25fps.mp4'
cap = cv2.VideoCapture(video_path)

# Jika gagal membuka video, coba buka kamera laptop
if not cap.isOpened():
    print("Tidak dapat membuka video. Membuka kamera laptop...")
    cap = cv2.VideoCapture(0)

# Tentukan format video output
output_path = './result/combined_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) / 2  # Kurangi FPS untuk mengurangi kecepatan output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# List untuk menyimpan nilai confidence dari safety check
all_safety_confidences = []

frame_skip = 2  # Melewati setiap 2 frame untuk mempercepat pemrosesan
frame_count = 0
prev_positions = {}  # Menyimpan posisi sebelumnya dari setiap objek

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Melewati frame untuk mempercepat
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Deteksi dengan safety model
    safety_results = safety_model(frame)
    for result in safety_results[0].boxes:
        if result.cls.item() not in [0, 1, 2, 3, 4]:  # Abaikan kelas tertentu
            class_name = safety_model.names[int(result.cls.item())]
            confidence = result.conf.item()
            all_safety_confidences.append(confidence)
            coordinates = result.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coordinates)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Deteksi dengan fall model
    fall_results = fall_model(frame)
    for idx, detection in enumerate(fall_results[0].boxes):
        if detection.conf > 0.5:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].numpy())
            label = str(detection.cls)

            # Hitung posisi tengah objek (centroid)
            centroid_y = (y1 + y2) // 2

            # Cek pergerakan dari frame sebelumnya ke frame sekarang
            if idx in prev_positions:
                # Jika objek bergerak ke bawah secara cepat, tandai sebagai jatuh
                if centroid_y - prev_positions[idx] > height * 0.1:  # Gerakan ke bawah signifikan
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Fall Detected ({label})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Update posisi sebelumnya
            prev_positions[idx] = centroid_y

    # Tulis frame yang sudah diprediksi ke video output
    out.write(frame)

    # Menampilkan video
    cv2.imshow('Combined Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Bersihkan resource
cap.release()
out.release()
cv2.destroyAllWindows()

# Buat dan simpan histogram dari nilai confidence safety check setelah video selesai diproses
if all_safety_confidences:
    plt.figure(figsize=(8, 6))
    plt.hist(all_safety_confidences, bins=10, range=(0, 1), color='blue', edgecolor='black')
    plt.title('Histogram Confidence Scores - Safety Check')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    histogram_path = './result/confidence_histogram.png'
    plt.savefig(histogram_path)
    plt.show()
    print(f"Histogram disimpan di: {histogram_path}")
else:
    print("Tidak ada data confidence untuk ditampilkan.")
