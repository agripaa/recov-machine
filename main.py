import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model YOLO
model = YOLO('best.pt')

# Path ke video (biarkan kosong atau gunakan path yang salah untuk menguji pembukaan kamera otomatis)
video_path = './images/'  # Contoh path yang salah untuk memicu kamera

# Buka video input atau kamera
cap = cv2.VideoCapture(video_path)

# Jika gagal membuka video, coba buka kamera laptop
if not cap.isOpened():
    print("Tidak dapat membuka video. Membuka kamera laptop...")
    cap = cv2.VideoCapture(0)  # '0' untuk kamera default laptop

# Tentukan format video output (optional)
output_path = './result/output_vid.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) if video_path and cap.isOpened() else 20.0  # Set FPS untuk video atau default kamera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# List untuk menyimpan nilai confidence dari semua frame
all_confidences = []

# Loop untuk menangkap video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Hentikan jika tidak ada frame yang tersisa

    # Buat prediksi pada frame
    results = model(frame)

    # Filter results to exclude classes '0' to '4' dan ambil kepercayaan
    for result in results[0].boxes:
        if result.cls.item() not in [0, 1, 2, 3, 4]:  # Abaikan kelas 0 hingga 4
            class_name = model.names[int(result.cls.item())]
            confidence = result.conf.item()
            all_confidences.append(confidence)  # Tambahkan ke list untuk rekapitulasi
            coordinates = result.xyxy[0].tolist()  # Convert tensor to list

            # Extract coordinates
            x1, y1, x2, y2 = map(int, coordinates)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"

            # Put label on frame
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Tulis frame yang sudah diprediksi ke video output
    out.write(frame)

    # Menampilkan video
    cv2.imshow('Predicted Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Tekan 'q' untuk berhenti
        break

# Bersihkan resource
cap.release()
out.release()
cv2.destroyAllWindows()

# Buat dan simpan histogram dari semua nilai confidence setelah video selesai diproses
if all_confidences:
    plt.figure(figsize=(8, 6))
    plt.hist(all_confidences, bins=10, range=(0, 1), color='blue', edgecolor='black')
    plt.title('Rekapitulasi Histogram Confidence Scores')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    
    # Simpan histogram ke folder result
    histogram_path = './histogram/confidence_histogram.png'
    plt.savefig(histogram_path)
    plt.show()
    
    print(f"Histogram disimpan di: {histogram_path}")
else:
    print("Tidak ada data confidence untuk ditampilkan.")
