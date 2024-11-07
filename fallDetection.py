import cv2  
import torch 
from ultralytics import YOLO

def detect_fall_video_file(input_video_path, output_video_path):
    # Buka video input
    cap = cv2.VideoCapture(input_video_path)
    
    # Dapatkan informasi dasar tentang video
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frame per detik
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Lebar frame
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Tinggi frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec untuk format mp4

    # Inisialisasi writer untuk output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Baca model YOLO
    model = YOLO('best_masday_colab.pt')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Jalankan deteksi pada setiap frame
        results = model(frame)
        
        # Gambar kotak deteksi pada frame
        for detection in results[0].boxes:
            if detection.conf > 0.5:  # Kepercayaan deteksi
                # Mengubah tensor menjadi numpy array dan kemudian mengonversinya ke integer
                x1, y1, x2, y2 = map(int, detection.xyxy[0].numpy())  # Koordinat kotak deteksi
                label = str(detection.cls)  # Mengubah kelas deteksi menjadi string
                
                # Gambar kotak dan label 'Fall Detected' jika mendeteksi jatuh
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Fall Detected ({label})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Tulis frame hasil deteksi ke output video
        out.write(frame)
        
        # (Optional) Tampilkan frame secara langsung (real-time) untuk debugging
        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Tutup semua proses video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Jalankan fungsi deteksi
video_path = './images/santri jatuh dari tangga #shorts.mp4'
output_vid = './result/output_fall.mp4'
detect_fall_video_file(video_path, output_vid)
