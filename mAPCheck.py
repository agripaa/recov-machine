import torch
from ultralytics import YOLO

# Load model yang sudah di-train
model = YOLO('best.pt')

# Evaluasi model pada dataset validasi untuk menghitung mAP
metrics = model.val(data='path/to/data.yaml')  # Path ke file YAML dataset

# Menampilkan hasil mAP
print(f"mAP@0.5: {metrics['metrics/mAP50']:.3f}")
print(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95']:.3f}")
