from ultralytics import YOLO

# This will auto-download the file if missing or corrupted
model = YOLO('yolov8n.pt')
print("✅ YOLOv8n weights loaded successfully!")
