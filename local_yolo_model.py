# local_yolo_model.py
from ultralytics import YOLO

# Adjust this path if your run folder name is different
MODEL_PATH = "runs/detect/train/weights/best.pt"

def load_local_model():
    """Load the trained YOLOv8 model for Clash Royale cards."""
    return YOLO(MODEL_PATH)
