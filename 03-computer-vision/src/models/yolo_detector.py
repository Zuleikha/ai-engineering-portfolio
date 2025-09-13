"""YOLO object detection implementation."""

print("Starting YOLO detector...")

try:
    import torch
    print("PyTorch imported successfully")
except ImportError as e:
    print(f"Error importing PyTorch: {e}")

try:
    from ultralytics import YOLO
    print("Ultralytics imported successfully")
except ImportError as e:
    print(f"Error importing Ultralytics: {e}")

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_model(self):
        try:
            self.model = YOLO(self.model_name)
            print(f"Loaded YOLO model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")

def test_yolo_detector():
    detector = YOLODetector("yolov8n.pt")
    detector.load_model()
    print("YOLO detector initialized successfully!")
    return detector

if __name__ == "__main__":
    test_yolo_detector()
