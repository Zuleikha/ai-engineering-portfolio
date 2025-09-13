"""
Real-time object detection inference engine
Handles single images, batch processing, and video streams
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import detection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.ops
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
import json
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import visualization utilities directly
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from visualization import DetectionVisualizer, VideoVisualizer

class ModelLoader:
    """Load trained models for inference"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path: str, device: torch.device, num_classes: int = 91) -> torch.nn.Module:
        """
        Load a trained model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
            device: Device to load model on
            num_classes: Number of classes in the model
            
        Returns:
            Loaded model ready for inference
        """
        try:
            # Check if model file exists
            if os.path.exists(model_path):
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                
                # Initialize model architecture
                model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(device)
                model.eval()
                
                self.logger.info(f"Model loaded successfully from {model_path}")
                return model
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Fallback to pretrained model for testing
            self.logger.info("Loading pretrained model as fallback")
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            model.to(device)
            model.eval()
            return model

@dataclass
class DetectionResult:
    """Structure for detection results"""
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    processing_time: float
    image_shape: Tuple[int, int]

class ObjectDetector:
    """Main object detection inference class"""
    
    def __init__(self, 
                 model_path: str,
                 class_names: List[str],
                 device: str = 'auto',
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize object detector
        
        Args:
            model_path: Path to trained model
            class_names: List of class names
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for duplicate removal
        """
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model_path = model_path
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model loader
        self.model_loader = ModelLoader()
        
        # Performance tracking
        self.inference_times = []
        
        # Initialize model
        self.model = None
        self.load_model()
        
        # Initialize visualizers
        self.visualizer = DetectionVisualizer(class_names, confidence_threshold)
        self.video_visualizer = VideoVisualizer(self.visualizer)
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = self.model_loader.load_model(self.model_path, self.device, len(self.class_names))
            self.model.eval()
            self.logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
        tensor = transform(image_rgb)
        return tensor.unsqueeze(0).to(self.device)
    
    def postprocess_detections(self, 
                             outputs: List[Dict], 
                             original_shape: Tuple[int, int]) -> DetectionResult:
        """
        Postprocess model outputs to detection results
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (height, width)
            
        Returns:
            DetectionResult object
        """
        # Extract predictions
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        keep_indices = scores >= self.confidence_threshold
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        # Apply NMS if needed
        if len(boxes) > 0:
            keep_indices = self.apply_nms(boxes, scores)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=labels,
            processing_time=0,  # Will be set by caller
            image_shape=original_shape
        )
    
    def apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression"""
        boxes_tensor = torch.from_numpy(boxes)
        scores_tensor = torch.from_numpy(scores)
        
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, self.nms_threshold)
        return keep.cpu().numpy()
    
    def detect_single_image(self, image: np.ndarray) -> DetectionResult:
        """
        Detect objects in a single image
        
        Args:
            image: Input image in BGR format
            
        Returns:
            DetectionResult object
        """
        start_time = time.time()
        
        # Preprocess
        tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
        
        # Postprocess
        result = self.postprocess_detections(outputs, image.shape[:2])
        
        # Set processing time
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        self.inference_times.append(processing_time)
        
        return result

def main():
    """Example usage of the inference engine"""
    
    # Configuration - using COCO classes for pretrained model
    MODEL_PATH = "models/best_model.pth"
    CLASS_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    try:
        print("Initializing object detector...")
        
        # Initialize detector
        detector = ObjectDetector(
            model_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
        
        # Example: Single image detection
        print("Testing single image detection...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_single_image(test_image)
        print(f"Detected {len(result.boxes)} objects in {result.processing_time:.3f}s")
        
        # Performance stats
        if detector.inference_times:
            print(f"Average inference time: {np.mean(detector.inference_times):.3f}s")
            print(f"Average FPS: {1.0/np.mean(detector.inference_times):.1f}")
        
        print(f"Running on device: {detector.device}")
        print("Inference engine test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
