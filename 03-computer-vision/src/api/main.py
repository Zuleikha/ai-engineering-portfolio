"""
FastAPI backend for object detection service
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import os
from PIL import Image
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(Path(__file__).parent.parent / 'inference'))

from detector import ObjectDetector, DetectionResult

# Pydantic models
class DetectionResponse(BaseModel):
    success: bool
    detections: List[Dict]
    processing_time: float
    image_shape: List[int]
    message: Optional[str] = None

# Global variables
app_state = {"detector": None, "start_time": time.time(), "request_count": 0}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting object detection service...")
    try:
        class_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard'
        ]
        
        app_state["detector"] = ObjectDetector(
            model_path="models/best_model.pth",
            class_names=class_names,
            confidence_threshold=0.5,
            nms_threshold=0.2  # FIXED: Changed from 0.4 to 0.2 for better duplicate removal
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
    
    yield

app = FastAPI(title="Object Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_detection_response(result: DetectionResult) -> Dict:
    """Format detection results with duplicate filtering"""
    
    detections = []
    for i in range(len(result.boxes)):
        class_id = int(result.classes[i])
        class_name = app_state["detector"].class_names[class_id] if class_id < len(app_state["detector"].class_names) else "unknown"
        
        if class_name not in ["background", "N/A"]:
            detections.append({
                "bbox": result.boxes[i].tolist(),
                "confidence": float(result.scores[i]),
                "class_name": class_name,
                "class_id": class_id
            })
    
    # ADDED: Additional filtering for very similar detections
    filtered_detections = []
    for detection in detections:
        is_duplicate = False
        for existing in filtered_detections:
            # If same class and very similar confidence, it might be a duplicate
            if (detection["class_name"] == existing["class_name"] and 
                abs(detection["confidence"] - existing["confidence"]) < 0.15):
                # Keep the one with higher confidence
                if detection["confidence"] > existing["confidence"]:
                    filtered_detections.remove(existing)
                    break
                else:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered_detections.append(detection)
    
    return {
        "detections": filtered_detections,
        "count": len(filtered_detections),
        "processing_time": result.processing_time,
        "image_shape": list(result.image_shape)
    }

@app.get("/")
async def root():
    return {"message": "Object Detection API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": app_state["detector"] is not None}

@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence_threshold: float = 0.5):
    if not app_state["detector"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detection
        app_state["detector"].confidence_threshold = confidence_threshold
        result = app_state["detector"].detect_single_image(image)
        
        # FIXED: Use the new formatting function
        response_data = format_detection_response(result)
        
        return DetectionResponse(
            success=True,
            detections=response_data["detections"],
            processing_time=result.processing_time,
            image_shape=response_data["image_shape"],
            message=f"Detected {response_data['count']} objects"
        )
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    if app_state["detector"] is None:
        return {"error": "Detector not loaded"}
    
    return {
        "total_requests": app_state["request_count"],
        "uptime": time.time() - app_state["start_time"],
        "device": str(app_state["detector"].device)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)