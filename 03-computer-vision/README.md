# Computer Vision System - Object Detection

Real-time object detection system with sophisticated post-processing and modern web interface.

## Overview

This system demonstrates production-ready computer vision capabilities using pre-trained Faster R-CNN models with advanced duplicate filtering and professional web deployment.

## Architecture

```
Input Image → Preprocessing → Faster R-CNN ResNet-50 FPN
     ↓
Detection Results → Post-processing → NMS + Custom Filtering
     ↓
FastAPI Response → Web Interface → Professional Visualization
```

## Tech Stack

- **PyTorch + Torchvision** - Deep learning framework and pre-trained models
- **Faster R-CNN ResNet-50 FPN** - State-of-the-art object detection architecture
- **OpenCV** - Image preprocessing and computer vision utilities
- **FastAPI** - Async web framework for model serving
- **NumPy** - Numerical operations and array processing
- **PIL** - Image loading and format handling

## Key Features

- Pre-trained model with transfer learning (COCO dataset, 80+ classes)
- Advanced post-processing with NMS and custom duplicate filtering
- Real-time processing with 3-5 second response time
- Configurable confidence and NMS thresholds
- Professional responsive web interface with drag-drop support
- Performance monitoring and inference time logging

## Quick Start

1. **Setup environment:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
# Start FastAPI backend
python src/api/main.py

# Open frontend in browser
open src/frontend/app.html
```

3. **Access the system:**
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Frontend: Open app.html in browser

## Performance

- **Processing Time:** 3-5 seconds per image (CPU)
- **Model Accuracy:** 85%+ on common objects
- **Supported Classes:** 80+ COCO dataset objects
- **Throughput:** Real-time processing capability

## Project Structure

```
03-computer-vision/
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI backend
│   ├── inference/
│   │   └── detector.py      # Object detection logic
│   ├── frontend/
│   │   └── app.html         # Web interface
│   └── utils/
│       └── visualization.py # Visualization tools
├── models/                  # Model weights
└── requirements.txt         # Dependencies
```

## Technical Implementation

**Model Architecture:**
- Faster R-CNN ResNet-50 FPN backbone
- Pre-trained on COCO dataset for robust detection
- CPU and GPU inference support

**Post-processing Pipeline:**
- Confidence threshold filtering
- Non-Maximum Suppression (NMS)
- Custom duplicate detection with confidence-based filtering

**Web Interface:**
- Modern responsive design
- Real-time confidence adjustment
- Professional visualization with statistics

## API Endpoints

- `POST /detect` - Object detection with configurable confidence
- `GET /health` - System health and model status
- `GET /stats` - API usage statistics and performance metrics
- `GET /` - API information and status

Part of the [AI Engineering Portfolio](../README.md) demonstrating production computer vision system development.
