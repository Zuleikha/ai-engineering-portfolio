"""
FastAPI deployment service for MLOps pipeline
Serves trained models with comprehensive monitoring and logging
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import logging
import time
import yaml
from pathlib import Path
import mlflow
import mlflow.pytorch
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
import asyncio
from contextlib import asynccontextmanager
import json

from ..models.model_manager import ModelManager
from ..monitoring.drift_detector import DriftDetector

# Metrics for monitoring
REQUEST_COUNT = Counter('model_requests_total', 'Total model requests', ['model', 'endpoint'])
REQUEST_LATENCY = Histogram('model_request_duration_seconds', 'Request latency', ['model'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['model', 'predicted_class'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    text: str
    model_name: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]
    confidence: float
    model_used: str
    processing_time: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time: float

class ModelStatus(BaseModel):
    model_name: str
    status: str
    last_used: Optional[str]
    total_requests: int

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    uptime: float
    total_requests: int

# Global state
app_state = {
    "model_manager": None,
    "drift_detector": None,
    "start_time": time.time(),
    "total_requests": 0,
    "model_stats": {},
    "config": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    logger.info("Starting MLOps deployment service...")
    
    try:
        # Load configuration
        with open("config/config.yaml", 'r') as f:
            app_state["config"] = yaml.safe_load(f)
        
        # Initialize model manager
        app_state["model_manager"] = ModelManager()
        
        # Load default model
        default_model = app_state["config"]["models"]["default"]
        app_state["model_manager"].load_model(default_model)
        app_state["model_manager"].set_active_model(default_model)
        
        # Initialize drift detector
        app_state["drift_detector"] = DriftDetector()
        
        # Start Prometheus metrics server
        start_http_server(8001)
        
        logger.info("MLOps deployment service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MLOps deployment service...")

# Create FastAPI app
app = FastAPI(
    title="MLOps Model Deployment API",
    description="Production model serving with monitoring and drift detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for model manager
def get_model_manager():
    if app_state["model_manager"] is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return app_state["model_manager"]

# API Endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "MLOps Model Deployment API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "models": "/models",
            "metrics": "/metrics"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app_state["start_time"]
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(app_state["model_manager"].models) if app_state["model_manager"] else 0,
        uptime=uptime,
        total_requests=app_state["total_requests"]
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager=Depends(get_model_manager)
):
    """Single prediction endpoint"""
    
    start_time = time.time()
    model_name = request.model_name or app_state["config"]["models"]["default"]
    
    try:
        # Update metrics
        REQUEST_COUNT.labels(model=model_name, endpoint='predict').inc()
        
        # Get model
        if model_name not in model_manager.models:
            model_manager.load_model(model_name)
        
        model = model_manager.models[model_name]
        
        # Make prediction
        with REQUEST_LATENCY.labels(model=model_name).time():
            probabilities = model.predict([request.text])
            prediction = int(np.argmax(probabilities[0]))
            confidence = float(np.max(probabilities[0]))
        
        processing_time = time.time() - start_time
        
        # Update prediction metrics
        PREDICTION_COUNT.labels(model=model_name, predicted_class=prediction).inc()
        
        # Update stats
        app_state["total_requests"] += 1
        if model_name not in app_state["model_stats"]:
            app_state["model_stats"][model_name] = {"requests": 0, "last_used": None}
        
        app_state["model_stats"][model_name]["requests"] += 1
        app_state["model_stats"][model_name]["last_used"] = pd.Timestamp.now().isoformat()
        
        # Schedule drift detection in background
        background_tasks.add_task(
            check_drift, 
            request.text, 
            prediction, 
            model_name
        )
        
        response = PredictionResponse(
            prediction=prediction,
            probability=probabilities[0].tolist(),
            confidence=confidence,
            model_used=model_name,
            processing_time=processing_time
        )
        
        logger.info(f"Prediction made: model={model_name}, prediction={prediction}, confidence={confidence:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager=Depends(get_model_manager)
):
    """Batch prediction endpoint"""
    
    start_time = time.time()
    model_name = request.model_name or app_state["config"]["models"]["default"]
    
    try:
        # Update metrics
        REQUEST_COUNT.labels(model=model_name, endpoint='predict_batch').inc()
        
        # Get model
        if model_name not in model_manager.models:
            model_manager.load_model(model_name)
        
        model = model_manager.models[model_name]
        
        # Make batch predictions
        with REQUEST_LATENCY.labels(model=model_name).time():
            probabilities = model.predict(request.texts)
            predictions = [int(np.argmax(prob)) for prob in probabilities]
            confidences = [float(np.max(prob)) for prob in probabilities]
        
        total_processing_time = time.time() - start_time
        
        # Create individual responses
        responses = []
        for i, (text, pred, prob, conf) in enumerate(zip(request.texts, predictions, probabilities, confidences)):
            individual_response = PredictionResponse(
                prediction=pred,
                probability=prob.tolist(),
                confidence=conf,
                model_used=model_name,
                processing_time=total_processing_time / len(request.texts)  # Approximate per-item time
            )
            responses.append(individual_response)
            
            # Update prediction metrics
            PREDICTION_COUNT.labels(model=model_name, predicted_class=pred).inc()
        
        # Update stats
        app_state["total_requests"] += 1
        if model_name not in app_state["model_stats"]:
            app_state["model_stats"][model_name] = {"requests": 0, "last_used": None}
        
        app_state["model_stats"][model_name]["requests"] += len(request.texts)
        app_state["model_stats"][model_name]["last_used"] = pd.Timestamp.now().isoformat()
        
        # Schedule drift detection for batch
        for text, pred in zip(request.texts, predictions):
            background_tasks.add_task(check_drift, text, pred, model_name)
        
        batch_response = BatchPredictionResponse(
            predictions=responses,
            total_processing_time=total_processing_time
        )
        
        logger.info(f"Batch prediction completed: model={model_name}, batch_size={len(request.texts)}")
        
        return batch_response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelStatus])
async def list_models(model_manager=Depends(get_model_manager)):
    """List available models and their status"""
    
    available_models = model_manager.get_available_models()
    model_statuses = []
    
    for model_name in available_models:
        stats = app_state["model_stats"].get(model_name, {"requests": 0, "last_used": None})
        
        status = ModelStatus(
            model_name=model_name,
            status="loaded" if model_name in model_manager.models else "available",
            last_used=stats["last_used"],
            total_requests=stats["requests"]
        )
        model_statuses.append(status)
    
    return model_statuses

@app.post("/models/{model_name}/load")
async def load_model(model_name: str, model_manager=Depends(get_model_manager)):
    """Load a specific model"""
    
    try:
        model_manager.load_model(model_name)
        logger.info(f"Model {model_name} loaded successfully")
        
        return {"message": f"Model {model_name} loaded successfully", "status": "loaded"}
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/models/{model_name}/activate")
async def activate_model(model_name: str, model_manager=Depends(get_model_manager)):
    """Set a model as the default active model"""
    
    try:
        if model_name not in model_manager.models:
            model_manager.load_model(model_name)
        
        model_manager.set_active_model(model_name)
        logger.info(f"Model {model_name} set as active")
        
        return {"message": f"Model {model_name} is now active", "status": "active"}
        
    except Exception as e:
        logger.error(f"Failed to activate model {model_name}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/stats")
async def get_stats():
    """Get detailed service statistics"""
    
    uptime = time.time() - app_state["start_time"]
    
    stats = {
        "service_uptime": uptime,
        "total_requests": app_state["total_requests"],
        "models_loaded": len(app_state["model_manager"].models) if app_state["model_manager"] else 0,
        "model_statistics": app_state["model_stats"],
        "memory_usage": "Not implemented",  # Could add psutil for memory monitoring
        "active_model": getattr(app_state["model_manager"], 'active_model', None)
    }
    
    return stats

# Background task for drift detection
async def check_drift(text: str, prediction: int, model_name: str):
    """Background task to check for data drift"""
    
    try:
        if app_state["drift_detector"]:
            # This is a simplified drift check
            # In practice, you'd accumulate data and check drift periodically
            drift_detected = app_state["drift_detector"].check_text_drift(text)
            
            if drift_detected:
                logger.warning(f"Potential drift detected for model {model_name}")
                # Could trigger retraining or alerts here
                
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "success": False}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "success": False}
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
