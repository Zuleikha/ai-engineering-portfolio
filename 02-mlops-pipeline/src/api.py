"""FastAPI service for MLOps pipeline"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import yaml
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Pipeline API",
    description="Production API for the MLOps training pipeline",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    texts: List[str]
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, str]

class TrainingRequest(BaseModel):
    dataset_name: str = "imdb"
    sample_size: int = 100
    epochs: int = 1
    batch_size: int = 16

class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
    def load_model(self, model_path: str = "models/trained_model"):
        """Load trained model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            if not Path(model_path).exists():
                logger.warning(f"Model not found at {model_path}")
                return False
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, texts: List[str], return_probabilities: bool = False) -> List[Dict[str, Any]]:
        """Make predictions on texts"""
        if not self.model_loaded:
            raise ValueError("Model not loaded")
        
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                
                result = {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "predicted_class": predicted_class,
                    "prediction": "positive" if predicted_class == 1 else "negative"
                }
                
                if return_probabilities:
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    result["probabilities"] = {
                        "negative": float(probabilities[0]),
                        "positive": float(probabilities[1])
                    }
                
                predictions.append(result)
        
        return predictions

model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting MLOps Pipeline API...")
    
    if Path("models/trained_model").exists():
        model_service.load_model()
    else:
        logger.warning("No trained model found. Use /train endpoint to train a model first.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MLOps Pipeline API",
        "version": "1.0.0",
        "model_loaded": model_service.model_loaded,
        "device": str(model_service.device)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_service.model_loaded,
        "device": str(model_service.device)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint"""
    if not model_service.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first using /train endpoint."
        )
    
    try:
        predictions = model_service.predict(
            request.texts,
            request.return_probabilities
        )
        
        return PredictionResponse(
            predictions=predictions,
            model_info={
                "device": str(model_service.device),
                "model_path": "models/trained_model"
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainingRequest):
    """Train a new model"""
    try:
        from dagster import materialize
        from src.dagster_definitions import all_assets
        
        config_path = Path("config/pipeline.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['data']['huggingface']['sample_size'] = request.sample_size
        config['training']['epochs'] = request.epochs
        config['training']['batch_size'] = request.batch_size
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Starting training with {request.sample_size} samples...")
        result = materialize(all_assets)
        
        if result.success:
            model_service.load_model()
            
            return {
                "status": "success",
                "message": "Model trained successfully",
                "config": {
                    "dataset_name": request.dataset_name,
                    "sample_size": request.sample_size,
                    "epochs": request.epochs,
                    "batch_size": request.batch_size
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
