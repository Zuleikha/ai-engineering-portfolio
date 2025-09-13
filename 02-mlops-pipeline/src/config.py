"""Configuration management for MLOps pipeline"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class for MLOps pipeline"""
    
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def pipeline_name(self) -> str:
        return self.get('pipeline.name', 'mlops-pipeline')
    
    @property
    def model_type(self) -> str:
        return self.get('training.model_type', 'pytorch')
    
    @property
    def batch_size(self) -> int:
        return self.get('training.batch_size', 32)

# Global config instance
config = Config()
