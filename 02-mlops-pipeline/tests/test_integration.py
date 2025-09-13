"""
Integration tests for complete MLOps pipeline
"""

import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_manager import ModelManager
from data.data_processor import DataManager
from evaluation.evaluator import ModelEvaluator

class TestMLOpsPipelineIntegration:
    
    @pytest.fixture
    def test_environment(self):
        """Set up complete test environment"""
        temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        config_path = Path(temp_dir) / "config.yaml"
        test_config = {
            "project": {"name": "test-project", "version": "1.0.0"},
            "models": {
                "available": {
                    "test_model": {
                        "source": "local",
                        "model_path": "models/test.pt",
                        "task": "classification",
                        "architecture": "lstm"
                    }
                },
                "default": "test_model"
            },
            "data": {
                "raw_path": "data/raw",
                "processed_path": "data/processed",
                "test_size": 0.2,
                "random_state": 42
            },
            "training": {
                "batch_size": 2,
                "learning_rate": 0.01,
                "epochs": 1
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create test data
        data_path = Path(temp_dir) / "test_data.csv"
        test_data = pd.DataFrame({
            'text': [
                "This is positive text",
                "This is negative text",
                "Another positive example",
                "Another negative example",
                "More positive content",
                "More negative content"
            ],
            'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
        })
        test_data.to_csv(data_path, index=False)
        
        yield {
            'temp_dir': temp_dir,
            'config_path': str(config_path),
            'data_path': str(data_path)
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_pipeline(self, test_environment):
        """Test complete pipeline from data to evaluation"""
        config_path = test_environment['config_path']
        data_path = test_environment['data_path']
        
        # Initialize components
        model_manager = ModelManager(config_path)
        data_manager = DataManager(config_path)
        
        # Test model loading
        model = model_manager.load_model("test_model")
        assert model is not None
        
        # Test data processing
        data_splits = data_manager.process_data(data_path, "custom")
        assert data_splits is not None
        assert 'train' in data_splits
        assert 'validation' in data_splits
        assert 'test' in data_splits
    
    def test_model_comparison_workflow(self, test_environment):
        """Test model comparison functionality"""
        config_path = test_environment['config_path']
        
        # Initialize model manager
        model_manager = ModelManager(config_path)
        
        # Test getting available models
        available_models = model_manager.get_available_models()
        assert "test_model" in available_models
        
        # Test loading model for comparison
        model = model_manager.load_model("test_model")
        assert model is not None
