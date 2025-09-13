"""
MLOps Pipeline Launcher
Automated startup script for the complete MLOps pipeline
"""

import os
import sys
import subprocess
import argparse
import logging
import time
from pathlib import Path
import yaml
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsPipelineLauncher:
    """Main launcher for MLOps pipeline components"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_path = self.project_root / "config" / "config.yaml"
        self.processes = []
        
    def check_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'torch', 'transformers', 'fastapi', 'uvicorn', 
            'mlflow', 'wandb', 'scikit-learn', 'pandas'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.error("Run: pip install -r requirements.txt")
            return False
        
        logger.info("All dependencies are installed")
        return True
    
    def setup_environment(self):
        """Setup environment and create necessary directories"""
        logger.info("Setting up environment...")
        
        # Create necessary directories
        directories = [
            "data/raw", "data/processed", "models", "logs", 
            "mlruns", "evaluation_results", "monitoring_results"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Check if .env file exists
        env_file = self.project_root / ".env"
        if not env_file.exists():
            logger.warning(".env file not found. Creating from template...")
            template_file = self.project_root / ".env.example"
            if template_file.exists():
                import shutil
                shutil.copy(template_file, env_file)
                logger.info("Created .env from template. Please configure your API keys.")
        
        logger.info("Environment setup complete")
    
    def create_sample_data(self):
        """Create sample data for testing"""
        logger.info("Creating sample data...")
        
        import pandas as pd
        
        sample_data = pd.DataFrame({
            'text': [
                "This is a great product, I love it!",
                "Terrible quality, would not recommend",
                "Amazing service and fast delivery",
                "Poor customer support experience",
                "Excellent value for money",
                "Waste of money, very disappointed",
                "Outstanding quality and design",
                "Not worth the price, overrated",
                "Perfect for my needs, highly recommended",
                "Broke after one day of use"
            ] * 10,  # Create 100 samples
            'label': ['positive', 'negative'] * 50
        })
        
        data_path = self.project_root / "data" / "sample_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        logger.info(f"Sample data created at {data_path}")
        return str(data_path)
    
    def run_training(self, data_path: str):
        """Run model training"""
        logger.info("Starting model training...")
        
        try:
            # Add src to Python path
            sys.path.append(str(self.project_root / "src"))
            
            from training.trainer import MLOpsTrainer
            
            trainer = MLOpsTrainer(str(self.config_path))
            
            # Train default model
            results = trainer.train_model(
                model_name="bert_classifier",
                data_path=data_path,
                hyperopt=False
            )
            
            logger.info(f"Training completed: {results}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def run_evaluation(self, data_path: str):
        """Run model evaluation"""
        logger.info("Starting model evaluation...")
        
        try:
            sys.path.append(str(self.project_root / "src"))
            
            from evaluation.evaluator import ModelEvaluator
            
            evaluator = ModelEvaluator(str(self.config_path))
            
            # Evaluate model
            results = evaluator.evaluate_model(
                model_name="bert_classifier",
                test_data_path=data_path
            )
            
            logger.info("Evaluation completed")
            return True
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False
    
    def start_api_server(self):
        """Start FastAPI server"""
        logger.info("Starting API server...")
        
        try:
            # Start FastAPI server in background
            api_script = self.project_root / "src" / "deployment" / "api.py"
            
            process = subprocess.Popen([
                sys.executable, str(api_script)
            ], cwd=str(self.project_root))
            
            self.processes.append(process)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            logger.info("API server started on http://localhost:8000")
            logger.info("API documentation available at http://localhost:8000/docs")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def start_mlflow_ui(self):
        """Start MLflow UI"""
        logger.info("Starting MLflow UI...")
        
        try:
            process = subprocess.Popen([
                "mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"
            ], cwd=str(self.project_root))
            
            self.processes.append(process)
            
            time.sleep(2)
            logger.info("MLflow UI started on http://localhost:5000")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MLflow UI: {e}")
            return False
    
    def run_tests(self):
        """Run test suite"""
        logger.info("Running tests...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/", "-v"
            ], cwd=str(self.project_root), capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("All tests passed")
                return True
            else:
                logger.error(f"Tests failed: {result.stdout}{result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return False
    
    def cleanup(self):
        """Cleanup processes"""
        logger.info("Cleaning up...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except:
                pass
    
    def run_full_pipeline(self):
        """Run the complete MLOps pipeline"""
        logger.info("Starting MLOps Pipeline...")
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Setup environment
            self.setup_environment()
            
            # Create sample data
            data_path = self.create_sample_data()
            
            # Run training
            if not self.run_training(data_path):
                logger.error("Pipeline failed at training stage")
                return False
            
            # Run evaluation
            if not self.run_evaluation(data_path):
                logger.error("Pipeline failed at evaluation stage")
                return False
            
            # Start services
            self.start_mlflow_ui()
            self.start_api_server()
            
            logger.info("="*60)
            logger.info("MLOps Pipeline Started Successfully!")
            logger.info("="*60)
            logger.info("Services running:")
            logger.info("- API Server: http://localhost:8000")
            logger.info("- API Docs: http://localhost:8000/docs")
            logger.info("- MLflow UI: http://localhost:5000")
            logger.info("="*60)
            logger.info("Press Ctrl+C to stop all services")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
        
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MLOps Pipeline Launcher")
    parser.add_argument("--mode", choices=["full", "train", "eval", "api", "test"], 
                       default="full", help="Run mode")
    parser.add_argument("--data", help="Path to data file")
    
    args = parser.parse_args()
    
    launcher = MLOpsPipelineLauncher()
    
    try:
        if args.mode == "full":
            success = launcher.run_full_pipeline()
        elif args.mode == "train":
            launcher.setup_environment()
            data_path = args.data or launcher.create_sample_data()
            success = launcher.run_training(data_path)
        elif args.mode == "eval":
            data_path = args.data or launcher.create_sample_data()
            success = launcher.run_evaluation(data_path)
        elif args.mode == "api":
            launcher.setup_environment()
            success = launcher.start_api_server()
            if success:
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        elif args.mode == "test":
            success = launcher.run_tests()
        
        if success:
            logger.info("Operation completed successfully")
        else:
            logger.error("Operation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    finally:
        launcher.cleanup()

if __name__ == "__main__":
    main()
