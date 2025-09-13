"""Test all module imports"""
import sys
import os
sys.path.append('.')

def test_imports():
    try:
        # Test basic imports
        import torch
        import transformers
        import datasets
        import dagster
        print("Core libraries: OK")
        
        # Test our modules
        from src.config import config
        print(f"Config loaded: {config.pipeline_name}")
        
        # Test HF modules exist
        import src.data.huggingface_datasets
        print("HF datasets module: OK")
        
        if os.path.exists('src/models/huggingface_models.py'):
            import src.models.huggingface_models
            print("HF models module: OK")
        
        print("All imports successful!")
        
    except Exception as e:
        print(f"Import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()
