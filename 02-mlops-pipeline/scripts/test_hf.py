#!/usr/bin/env python3
"""Simple test for Hugging Face integration"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_integration():
    try:
        from src.data.huggingface_datasets import HuggingFaceDatasetManager, HuggingFaceDatasetConfig
        print("Hugging Face modules imported successfully!")
        
        # Test dataset loading
        config = HuggingFaceDatasetConfig(dataset_name="imdb", sample_size=10)
        manager = HuggingFaceDatasetManager(config)
        
        print("Testing dataset download...")
        dataset = manager.load_dataset_from_hub()
        print(f"Dataset loaded: {len(dataset['train'])} train examples")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_integration()
