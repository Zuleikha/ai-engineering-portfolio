#!/usr/bin/env python3
"""Test model training assets"""
import sys
import os
sys.path.append('.')

from dagster import materialize
from src.dagster_definitions import all_assets

def test_model_config():
    """Test model configuration"""
    from src.dagster_definitions import pipeline_config, model_config
    
    result = materialize([pipeline_config, model_config])
    if result.success:
        print("Model config asset: PASSED")
    else:
        print("Model config asset: FAILED")

def test_pretrained_setup():
    """Test pretrained model setup"""
    from src.dagster_definitions import pipeline_config, model_config, pretrained_model_setup
    
    print("Testing pretrained model setup (downloading model)...")
    result = materialize([pipeline_config, model_config, pretrained_model_setup])
    if result.success:
        print("Pretrained model setup: PASSED")
    else:
        print("Pretrained model setup: FAILED")

def test_training_preparation():
    """Test training dataset preparation"""
    from src.dagster_definitions import (
        pipeline_config, dataset_info, raw_dataset, 
        processed_train_data, processed_test_data,
        model_config, pretrained_model_setup, training_datasets
    )
    
    print("Testing training dataset preparation...")
    result = materialize([
        pipeline_config, dataset_info, raw_dataset,
        processed_train_data, processed_test_data,
        model_config, pretrained_model_setup, training_datasets
    ])
    if result.success:
        print("Training datasets preparation: PASSED")
    else:
        print("Training datasets preparation: FAILED")

def test_full_training():
    """Test complete model training pipeline"""
    print("Testing complete model training pipeline...")
    print("This will take several minutes...")
    
    result = materialize(all_assets)
    if result.success:
        print("Complete training pipeline: PASSED")
        print("Model training completed successfully!")
    else:
        print("Complete training pipeline: FAILED")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test model training pipeline")
    parser.add_argument("--config", action="store_true", help="Test model config only")
    parser.add_argument("--setup", action="store_true", help="Test pretrained model setup")
    parser.add_argument("--prepare", action="store_true", help="Test training preparation")
    parser.add_argument("--train", action="store_true", help="Test full training")
    
    args = parser.parse_args()
    
    if args.config:
        test_model_config()
    elif args.setup:
        test_pretrained_setup()
    elif args.prepare:
        test_training_preparation()
    elif args.train:
        test_full_training()
    else:
        print("Testing model training components...")
        test_model_config()
        test_pretrained_setup()
        test_training_preparation()
        print("\nTo run full training: python scripts/test_model_training.py --train")
