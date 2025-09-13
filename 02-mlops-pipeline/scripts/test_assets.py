#!/usr/bin/env python3
"""Test Dagster assets individually"""
import sys
import os
sys.path.append('.')

from dagster import materialize
from src.dagster_definitions import all_assets

def test_pipeline_config():
    """Test pipeline configuration asset"""
    from src.dagster_definitions import pipeline_config
    
    result = materialize([pipeline_config])
    if result.success:
        print("Pipeline config asset: PASSED")
    else:
        print("Pipeline config asset: FAILED")
        print(result.failure_data)

def test_dataset_info():
    """Test dataset info asset"""
    from src.dagster_definitions import pipeline_config, dataset_info
    
    result = materialize([pipeline_config, dataset_info])
    if result.success:
        print("Dataset info asset: PASSED")
    else:
        print("Dataset info asset: FAILED")

def test_raw_dataset():
    """Test raw dataset loading"""
    from src.dagster_definitions import pipeline_config, dataset_info, raw_dataset
    
    print("Testing raw dataset loading (this may take a moment)...")
    result = materialize([pipeline_config, dataset_info, raw_dataset])
    if result.success:
        print("Raw dataset asset: PASSED")
    else:
        print("Raw dataset asset: FAILED")

def test_all_assets():
    """Test all assets together"""
    print("Testing complete asset pipeline...")
    result = materialize(all_assets)
    if result.success:
        print("Complete pipeline: PASSED")
        print("All assets materialized successfully!")
    else:
        print("Complete pipeline: FAILED")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Dagster assets")
    parser.add_argument("--config", action="store_true", help="Test config asset only")
    parser.add_argument("--dataset", action="store_true", help="Test dataset assets")
    parser.add_argument("--all", action="store_true", help="Test all assets")
    
    args = parser.parse_args()
    
    if args.config:
        test_pipeline_config()
    elif args.dataset:
        test_raw_dataset()
    elif args.all:
        test_all_assets()
    else:
        print("Testing individual assets...")
        test_pipeline_config()
        test_dataset_info()
        test_raw_dataset()
