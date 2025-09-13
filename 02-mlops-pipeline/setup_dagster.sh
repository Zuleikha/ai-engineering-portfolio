# Create Dagster assets and pipeline orchestration

echo "Setting up Dagster orchestration for MLOps pipeline..."

# Create Dagster definitions and assets
cat > src/dagster_definitions.py << 'EOF'
"""Dagster definitions for MLOps pipeline"""
from dagster import (
    asset, 
    AssetIn, 
    Config, 
    Definitions,
    load_assets_from_modules,
    job,
    schedule,
    sensor,
    DefaultSensorStatus
)
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path
import yaml

# Import our modules
from src.data import huggingface_datasets
from src.config import config

class PipelineConfig(Config):
    """Configuration for the ML pipeline"""
    dataset_name: str = "imdb"
    model_name: str = "distilbert-base-uncased"
    sample_size: int = 1000
    epochs: int = 1
    batch_size: int = 8

@asset
def pipeline_config() -> Dict[str, Any]:
    """Load pipeline configuration"""
    with open('config/pipeline.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

@asset
def dataset_info(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get information about the dataset to be used"""
    hf_config = pipeline_config['data']['huggingface']
    
    info = {
        "dataset_name": hf_config['dataset_name'],
        "sample_size": hf_config['sample_size'],
        "text_column": hf_config['text_column'],
        "label_column": hf_config['label_column']
    }
    
    return info

@asset
def raw_dataset(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """Load raw dataset from Hugging Face"""
    from datasets import load_dataset
    
    dataset_name = dataset_info['dataset_name']
    sample_size = dataset_info['sample_size']
    
    # Load dataset
    if sample_size:
        dataset = load_dataset(dataset_name, split=f"train[:{sample_size}]")
        test_dataset = load_dataset(dataset_name, split=f"test[:{sample_size//4}]")
    else:
        full_dataset = load_dataset(dataset_name)
        dataset = full_dataset['train']
        test_dataset = full_dataset['test']
    
    # Convert to pandas for easier handling in pipeline
    train_df = dataset.to_pandas()
    test_df = test_dataset.to_pandas()
    
    # Save to local storage
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_parquet("data/processed/train_data.parquet")
    test_df.to_parquet("data/processed/test_data.parquet")
    
    return {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_path": "data/processed/train_data.parquet",
        "test_path": "data/processed/test_data.parquet",
        "columns": list(train_df.columns)
    }

@asset
def processed_train_data(raw_dataset: Dict[str, Any]) -> pd.DataFrame:
    """Load and preprocess training data"""
    train_df = pd.read_parquet(raw_dataset['train_path'])
    
    # Basic preprocessing
    train_df = train_df.dropna()
    train_df['text_length'] = train_df['text'].str.len()
    
    # Filter out extremely short or long texts
    train_df = train_df[
        (train_df['text_length'] >= 10) & 
        (train_df['text_length'] <= 5000)
    ]
    
    return train_df

@asset
def processed_test_data(raw_dataset: Dict[str, Any]) -> pd.DataFrame:
    """Load and preprocess test data"""
    test_df = pd.read_parquet(raw_dataset['test_path'])
    
    # Apply same preprocessing as training data
    test_df = test_df.dropna()
    test_df['text_length'] = test_df['text'].str.len()
    test_df = test_df[
        (test_df['text_length'] >= 10) & 
        (test_df['text_length'] <= 5000)
    ]
    
    return test_df

@asset
def data_quality_report(
    processed_train_data: pd.DataFrame,
    processed_test_data: pd.DataFrame
) -> Dict[str, Any]:
    """Generate data quality report"""
    
    def analyze_dataframe(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        return {
            f"{split_name}_samples": len(df),
            f"{split_name}_missing_text": df['text'].isna().sum(),
            f"{split_name}_missing_labels": df['label'].isna().sum(),
            f"{split_name}_avg_text_length": df['text_length'].mean(),
            f"{split_name}_label_distribution": df['label'].value_counts().to_dict()
        }
    
    train_stats = analyze_dataframe(processed_train_data, "train")
    test_stats = analyze_dataframe(processed_test_data, "test")
    
    quality_report = {**train_stats, **test_stats}
    
    # Save report
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/data_quality_report.yaml", "w") as f:
        yaml.dump(quality_report, f)
    
    return quality_report

# Define the assets as a module
all_assets = [
    pipeline_config,
    dataset_info, 
    raw_dataset,
    processed_train_data,
    processed_test_data,
    data_quality_report
]

# Create a job that materializes all assets
@job
def data_pipeline_job():
    """Job to run the complete data pipeline"""
    return data_quality_report()

# Create a schedule to run daily
@schedule(
    job=data_pipeline_job,
    cron_schedule="0 2 * * *"  # Run at 2 AM daily
)
def daily_data_pipeline_schedule():
    """Schedule to run data pipeline daily"""
    return {}

# Create the Dagster definitions
defs = Definitions(
    assets=all_assets,
    jobs=[data_pipeline_job],
    schedules=[daily_data_pipeline_schedule]
)
EOF

# Create a simple Dagster workspace
cat > workspace.yaml << 'EOF'
load_from:
  - python_file: src/dagster_definitions.py
EOF

# Create a script to run Dagster
cat > scripts/start_dagster.py << 'EOF'
#!/usr/bin/env python3
"""Start Dagster development server"""
import subprocess
import sys
import os

def start_dagster():
    """Start Dagster development server"""
    # Set environment variables
    os.environ['DAGSTER_HOME'] = os.getcwd()
    
    print("Starting Dagster development server...")
    print("This will open a web interface at http://localhost:3000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start Dagster dev server
        subprocess.run([
            sys.executable, "-m", "dagster", "dev",
            "-f", "src/dagster_definitions.py"
        ])
    except KeyboardInterrupt:
        print("\nStopping Dagster server...")
    except Exception as e:
        print(f"Error starting Dagster: {e}")

if __name__ == "__main__":
    start_dagster()
EOF

chmod +x scripts/start_dagster.py

# Create a script to test individual assets
cat > scripts/test_assets.py << 'EOF'
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
        print("✓ Pipeline config asset: PASSED")
    else:
        print("✗ Pipeline config asset: FAILED")
        print(result.failure_data)

def test_dataset_info():
    """Test dataset info asset"""
    from src.dagster_definitions import pipeline_config, dataset_info
    
    result = materialize([pipeline_config, dataset_info])
    if result.success:
        print("✓ Dataset info asset: PASSED")
    else:
        print("✗ Dataset info asset: FAILED")

def test_raw_dataset():
    """Test raw dataset loading"""
    from src.dagster_definitions import pipeline_config, dataset_info, raw_dataset
    
    print("Testing raw dataset loading (this may take a moment)...")
    result = materialize([pipeline_config, dataset_info, raw_dataset])
    if result.success:
        print("✓ Raw dataset asset: PASSED")
    else:
        print("✗ Raw dataset asset: FAILED")

def test_all_assets():
    """Test all assets together"""
    print("Testing complete asset pipeline...")
    result = materialize(all_assets)
    if result.success:
        print("✓ Complete pipeline: PASSED")
        print("All assets materialized successfully!")
    else:
        print("✗ Complete pipeline: FAILED")

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
EOF

chmod +x scripts/test_assets.py

echo "Dagster orchestration setup completed!"
echo ""
echo "Next steps:"
echo "1. Test individual assets: python scripts/test_assets.py --config"
echo "2. Test dataset loading: python scripts/test_assets.py --dataset" 
echo "3. Test complete pipeline: python scripts/test_assets.py --all"
echo "4. Start Dagster UI: python scripts/start_dagster.py"
echo ""
echo "The Dagster UI will be available at http://localhost:3000"