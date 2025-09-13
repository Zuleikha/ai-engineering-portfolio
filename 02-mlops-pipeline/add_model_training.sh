# Integrate Hugging Face model training with the existing pipeline

echo "Adding model training assets to Dagster pipeline..."

# Update the Dagster definitions to include model training
cat > src/dagster_definitions.py << 'EOF'
"""Dagster definitions for MLOps pipeline with model training"""
from dagster import (
    asset, 
    AssetIn, 
    Config, 
    Definitions,
    load_assets_from_modules,
    job,
    op,
    schedule,
    sensor,
    DefaultSensorStatus
)
from typing import Dict, Any, List, Tuple
import pandas as pd
from pathlib import Path
import yaml
import torch
from datetime import datetime

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

@asset
def model_config(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration"""
    training_config = pipeline_config['training']
    
    model_config = {
        "model_name": training_config['pretrained_model'],
        "task_type": training_config['task_type'],
        "num_labels": 2,  # Binary classification for IMDB
        "max_length": training_config['max_length'],
        "learning_rate": training_config['learning_rate'],
        "batch_size": training_config['batch_size'],
        "num_epochs": training_config['epochs'],
        "warmup_steps": training_config['warmup_steps']
    }
    
    return model_config

@asset
def pretrained_model_setup(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup pretrained model and tokenizer"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = model_config['model_name']
    num_labels = model_config['num_labels']
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Save tokenizer for later use
    Path("models/pretrained").mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained("models/pretrained")
    
    # Get model info
    model_info = {
        "model_name": model_name,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
        "tokenizer_vocab_size": len(tokenizer),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    return model_info

@asset
def training_datasets(
    processed_train_data: pd.DataFrame,
    processed_test_data: pd.DataFrame,
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare datasets for training"""
    from transformers import AutoTokenizer
    from datasets import Dataset
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/pretrained")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=model_config['max_length']
        )
    
    # Create training dataset
    train_dataset = Dataset.from_pandas(processed_train_data[['text', 'label']])
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    
    # Create validation dataset (use test data as validation)
    val_dataset = Dataset.from_pandas(processed_test_data[['text', 'label']])
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Save datasets
    Path("data/processed/tokenized").mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk("data/processed/tokenized/train")
    val_dataset.save_to_disk("data/processed/tokenized/validation")
    
    return {
        "train_dataset_path": "data/processed/tokenized/train",
        "val_dataset_path": "data/processed/tokenized/validation",
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "tokenizer_path": "models/pretrained"
    }

@asset
def trained_model(
    training_datasets: Dict[str, Any],
    model_config: Dict[str, Any],
    pretrained_model_setup: Dict[str, Any]
) -> Dict[str, Any]:
    """Train the model"""
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from datasets import load_from_disk
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(training_datasets['tokenizer_path'])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['model_name'],
        num_labels=model_config['num_labels']
    )
    
    # Load datasets
    train_dataset = load_from_disk(training_datasets['train_dataset_path'])
    val_dataset = load_from_disk(training_datasets['val_dataset_path'])
    
    # Define compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Setup training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/training_output_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=model_config['num_epochs'],
        per_device_train_batch_size=model_config['batch_size'],
        per_device_eval_batch_size=model_config['batch_size'],
        warmup_steps=model_config['warmup_steps'],
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=model_config['learning_rate'],
        report_to=None  # Disable wandb/tensorboard for now
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    
    # Save the final model
    final_model_path = "models/trained_model"
    Path(final_model_path).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_model_path)
    
    # Prepare training summary
    training_summary = {
        "model_path": final_model_path,
        "training_output_dir": output_dir,
        "train_runtime": train_result.metrics.get('train_runtime', 0),
        "train_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
        "eval_accuracy": eval_result.get('eval_accuracy', 0),
        "eval_f1": eval_result.get('eval_f1', 0),
        "eval_precision": eval_result.get('eval_precision', 0),
        "eval_recall": eval_result.get('eval_recall', 0),
        "eval_loss": eval_result.get('eval_loss', 0),
        "model_config": model_config,
        "training_timestamp": timestamp
    }
    
    # Save training summary
    with open(f"{final_model_path}/training_summary.yaml", "w") as f:
        yaml.dump(training_summary, f)
    
    return training_summary

@asset
def model_evaluation_report(trained_model: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive model evaluation report"""
    
    evaluation_report = {
        "model_performance": {
            "accuracy": trained_model['eval_accuracy'],
            "f1_score": trained_model['eval_f1'],
            "precision": trained_model['eval_precision'],
            "recall": trained_model['eval_recall'],
            "loss": trained_model['eval_loss']
        },
        "training_efficiency": {
            "training_time_seconds": trained_model['train_runtime'],
            "samples_per_second": trained_model['train_samples_per_second']
        },
        "model_metadata": {
            "model_path": trained_model['model_path'],
            "timestamp": trained_model['training_timestamp'],
            "model_name": trained_model['model_config']['model_name']
        },
        "quality_assessment": {
            "ready_for_production": trained_model['eval_accuracy'] > 0.8,
            "performance_grade": "A" if trained_model['eval_f1'] > 0.85 else "B" if trained_model['eval_f1'] > 0.75 else "C"
        }
    }
    
    # Save evaluation report
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/model_evaluation_report.yaml", "w") as f:
        yaml.dump(evaluation_report, f)
    
    return evaluation_report

# Define all assets
all_assets = [
    pipeline_config,
    dataset_info, 
    raw_dataset,
    processed_train_data,
    processed_test_data,
    data_quality_report,
    model_config,
    pretrained_model_setup,
    training_datasets,
    trained_model,
    model_evaluation_report
]

# Create the Dagster definitions
defs = Definitions(
    assets=all_assets
)
EOF

# Create a script to test model training
cat > scripts/test_model_training.py << 'EOF'
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
EOF

chmod +x scripts/test_model_training.py

echo "Model training assets added successfully!"
echo ""
echo "Next steps:"
echo "1. Test model config: python scripts/test_model_training.py --config"
echo "2. Test pretrained setup: python scripts/test_model_training.py --setup"
echo "3. Test training prep: python scripts/test_model_training.py --prepare"
echo "4. Run full training: python scripts/test_model_training.py --train"
echo ""
echo "The full training will take 3-5 minutes for the small dataset."