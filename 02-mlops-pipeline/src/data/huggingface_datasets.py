"""Hugging Face datasets integration"""
from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

class HuggingFaceDatasetConfig:
    """Configuration for Hugging Face datasets"""
    def __init__(self, dataset_name="imdb", subset=None, sample_size=None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.sample_size = sample_size
        self.text_column = "text"
        self.label_column = "label"

class HuggingFaceDatasetManager:
    """Manage Hugging Face datasets for ML pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.dataset = None
    
    def load_dataset_from_hub(self):
        """Load dataset from Hugging Face Hub"""
        if self.config.subset:
            dataset = load_dataset(self.config.dataset_name, self.config.subset)
        else:
            dataset = load_dataset(self.config.dataset_name)
        
        # Sample data if specified
        if self.config.sample_size:
            sampled_dataset = {}
            for split_name in dataset.keys():
                if len(dataset[split_name]) > self.config.sample_size:
                    sampled_dataset[split_name] = dataset[split_name].select(
                        range(self.config.sample_size)
                    )
                else:
                    sampled_dataset[split_name] = dataset[split_name]
            dataset = DatasetDict(sampled_dataset)
        
        self.dataset = dataset
        return dataset
