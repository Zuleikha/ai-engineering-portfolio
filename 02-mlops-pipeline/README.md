# MLOps Pipeline Project

End-to-end MLOps system demonstrating modern workflow orchestration, automated model lifecycle management, and production monitoring capabilities.

## Project Structure

```
03-mlops-pipeline/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation utilities
│   ├── deployment/        # Deployment code
│   └── utils/             # Helper functions
├── config/                # Configuration files
├── infrastructure/        # Kubernetes & Terraform configs
├── monitoring/           # Monitoring dashboards
├── tests/                # Test suite
└── docker/               # Docker configurations
```

## Quick Start

1. **Environment Setup**
   ```bash
   conda env create -f environment.yml
   conda activate mlops-pipeline
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Pipeline**
   ```bash
   dagster dev
   ```

## Features

- Automated data pipeline with quality validation
- Distributed model training with hyperparameter optimization
- Model versioning and registry management
- Automated testing for model performance regression
- Deployment automation with blue-green deployment strategy
- Production monitoring with data drift and performance tracking

## Technology Stack

- **Orchestration**: Dagster
- **Experiment Tracking**: Weights & Biases / MLflow
- **Model Registry**: MLflow
- **CI/CD**: GitHub Actions
- **Monitoring**: Evidently AI
- **Infrastructure**: Kubernetes with Helm charts
- **Cloud Platform**: AWS (EKS, S3, RDS)
