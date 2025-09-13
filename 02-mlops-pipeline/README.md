# MLOps Pipeline - Production Machine Learning Operations

End-to-end MLOps system with Dagster orchestration, automated model lifecycle management, and containerized deployment.

## Overview

This system demonstrates modern MLOps practices including asset-based pipeline orchestration, model training automation, and production serving with health monitoring.

## Architecture

Data Sources → Dagster Assets → Model Training → Model Registry
↓
FastAPI Serving ← Model Deployment ← Model Validation
↓
Health Checks → Monitoring → Performance Metrics

## Tech Stack

- **Dagster** - Modern data orchestration and pipeline management
- **FastAPI** - High-performance model serving API
- **Docker** - Containerization and deployment
- **Scikit-learn** - Machine learning models
- **Pandas** - Data processing and feature engineering
- **Uvicorn** - ASGI server for production deployment

## Key Features

- Asset-based pipelines with declarative data lineage
- Automated model training and validation workflows
- Model versioning and registry management
- Production serving with health checks
- Containerized deployment with Docker Compose
- Integrated logging and performance metrics

## Quick Start

1. **Setup environment:**
```bash
pip install -r requirements.txt
