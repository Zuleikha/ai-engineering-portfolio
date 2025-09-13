#!/bin/bash

echo "Starting MLOps Pipeline Development Environment..."

# Check if we're already in the right directory
if [ ! -f "src/dagster_definitions.py" ]; then
    echo "Navigating to project directory..."
    cd /a/EngineerPortFolio/mlops-pipeline
fi

# Activate virtual environment
if [ ! "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source mlops-env/Scripts/activate
else
    echo "Virtual environment already active: $VIRTUAL_ENV"
fi

# Show current status
echo "=== Project Status ==="
echo "Directory: $(pwd)"
echo "Virtual env: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo "Git branch: $(git branch --show-current)"

# Check if Docker is running
if docker info >/dev/null 2>&1; then
    echo "Docker: Running"
else
    echo "Docker: Not running - start Docker Desktop if needed"
fi

echo ""
echo "=== Quick Commands ==="
echo "Test pipeline: python scripts/test_model_training.py --train"
echo "Start API: python src/api.py"
echo "Build containers: docker-compose build"
echo "Start containers: docker-compose up -d"
echo ""
echo "MLOps environment ready!"
