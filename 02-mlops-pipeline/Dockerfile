FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV WANDB_DISABLED=true
ENV DAGSTER_HOME=/app/dagster_home

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed models dagster_home

EXPOSE 8000 3000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
