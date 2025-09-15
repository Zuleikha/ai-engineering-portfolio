FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw models

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "src/main.py"]