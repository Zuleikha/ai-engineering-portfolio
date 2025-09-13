"""Simple startup script for the RAG API."""

import uvicorn
import sys
import os

# Add current directory to Python path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
