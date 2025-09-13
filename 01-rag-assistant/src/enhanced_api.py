"""Enhanced FastAPI with PDF upload support."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from pdf_processor import PDFProcessor

load_dotenv()

app = FastAPI(
    title="Enhanced RAG Assistant API",
    description="RAG system with PDF upload support",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
print("Initializing Enhanced RAG system...")
rag_system = RAGSystem()
pdf_processor = PDFProcessor()
print("Enhanced RAG system ready!")

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: float = 0.0

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    chunks: Optional[int] = None
    characters: Optional[int] = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Enhanced RAG Assistant API",
        "version": "2.0.0",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "features": ["PDF Upload", "Text Processing", "AI Chat"],
        "supported_formats": [".pdf", ".txt"]
    }

@app.post("/upload-pdf", response_model=FileUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    try:
        # Check if file is PDF
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process PDF
        result = pdf_processor.process_uploaded_pdf(file_content, file.filename)
        
        if not result:
            raise HTTPException(status_code=400, detail="Failed to process PDF file")
        
        # Add to existing RAG system
        rag_system.vector_store.add_documents(result["chunks"])
        rag_system.vector_store.save_to_file("models/vector_store.json")
        
        return FileUploadResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            filename=file.filename,
            chunks=result["total_chunks"],
            characters=result["total_chars"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using RAG system."""
    try:
        result = rag_system.query(request.question, request.conversation_id)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enhanced RAG Assistant API - Now with PDF Upload!",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "upload": "/upload-pdf"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Enhanced RAG Assistant API...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
