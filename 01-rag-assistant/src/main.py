"""Main FastAPI application for RAG Assistant."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Import our RAG system
from rag_system import RAGSystem

load_dotenv()

app = FastAPI(
    title="RAG Assistant API",
    description="Production-ready RAG system with LLM integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
print("Initializing RAG system...")
rag_system = RAGSystem()
print("RAG system ready!")

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: float = 0.0

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RAG Assistant API",
        "version": "1.0.0",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "rag_system_initialized": True
    }

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

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history for a user."""
    history = rag_system.get_conversation_history(conversation_id)
    return {"conversation_id": conversation_id, "history": history}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to RAG Assistant API - Real AI Powered!",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting RAG Assistant API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
