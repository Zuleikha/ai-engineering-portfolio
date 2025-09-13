# RAG Assistant - Natural Language Processing System

A production-ready Retrieval-Augmented Generation system with conversation memory, document processing, and intelligent search capabilities.

## Overview

This system processes documents, creates searchable embeddings, and provides contextual responses using OpenAI's GPT models with conversation history tracking.

## Architecture

Streamlit Frontend → FastAPI Backend → RAG System
↓
Document Processing → Vector Store → LLM Integration
↓                 ↓              ↓
OpenAI Embeddings → Cosine Search → GPT-3.5-turbo

## Tech Stack

- **FastAPI** - Async backend API server
- **Streamlit** - Interactive web interface  
- **OpenAI API** - Embeddings (text-embedding-ada-002) and LLM (GPT-3.5-turbo)
- **LangChain** - Document processing and text splitting
- **NumPy** - Vector operations and similarity calculations

## Key Features

- Smart document chunking (1000 chars, 200 overlap)
- Vector search with cosine similarity
- Per-user conversation memory (10 exchanges)
- Source attribution in responses
- Confidence scoring based on similarity
- Persistent vector store with auto-save

## Quick Start

1. **Setup environment:**
```bash
pip install -r requirements.txt
