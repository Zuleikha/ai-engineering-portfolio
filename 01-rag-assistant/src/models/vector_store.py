"""Simple vector storage and retrieval."""

import numpy as np
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class SimpleVectorStore:
    """Simple in-memory vector store for RAG."""
    
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def add_documents(self, processed_docs: List[Dict[str, Any]]):
        """Add processed documents to the store."""
        for doc in processed_docs:
            self.documents.append({
                "content": doc["content"],
                "metadata": doc["metadata"]
            })
            self.embeddings.append(doc["embedding"])
        
        print(f"Added {len(processed_docs)} documents to vector store")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return []
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for most relevant documents."""
        if not self.documents:
            return []
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Return top documents with scores
        results = []
        for idx, score in top_results:
            results.append({
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx]["metadata"],
                "similarity_score": score
            })
        
        return results
    
    def save_to_file(self, filepath: str):
        """Save vector store to file."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Vector store saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load vector store from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            
            print(f"Vector store loaded from {filepath}")
            print(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            print(f"Error loading vector store: {e}")

def test_vector_store():
    """Test the vector store independently."""
    # First create some test data without importing document processor
    
    # Create sample processed documents manually
    sample_processed_docs = [
        {
            "content": "Machine Learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "metadata": {"filename": "sample.txt", "chunk_id": 0},
            "embedding": [0.1] * 1536  # Mock embedding for testing
        },
        {
            "content": "Deep Learning uses neural networks with multiple layers for complex pattern recognition.",
            "metadata": {"filename": "sample.txt", "chunk_id": 1},
            "embedding": [0.2] * 1536  # Mock embedding for testing
        }
    ]
    
    # Test with real embeddings if OpenAI is available
    vector_store = SimpleVectorStore()
    
    # Test embedding generation
    test_query = "What is machine learning?"
    query_embedding = vector_store.get_query_embedding(test_query)
    
    if query_embedding:
        print("✅ OpenAI embeddings working!")
        
        # Update sample docs with real embeddings
        texts = [doc["content"] for doc in sample_processed_docs]
        real_embeddings = []
        
        for text in texts:
            embedding = vector_store.get_query_embedding(text)
            if embedding:
                real_embeddings.append(embedding)
        
        if real_embeddings:
            for i, embedding in enumerate(real_embeddings):
                sample_processed_docs[i]["embedding"] = embedding
    
    # Add documents to vector store
    vector_store.add_documents(sample_processed_docs)
    
    # Test search
    query = "What is machine learning?"
    results = vector_store.search(query, top_k=2)
    
    print(f"\nQuery: {query}")
    print("Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['similarity_score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        print()
    
    # Save vector store
    Path("models").mkdir(exist_ok=True)
    vector_store.save_to_file("models/vector_store.json")
    
    print("✅ Vector store test completed successfully!")
    return vector_store

if __name__ == "__main__":
    test_vector_store()
