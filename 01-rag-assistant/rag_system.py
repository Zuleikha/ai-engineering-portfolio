"""Complete RAG system - all components in one file to avoid import issues."""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

class DocumentProcessor:
    """Process and chunk documents for RAG."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def load_text_files(self, directory: str) -> List[Document]:
        """Load text files from directory."""
        documents = []
        data_dir = Path(directory)
        
        if not data_dir.exists():
            print(f"Directory {directory} does not exist")
            return documents
        
        for file_path in data_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": "text"
                    }
                )
                documents.append(doc)
                print(f"Loaded: {file_path.name}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
                chunked_docs.append(chunk)
        
        print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for text chunks."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []
    
    def process_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Complete document processing pipeline."""
        documents = self.load_text_files(directory)
        if not documents:
            print("No documents found!")
            return []
        
        chunks = self.chunk_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.get_embeddings(texts)
        
        processed_docs = []
        for chunk, embedding in zip(chunks, embeddings):
            processed_docs.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embedding
            })
        
        return processed_docs

class VectorStore:
    """Simple vector store for RAG."""
    
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
        
        query_embedding = self.get_query_embedding(query)
        if not query_embedding:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
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
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
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

class RAGSystem:
    """Complete RAG system for question answering."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        self.conversation_memory: Dict[str, List[Dict[str, str]]] = {}
        
        self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create new one."""
        try:
            self.vector_store.load_from_file("models/vector_store.json")
            print("✅ Loaded existing vector store")
        except:
            print("📁 No existing vector store found, creating new one...")
            self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize vector store with documents."""
        try:
            processed_docs = self.document_processor.process_documents("data/raw")
            if processed_docs:
                self.vector_store.add_documents(processed_docs)
                self.vector_store.save_to_file("models/vector_store.json")
                print("✅ Created and saved new vector store")
            else:
                print("⚠️ No documents found to process")
        except Exception as e:
            print(f"⚠️ Could not initialize vector store: {e}")
    
    def add_documents(self, directory_path: str):
        """Add new documents to the RAG system."""
        processed_docs = self.document_processor.process_documents(directory_path)
        if processed_docs:
            self.vector_store.add_documents(processed_docs)
            self.vector_store.save_to_file("models/vector_store.json")
            return len(processed_docs)
        return 0
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a user."""
        return self.conversation_memory.get(conversation_id, [])
    
    def add_to_conversation(self, conversation_id: str, user_message: str, assistant_response: str):
        """Add exchange to conversation memory."""
        if conversation_id not in self.conversation_memory:
            self.conversation_memory[conversation_id] = []
        
        self.conversation_memory[conversation_id].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # Keep only last 10 exchanges (20 messages)
        if len(self.conversation_memory[conversation_id]) > 20:
            self.conversation_memory[conversation_id] = self.conversation_memory[conversation_id][-20:]
    
    def create_context_prompt(self, query: str, search_results: List[Dict[str, Any]], 
                            conversation_history: List[Dict[str, str]]) -> str:
        """Create a prompt with context and conversation history."""
        
        context_parts = []
        for i, result in enumerate(search_results):
            source = result["metadata"].get("filename", "unknown")
            context_parts.append(f"[Source {i+1}: {source}]\n{result['content']}")
        
        context = "\n\n".join(context_parts)
        
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-6:]
            conversation_context = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in recent_history
            ])
        
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context documents. 

CONTEXT DOCUMENTS:
{context}

CONVERSATION HISTORY:
{conversation_context}

CURRENT QUESTION: {query}

INSTRUCTIONS:
1. Answer the question using ONLY information from the context documents
2. If the context doesn't contain enough information, say so clearly
3. Cite the specific sources you use in your answer
4. Be conversational and refer to previous messages if relevant
5. Keep your answer concise but comprehensive

ANSWER:"""
        
        return prompt
    
    def query(self, question: str, conversation_id: str = "default") -> Dict[str, Any]:
        """Process a question and return RAG response."""
        try:
            search_results = self.vector_store.search(question, top_k=3)
            
            if not search_results:
                return {
                    "answer": "I don't have any relevant documents to answer your question. Please add some documents first.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            conversation_history = self.get_conversation_history(conversation_id)
            prompt = self.create_context_prompt(question, search_results, conversation_history)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            sources = [
                result["metadata"].get("filename", "unknown document")
                for result in search_results
            ]
            
            avg_similarity = sum(r["similarity_score"] for r in search_results) / len(search_results)
            confidence = min(avg_similarity * 1.2, 1.0)
            
            self.add_to_conversation(conversation_id, question, answer)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error in RAG query: {e}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }

def test_rag_system():
    """Test the complete RAG system."""
    print("🚀 Testing RAG System...")
    
    rag = RAGSystem()
    
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?", 
        "What's the difference between machine learning and deep learning?"
    ]
    
    for query in test_queries:
        print(f"\n🤔 Question: {query}")
        result = rag.query(query, "test_user")
        print(f"🤖 Answer: {result['answer']}")
        print(f"📚 Sources: {', '.join(result['sources'])}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print("-" * 50)
    
    return rag

if __name__ == "__main__":
    test_rag_system()
