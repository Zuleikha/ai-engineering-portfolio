"""Document processing and chunking for RAG system."""

import os
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import openai
from dotenv import load_dotenv

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
            
            # Add chunk information to metadata
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
        # Load documents
        documents = self.load_text_files(directory)
        if not documents:
            print("No documents found!")
            return []
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Get embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.get_embeddings(texts)
        
        # Combine everything
        processed_docs = []
        for chunk, embedding in zip(chunks, embeddings):
            processed_docs.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embedding
            })
        
        return processed_docs

# Test function
def test_document_processor():
    """Test the document processor."""
    processor = DocumentProcessor()
    
    # Create sample document for testing
    test_dir = Path("data/raw")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    sample_doc = test_dir / "sample.txt"
    with open(sample_doc, 'w') as f:
        f.write("""
        Machine Learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. It includes supervised learning, unsupervised learning, 
        and reinforcement learning approaches.
        
        Deep Learning is a specialized area of machine learning that uses neural networks
        with multiple layers. It has been particularly successful in computer vision,
        natural language processing, and speech recognition tasks.
        
        Natural Language Processing (NLP) is a field that combines computer science and 
        linguistics to help computers understand human language. It includes tasks like
        text classification, named entity recognition, and language translation.
        """)
    
    # Test processing
    results = processor.process_documents("data/raw")
    print(f"Processed {len(results)} chunks successfully!")
    
    return results

if __name__ == "__main__":
    test_document_processor()
