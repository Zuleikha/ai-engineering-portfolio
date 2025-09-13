"""PDF document processor for RAG system."""

import os
import PyPDF2
import io
from typing import Optional, List, Dict, Any
from pathlib import Path
import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

class PDFProcessor:
    """Process PDF files for RAG system."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> str:
        """Extract text from PDF bytes (for uploaded files)."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                except Exception as e:
                    print(f"Error extracting page {page_num + 1} from {filename}: {e}")
                    continue
            
            return text.strip()
            
        except Exception as e:
            print(f"Error reading PDF bytes for {filename}: {e}")
            return ""
    
    def process_uploaded_pdf(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Process an uploaded PDF file and return processed chunks."""
        try:
            # Extract text from PDF
            content = self.extract_text_from_pdf_bytes(file_bytes, filename)
            
            if not content.strip():
                print(f"No text extracted from {filename}")
                return None
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "source": f"uploaded/{filename}",
                    "filename": filename,
                    "file_type": ".pdf",
                    "file_size": len(content)
                }
            )
            
            # Chunk the document
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
            
            # Get embeddings
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.get_embeddings(texts)
            
            if not embeddings or len(embeddings) != len(chunks):
                print(f"Failed to get embeddings for {filename}")
                return None
            
            # Return processed document data
            processed_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                processed_chunks.append({
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "embedding": embedding
                })
            
            return {
                "filename": filename,
                "file_type": ".pdf",
                "chunks": processed_chunks,
                "total_chunks": len(processed_chunks),
                "total_chars": len(content)
            }
            
        except Exception as e:
            print(f"Error processing uploaded PDF {filename}: {e}")
            return None
    
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

# Test function
def test_pdf_processor():
    """Test the PDF processor."""
    print("PDF processor created successfully")
    processor = PDFProcessor()
    print("Ready to process PDF uploads")

if __name__ == "__main__":
    test_pdf_processor()
