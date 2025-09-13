"""Test RAG components without import issues."""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append('src')

def test_basic_setup():
    """Test basic setup and imports."""
    print("Testing basic imports...")
    
    try:
        import openai
        print("✅ OpenAI imported")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            print("✅ OpenAI API key configured")
            
            # Test simple API call
            client = openai.OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=["test text"]
            )
            print("✅ OpenAI API working")
            print(f"✅ Embedding dimensions: {len(response.data[0].embedding)}")
            
        else:
            print("❌ OpenAI API key not configured")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_document_creation():
    """Test document creation and processing."""
    print("\nTesting document creation...")
    
    # Create sample document
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = data_dir / "test_doc.txt"
    with open(sample_file, 'w') as f:
        f.write("""
        Artificial Intelligence (AI) is a broad field of computer science focused on creating 
        systems that can perform tasks that typically require human intelligence. This includes 
        learning, reasoning, problem-solving, perception, and language understanding.
        
        Machine Learning is a subset of AI that focuses on algorithms that can learn and improve 
        from data without being explicitly programmed. It includes supervised learning, 
        unsupervised learning, and reinforcement learning.
        
        Deep Learning is a specialized subset of machine learning that uses neural networks 
        with multiple layers to model and understand complex patterns in data.
        """)
    
    print(f"✅ Created test document: {sample_file}")
    print(f"✅ Document size: {sample_file.stat().st_size} bytes")

def test_langchain_imports():
    """Test LangChain imports."""
    print("\nTesting LangChain imports...")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        print("✅ LangChain imported successfully")
        
        # Test text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        test_text = "This is a test document. " * 100
        chunks = splitter.split_text(test_text)
        print(f"✅ Text splitting working: {len(chunks)} chunks created")
        
    except Exception as e:
        print(f"❌ LangChain error: {e}")
        print("Installing missing dependencies...")
        os.system("pip install langchain langchain-community")

if __name__ == "__main__":
    test_basic_setup()
    test_document_creation()
    test_langchain_imports()
    print("\n🎉 Component testing completed!")
