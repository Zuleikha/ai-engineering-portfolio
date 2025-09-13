"""Test script to verify the setup is working correctly."""

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import openai
        print("OpenAI package imported successfully")
        
        import langchain
        print("LangChain package imported successfully")
        
        import qdrant_client
        print("Qdrant client imported successfully")
        
        import streamlit
        print("Streamlit imported successfully")
        
        import fastapi
        print("FastAPI imported successfully")
        
        import pandas
        print("Pandas imported successfully")
        
        print("\nAll packages imported successfully!")
        print("Your environment is ready for development!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please check your installation.")

if __name__ == "__main__":
    test_imports()
