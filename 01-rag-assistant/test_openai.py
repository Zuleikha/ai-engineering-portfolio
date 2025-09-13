"""Test OpenAI API connection."""

import os
from dotenv import load_dotenv
import openai

load_dotenv()

def test_openai_connection():
    """Test if OpenAI API key is working."""
    try:
        # Set API key
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'OpenAI connection successful!'"}
            ],
            max_tokens=10
        )
        
        print("✅ OpenAI API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai_connection()
