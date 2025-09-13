"""Streamlit frontend for RAG Assistant."""

import streamlit as st
import requests
import json
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("**Ask questions about your documents using advanced AI**")

# Sidebar for configuration and status
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL configuration
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    # API connection test
    if st.button("🔍 Test API Connection"):
        try:
            with st.spinner("Testing connection..."):
                response = requests.get(f"{api_url}/health", timeout=5)
                
            if response.status_code == 200:
                data = response.json()
                st.markdown('<p class="status-success">✅ API connection successful!</p>', 
                          unsafe_allow_html=True)
                
                # Display API status details
                st.json({
                    "Status": data.get("status", "unknown"),
                    "Service": data.get("service", "unknown"),
                    "Version": data.get("version", "unknown"),
                    "OpenAI Configured": data.get("openai_configured", False)
                })
            else:
                st.markdown('<p class="status-error">❌ API connection failed</p>', 
                          unsafe_allow_html=True)
                st.error(f"Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.markdown('<p class="status-error">❌ Connection error</p>', 
                      unsafe_allow_html=True)
            st.error(f"Error: {str(e)}")
    
    # App information
    st.header("📊 App Status")
    st.info("**Current Phase:** Basic Setup Complete\n**Next:** RAG Implementation")

# Main chat interface
st.header("💬 Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello! I'm your RAG Assistant. I'm currently in setup mode with mock responses. Ask me any question to test the interface!"
        }
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
            
            # Display additional info if available
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"• {source}")
            
            if "confidence" in message:
                st.progress(message["confidence"])
                st.caption(f"Confidence: {message['confidence']:.0%}")
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from API
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{api_url}/query",
                    json={"question": prompt, "conversation_id": "streamlit_user"},
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                confidence = data.get("confidence", 0.0)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"• {source}")
                
                # Display confidence
                if confidence > 0:
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.0%}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence
                })
                
            else:
                error_msg = f"API Error: {response.status_code}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Sorry, I encountered an error: {error_msg}"
                })
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Sorry, I couldn't connect to the API: {error_msg}"
            })

# Footer
st.markdown("---")
st.markdown("**RAG Assistant v1.0** - Built with FastAPI + Streamlit")
