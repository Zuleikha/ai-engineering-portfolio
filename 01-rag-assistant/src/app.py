"""Streamlit frontend for RAG Assistant - Modern Professional Interface."""

import streamlit as st
import requests
import json
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern professional CSS styling with better readability
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f0f2f5 0%, #e8eaf6 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.08);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4c51bf 0%, #5a67d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: none;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #2d3748;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #4c51bf 0%, #5a67d8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 3px 10px rgba(76, 81, 191, 0.25);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
        width: 100%;
        min-height: 3rem;
        text-transform: none;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 81, 191, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Secondary Button */
    .secondary-button > button {
        background: linear-gradient(135deg, #38a169 0%, #48bb78 100%);
        box-shadow: 0 3px 10px rgba(56, 161, 105, 0.25);
    }
    
    .secondary-button > button:hover {
        box-shadow: 0 6px 20px rgba(56, 161, 105, 0.35);
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        color: #2d3748;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #5a67d8;
        box-shadow: 0 0 0 3px rgba(90, 103, 216, 0.1);
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background: white;
        border-radius: 15px;
        margin: 1rem 0;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border-left: 4px solid #5a67d8;
    }
    
    /* Status Indicators */
    .status-success {
        background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%);
        color: #1a202c;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 3px 10px rgba(72, 187, 120, 0.15);
        margin: 1rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
        color: #1a202c;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 3px 10px rgba(245, 101, 101, 0.15);
        margin: 1rem 0;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #5a67d8 0%, #4c51bf 100%);
        border-radius: 10px;
    }
    
    /* Info Box */
    .stAlert {
        background: linear-gradient(135deg, #bee3f8 0%, #90cdf4 100%);
        border: none;
        border-radius: 12px;
        color: #1a202c;
        padding: 1.5rem;
        box-shadow: 0 3px 10px rgba(59, 130, 246, 0.12);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 10px;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* JSON Display */
    .stJson {
        background: #f7fafc;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        color: #2d3748;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #5a67d8;
    }
    
    /* Chat Input */
    .stChatInput > div > div > div > div {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #5a67d8;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 2px solid #e2e8f0;
    }
    
    /* Sidebar text colors */
    .css-1d391kg p {
        color: #2d3748;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #2d3748;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents using advanced AI</p>', unsafe_allow_html=True)

# Sidebar for configuration and status
with st.sidebar:
    st.markdown('<h2 class="section-header">Configuration</h2>', unsafe_allow_html=True)
    
    # API URL configuration
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    # API connection test
    col1, col2 = st.columns([1, 1])
    with col1:
        test_button = st.button("Test Connection", key="test_api")
    
    if test_button:
        try:
            with st.spinner("Testing connection..."):
                response = requests.get(f"{api_url}/health", timeout=5)
                
            if response.status_code == 200:
                data = response.json()
                st.markdown('<div class="status-success">API connection successful!</div>', 
                          unsafe_allow_html=True)
                
                # Display API status details
                st.json({
                    "Status": data.get("status", "unknown"),
                    "Service": data.get("service", "unknown"),
                    "Version": data.get("version", "unknown"),
                    "OpenAI Configured": data.get("openai_configured", False)
                })
            else:
                st.markdown('<div class="status-error">API connection failed</div>', 
                          unsafe_allow_html=True)
                st.error(f"Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.markdown('<div class="status-error">Connection error</div>', 
                      unsafe_allow_html=True)
            st.error(f"Error: {str(e)}")
    
    # App information
    st.markdown('<h2 class="section-header">App Status</h2>', unsafe_allow_html=True)
    st.info("**Current Phase:** Basic Setup Complete\n**Next:** RAG Implementation")
    
    # Clear chat button
    with col2:
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "Hello! I'm your RAG Assistant. I'm currently in setup mode with mock responses. Ask me any question to test the interface!"
                }
            ]
            st.rerun()

# Main chat interface
st.markdown('<h2 class="section-header">Chat Interface</h2>', unsafe_allow_html=True)

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
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"{i}. {source}")
            
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
                    with st.expander("Sources"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. {source}")
                
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
st.markdown('<div class="footer">RAG Assistant v1.0 - Built with FastAPI + Streamlit</div>', unsafe_allow_html=True)