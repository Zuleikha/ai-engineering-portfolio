"""Enhanced Streamlit app with PDF upload functionality."""

import streamlit as st
import requests
import json
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Enhanced RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Enhanced RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("**AI-powered document chat with PDF upload support**")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API URL
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    # Connection test
    if st.button("Test API Connection"):
        try:
            with st.spinner("Testing connection..."):
                response = requests.get(f"{api_url}/health", timeout=5)
                
            if response.status_code == 200:
                data = response.json()
                st.success("API connection successful!")
                st.json({
                    "Status": data.get("status", "unknown"),
                    "Version": data.get("version", "unknown"),
                    "Features": data.get("features", []),
                    "Formats": data.get("supported_formats", [])
                })
            else:
                st.error(f"API connection failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
    
    st.header("System Info")
    st.info("**Current Version:** Enhanced with PDF Upload")

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Document Upload")
    
    # File upload section
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF file to add to your knowledge base"
        )
        
        if uploaded_file is not None:
            # Show file details
            st.write("**File Details:**")
            st.write(f"Name: {uploaded_file.name}")
            st.write(f"Size: {uploaded_file.size} bytes")
            
            # Upload button
            if st.button("Process PDF", type="primary"):
                try:
                    with st.spinner("Processing PDF..."):
                        # Prepare file for upload
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        
                        # Send to API
                        response = requests.post(
                            f"{api_url}/upload-pdf",
                            files=files,
                            timeout=60
                        )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("PDF processed successfully!")
                        st.json({
                            "Filename": data.get("filename"),
                            "Chunks Created": data.get("chunks"),
                            "Characters Processed": data.get("characters")
                        })
                    else:
                        st.error(f"Upload failed: {response.status_code}")
                        st.error(response.text)
                        
                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.header("Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! I'm your enhanced RAG assistant. Upload a PDF document and ask me questions about it!"
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
                        for source in message["sources"]:
                            st.write(f"• {source}")
                
                if "confidence" in message:
                    st.progress(message["confidence"])
                    st.caption(f"Confidence: {message['confidence']:.0%}")
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
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
                            for source in sources:
                                st.write(f"• {source}")
                    
                    # Display confidence
                    if confidence > 0:
                        st.progress(confidence)
                        st.caption(f"Confidence: {confidence:.0%}")
                    
                    # Add to chat history
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
st.markdown("**Enhanced RAG Assistant v2.0** - Now with PDF Upload Support!")
