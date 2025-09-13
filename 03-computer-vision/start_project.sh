#!/bin/bash

echo "Starting Object Detection Project..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv venv"
    read -p "Press any key to exit..."
    exit 1
fi

echo "Activating virtual environment..."
source venv/Scripts/activate

echo "Starting FastAPI backend..."
# Start API server in background
cd src/api
python main.py &
API_PID=$!
cd ../..

echo "Waiting 3 seconds for server to start..."
sleep 3

echo "Opening frontend in browser..."
start src/frontend/app.html

echo
echo "Project started successfully!"
echo "- API Server: http://localhost:8000 (PID: $API_PID)"
echo "- Frontend: opened in default browser"
echo
echo "Press Ctrl+C to stop the API server"

# Keep script running so you can stop the background process
wait $API_PID