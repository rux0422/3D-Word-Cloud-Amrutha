#!/bin/bash
# ============================================
# 3D Word Cloud - Start Script for Unix/Mac
# Starts both frontend and backend servers
# ============================================

echo ""
echo "======================================"
echo "  3D Word Cloud - Starting Servers"
echo "======================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Backend
echo "Starting Backend Server on port 8001..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start Frontend
echo "Starting Frontend Server on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "======================================"
echo "  Servers Started!"
echo "======================================"
echo ""
echo "  Backend API: http://localhost:8001"
echo "  API Docs:    http://localhost:8001/docs"
echo "  Frontend:    http://localhost:3000"
echo ""
echo "  Press Ctrl+C to stop all servers."
echo ""

# Open browser (works on Mac and some Linux)
sleep 3
if command -v open &> /dev/null; then
    open http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000
fi

# Wait for processes
wait
