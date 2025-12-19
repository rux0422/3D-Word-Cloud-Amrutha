#!/bin/bash
# ============================================
# 3D Word Cloud - Setup Script for Unix/Mac
# Installs all dependencies and starts servers
# ============================================

echo ""
echo "======================================"
echo "  3D Word Cloud - Setup Script"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo "Please install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed."
    echo "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

echo "[INFO] Python and Node.js detected."
echo ""

# Install Backend Dependencies
echo "======================================"
echo "  Installing Backend Dependencies"
echo "======================================"
echo ""

cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing packages..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('maxent_ne_chunker', quiet=True); nltk.download('maxent_ne_chunker_tab', quiet=True); nltk.download('words', quiet=True)"

cd ..

echo ""
echo "======================================"
echo "  Installing Frontend Dependencies"
echo "======================================"
echo ""

cd frontend
npm install
cd ..

echo ""
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "To start the application, run: ./start.sh"
echo ""
echo "Or manually:"
echo "  Backend:  cd backend && source venv/bin/activate && python main.py"
echo "  Frontend: cd frontend && npm run dev"
echo ""
