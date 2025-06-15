#!/bin/bash

# SmartRAG startup script
set -e

echo "🚀 Starting SmartRAG..."

# Load environment variables
if [ -f .env ]; then
    echo "📄 Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run database migrations if needed
if [ "$DATABASE_URL" ]; then
    echo "🗄️ Running database migrations..."
    alembic upgrade head
fi

# Start the application
echo "🌟 Starting SmartRAG API server..."
python main.py