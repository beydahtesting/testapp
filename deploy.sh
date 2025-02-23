#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install flask opencv-python-headless numpy matplotlib pillow

export FLASK_APP=app.py
export FLASK_ENV=development

mkdir -p uploads output

echo "Starting Flask server on port 5000..."
flask run --host=0.0.0.0 --port=5000
