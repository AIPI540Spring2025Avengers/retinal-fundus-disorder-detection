#!/bin/bash
set -e
set -o pipefail

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check and load dataset if needed
if [ ! -d "data/raw/train" ] || [ -z "$(ls -A data/raw/train 2>/dev/null)" ]; then
    echo "Downloading dataset..."
    bash scripts/make_dataset.sh
else
    echo "Dataset already exists in data/raw/, skipping download."
fi

# Run preprocessing
echo "Running preprocessing..."
python scripts/traditional/preprocessingML.py

# Train and serialize model
echo "Training model..."
python scripts/traditional/modelML.py

echo "✅ Setup completed successfully!"

## Above code generated using the DeepSeek R1 model through Perplexity, and then tweaked. The prompt was:
##
## Can you generate a setup.sh bash script that runs
## 1. creates a Python virtual environment using `python -m venv .venv` (if a virtual environment at .venv doesn't already exist)
## 2. activates the virtual environment (if not already activated) using `source .venv/bin/activate`
## 3. installs requirements from requirements.txt using pip install -r requirements.txt
## 4. runs scripts/make_dataset.sh, a bash script that loads a dataset into the data/raw/ directory. We should only run this script if the data/raw directory is empty.
## 5. runs traditional/preprocessingML.py, which generates features for a traditional ML model
## 6. runs traditional/modelML.py, which generates serialized model and encoder
