#!/bin/bash
# Setup script for MACA project Python environment

echo "Setting up MACA Python virtual environment..."

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install poppler for PDF processing (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Checking for poppler..."
    if ! command -v pdfinfo &> /dev/null; then
        echo "Installing poppler via Homebrew..."
        brew install poppler
    else
        echo "✓ poppler already installed"
    fi
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the virtual environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the extraction script:"
echo "  python scripts/extract_pdf.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
