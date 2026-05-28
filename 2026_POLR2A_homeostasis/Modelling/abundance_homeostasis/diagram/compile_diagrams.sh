#!/bin/bash

# Compile the competition model TikZ diagram to PDF and PNG

echo "Compiling competition model diagram..."

# Compile competition model
pdflatex -interaction=nonstopmode competition_model.tex

# Convert PDF to PNG (using macOS sips utility)
sips -s format png competition_model.pdf --out competition_model.png

# Clean up auxiliary files
rm -f competition_model.aux competition_model.log

echo "Done! Generated competition_model.pdf and competition_model.png"
