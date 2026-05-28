#!/bin/bash

# Compile the TikZ diagram to PDF and PNG

echo "Compiling state diagram..."

# Compile LaTeX to PDF
pdflatex -interaction=nonstopmode state_diagram.tex

# Convert PDF to PNG (using macOS sips utility)
sips -s format png state_diagram.pdf --out state_diagram.png

# Clean up auxiliary files
rm -f state_diagram.aux state_diagram.log

echo "Done! Generated state_diagram.pdf and state_diagram.png"
