#!/usr/bin/env python3
"""
PDF Content Extraction Script for MACA Research Paper

This script extracts:
- Images and figures
- Text content organized by sections
- Mathematical equations (where possible)
- Tables
- Algorithm blocks
"""

import os
import sys
from pathlib import Path
import json

def check_dependencies():
    """Check if required libraries are installed"""
    required = {
        'PyMuPDF': 'fitz',
        'pdf2image': 'pdf2image',
        'Pillow': 'PIL'
    }

    missing = []
    for name, module in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"Missing required libraries: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def extract_images(pdf_path, output_dir):
    """Extract all images from the PDF"""
    import fitz  # PyMuPDF

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    images_metadata = []

    print(f"Extracting images from {len(doc)} pages...")

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Save image
                image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                image_path = output_dir / image_filename

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                images_metadata.append({
                    "filename": image_filename,
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "format": image_ext,
                    "description": f"Image from page {page_num + 1}"
                })

                print(f"  Extracted: {image_filename}")

            except Exception as e:
                print(f"  Error extracting image on page {page_num + 1}: {e}")

    # Save metadata
    metadata_path = output_dir / "images_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(images_metadata, f, indent=2)

    print(f"\nExtracted {len(images_metadata)} images to {output_dir}")
    print(f"Metadata saved to {metadata_path}")

    doc.close()
    return images_metadata

def extract_text_by_section(pdf_path, output_dir):
    """Extract text content organized by sections"""
    import fitz

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    full_text = []
    sections = {}
    current_section = "Introduction"

    print(f"\nExtracting text from {len(doc)} pages...")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text.append(f"--- Page {page_num + 1} ---\n{text}\n")

    # Save full text
    full_text_path = output_dir / "full_text.txt"
    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))

    print(f"Full text saved to {full_text_path}")

    doc.close()
    return full_text_path

def extract_figures_as_pages(pdf_path, output_dir):
    """Convert PDF pages with figures to images for reference"""
    from pdf2image import convert_from_path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nConverting pages with figures to images...")

    # Figure pages from the PDF (manually identified key pages)
    figure_pages = [1, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 28, 29, 30, 34, 35, 36, 37]

    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=max(figure_pages))

        for page_num in figure_pages:
            if page_num <= len(images):
                img = images[page_num - 1]
                img_path = output_dir / f"figure_page_{page_num}.png"
                img.save(img_path, "PNG")
                print(f"  Saved: figure_page_{page_num}.png")

        print(f"\nFigure pages saved to {output_dir}")

    except Exception as e:
        print(f"Error converting pages: {e}")
        print("Note: pdf2image requires poppler-utils to be installed")
        print("  macOS: brew install poppler")

def create_markdown_template(output_dir):
    """Create markdown template files for organization"""
    output_dir = Path(output_dir)

    templates = {
        "00_overview.md": """# MACA Research Overview

## Title
Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment

## Key Findings
- TODO: Extract key findings

## Core Concepts
- TODO: List core concepts

## Applications to Claude Code
- TODO: Document applications
""",
        "01_self_consistency.md": """# Self-Consistency Formalization

## Mathematical Definition
TODO: Extract mathematical formalization

## Metrics
TODO: Document metrics

## Measurement
TODO: Explain measurement approaches
""",
        "02_maca_framework.md": """# MACA Framework

## Components
TODO: List framework components

## Algorithm
TODO: Extract algorithm details

## Training Objectives
TODO: Document training objectives
""",
        "03_experimental_results.md": """# Experimental Results

## Datasets
TODO: List datasets

## Performance Improvements
TODO: Document improvements

## Analysis
TODO: Add analysis
"""
    }

    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in templates.items():
        filepath = docs_dir / filename
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created template: {filepath}")

def main():
    """Main extraction pipeline"""
    if not check_dependencies():
        print("\nNote: Some features will be limited without required libraries")
        print("You can still proceed with manual extraction using the PDF reader")
        return

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    pdf_path = project_dir / "2509.15172v2.pdf"
    images_dir = project_dir / "images"
    extracted_dir = project_dir / "extracted_data"

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)

    print("="*60)
    print("MACA Research PDF Extraction")
    print("="*60)

    # Extract images
    print("\n[1/4] Extracting embedded images...")
    try:
        extract_images(pdf_path, images_dir)
    except Exception as e:
        print(f"Error extracting images: {e}")

    # Extract text
    print("\n[2/4] Extracting text content...")
    try:
        extract_text_by_section(pdf_path, extracted_dir)
    except Exception as e:
        print(f"Error extracting text: {e}")

    # Extract figure pages
    print("\n[3/4] Converting figure pages to images...")
    try:
        extract_figures_as_pages(pdf_path, images_dir / "figures")
    except Exception as e:
        print(f"Error converting figures: {e}")

    # Create templates
    print("\n[4/4] Creating markdown templates...")
    create_markdown_template(project_dir)

    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review extracted images in: {images_dir}")
    print(f"2. Review extracted text in: {extracted_dir}")
    print(f"3. Fill in markdown templates in: {project_dir}/docs")
    print(f"4. Add image descriptions to images_metadata.json")

if __name__ == "__main__":
    main()
