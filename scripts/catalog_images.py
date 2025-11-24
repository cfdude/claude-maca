#!/usr/bin/env python3
"""
Image Cataloging Script for CMA Book

Analyzes all images extracted from CMA_Book_033122.docx and creates a catalog
with context, classification, and priority for training data preparation.
"""

import json
from pathlib import Path

from PIL import Image


def find_image_context(md_content, img_name):
    """
    Find the section and surrounding text where image appears in markdown.

    Args:
        md_content: Full markdown file content
        img_name: Image filename to search for

    Returns:
        Dict with section, surrounding text, and line number
    """
    lines = md_content.split("\n")

    for i, line in enumerate(lines):
        if img_name in line:
            # Find preceding header
            section = find_preceding_header(lines, i)

            # Get surrounding text (5 lines before and after)
            start = max(0, i - 5)
            end = min(len(lines), i + 5)
            surrounding = "\n".join(lines[start:end])

            return {"section": section, "surrounding_text": surrounding, "line_number": i}

    # Image not found in markdown
    return {"section": "Unknown", "surrounding_text": "", "line_number": -1}


def find_preceding_header(lines, current_line):
    """
    Walk backwards from current line to find the section header.

    Args:
        lines: List of markdown lines
        current_line: Current line index

    Returns:
        Section header text (without # symbols)
    """
    for i in range(current_line, -1, -1):
        if lines[i].startswith("##"):
            return lines[i].strip("#").strip()
    return "Unknown Section"


def classify_image_type(context):
    """
    Classify image type based on surrounding context.

    Args:
        context: Dict with section and surrounding_text

    Returns:
        String classification: 'chart', 'diagram', 'table', 'candlestick', 'formula', 'other'
    """
    text = (context["section"] + " " + context["surrounding_text"]).lower()

    if any(word in text for word in ["chart", "graph", "trend", "line chart"]):
        return "chart"
    elif any(word in text for word in ["diagram", "flow", "cycle", "process"]):
        return "diagram"
    elif any(word in text for word in ["table", "comparison", "vs ", "versus"]):
        return "table"
    elif any(word in text for word in ["candlestick", "pattern", "technical", "candle"]):
        return "candlestick"
    elif any(word in text for word in ["formula", "equation", "calculation", "math"]):
        return "formula"
    elif any(word in text for word in ["logo", "header", "footer", "title"]):
        return "branding"
    else:
        return "other"


def assess_priority(context, img_type):
    """
    Determine training priority based on content importance.

    Args:
        context: Dict with section and surrounding_text
        img_type: Image classification

    Returns:
        String: 'high', 'medium', 'low', or 'skip'
    """
    section = context["section"].lower()
    text = context["surrounding_text"].lower()

    # Skip branding/decorative
    if img_type == "branding":
        return "skip"

    # High priority: Core concepts, loan selection, client advisory
    high_priority_keywords = [
        "choosing",
        "best loan",
        "advisory",
        "how to",
        "mortgage cycle",
        "bond price",
        "yield",
        "apr",
        "interest rate",
        "refinanc",
        "debt consolidation",
        "15-year",
        "30-year",
        "rate lock",
    ]

    if any(keyword in section for keyword in high_priority_keywords):
        return "high"

    if any(keyword in text for keyword in high_priority_keywords):
        return "high"

    # Medium priority: Market understanding, technical concepts
    medium_priority_keywords = [
        "bond",
        "market",
        "fed",
        "treasury",
        "economic",
        "recession",
        "technical",
        "candlestick",
        "employment",
        "housing",
    ]

    if any(keyword in section for keyword in medium_priority_keywords):
        return "medium"

    # Low priority: Historical data, examples
    if any(keyword in section for keyword in ["example", "historical", "past"]):
        return "low"

    # Default to low
    return "low"


def catalog_images(media_dir, markdown_file, output_file):
    """
    Create comprehensive catalog of all images.

    Args:
        media_dir: Path to media directory with images
        markdown_file: Path to markdown file with image references
        output_file: Path for output JSON catalog

    Returns:
        List of image catalog entries
    """
    media_path = Path(media_dir)

    # Get all image files
    images = []
    for ext in ["*.jpeg", "*.jpg", "*.png"]:
        images.extend(media_path.glob(ext))

    images = sorted(images)

    # Read markdown content
    with open(markdown_file, encoding="utf-8") as f:
        md_content = f.read()

    catalog = []

    print(f"Processing {len(images)} images...")

    for img_file in images:
        # Find context in markdown
        img_name = img_file.name
        context = find_image_context(md_content, img_name)

        # Get image dimensions
        try:
            with Image.open(img_file) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Could not open {img_name}: {e}")
            width, height = 0, 0

        # Classify and prioritize
        img_type = classify_image_type(context)
        priority = assess_priority(context, img_type)

        catalog_entry = {
            "filename": img_name,
            "path": str(img_file),
            "width": width,
            "height": height,
            "type": img_type,
            "priority": priority,
            "section": context["section"],
            "line_number": context["line_number"],
            "context_preview": context["surrounding_text"][:200] + "..."
            if len(context["surrounding_text"]) > 200
            else context["surrounding_text"],
            # To be filled manually or via AI
            "description": "",
            "key_insight": "",
            "training_prompts": [],
            "client_scenarios": [],
        }

        catalog.append(catalog_entry)

    # Save catalog
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    # Print statistics
    print(f"\n✓ Cataloged {len(catalog)} images")
    print("\nBy Type:")
    type_counts = {}
    for entry in catalog:
        type_counts[entry["type"]] = type_counts.get(entry["type"], 0) + 1
    for img_type, count in sorted(type_counts.items()):
        print(f"  {img_type}: {count}")

    print("\nBy Priority:")
    priority_counts = {}
    for entry in catalog:
        priority_counts[entry["priority"]] = priority_counts.get(entry["priority"], 0) + 1
    for priority, count in sorted(priority_counts.items()):
        print(f"  {priority}: {count}")

    print(f"\n✓ Catalog saved to: {output_file}")

    # Save high priority subset
    high_priority = [e for e in catalog if e["priority"] == "high"]
    high_priority_file = output_file.replace(".json", "_high_priority.json")
    with open(high_priority_file, "w", encoding="utf-8") as f:
        json.dump(high_priority, f, indent=2, ensure_ascii=False)

    print(f"✓ High priority images ({len(high_priority)}) saved to: {high_priority_file}")

    return catalog


if __name__ == "__main__":
    # Default paths
    media_dir = "docs/mortgage-domain/media"
    markdown_file = "docs/mortgage-domain/source_material.md"
    output_file = "data/image_catalog.json"

    # Create data directory if needed
    Path("data").mkdir(exist_ok=True)

    # Run cataloging
    catalog = catalog_images(media_dir, markdown_file, output_file)

    print(f"\n{'=' * 60}")
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Review data/image_catalog_high_priority.json")
    print("2. Add descriptions to high-priority images")
    print("3. Use descriptions in training Q&A pairs")
    print(f"{'=' * 60}\n")
