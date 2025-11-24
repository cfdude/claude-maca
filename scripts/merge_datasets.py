#!/usr/bin/env python3
"""Merge statistical context examples into main training dataset"""

import json
from pathlib import Path

# Load existing dataset
with open("data/training_dataset_with_visuals.json", "r") as f:
    main_dataset = json.load(f)

# Load statistical examples
with open("data/statistical_context_examples.json", "r") as f:
    statistical_examples = json.load(f)

# Merge datasets
merged_dataset = main_dataset + statistical_examples

# Save merged dataset
with open("data/training_dataset_complete.json", "w") as f:
    json.dump(merged_dataset, f, indent=2)

print(f"Merged dataset created:")
print(f"  Core examples: {len(main_dataset)}")
print(f"  Statistical examples: {len(statistical_examples)}")
print(f"  Total examples: {len(merged_dataset)}")
print(f"  Saved to: data/training_dataset_complete.json")

# Create summary by category
categories = {}
for example in merged_dataset:
    cat = example["metadata"]["category"]
    categories[cat] = categories.get(cat, 0) + 1

print(f"\nBreakdown by category:")
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")
