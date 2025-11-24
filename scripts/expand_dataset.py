#!/usr/bin/env python3
"""
Dataset Expansion Script for CMA Training Data

This script helps systematically extract Q&A pairs from the CMA book source material.
It provides a framework for creating training examples across all 5 modules.
"""

import json
from pathlib import Path

# Module content areas to extract from
MODULE_SECTIONS = {
    "A": {
        "name": "Be an Advisor to Combat Fintech",
        "sections": [
            {"name": "How the Mortgage Market Works", "lines": "148-300"},
            {"name": "Mortgage Cycle Deep Dive", "lines": "300-450"},
            {"name": "Refinances and Breakeven", "lines": "450-550"},
            {"name": "Debt Consolidation", "lines": "550-650"},
            {"name": "Annual Percentage Rate (APR)", "lines": "650-800"},
            {"name": "Choosing the Best Loan", "lines": "800-900"},
        ],
    },
    "B": {
        "name": "Understanding the Markets",
        "sections": [
            {"name": "Bonds and Mortgage Bonds", "lines": "900-1100"},
            {"name": "Yield to Maturity", "lines": "1100-1250"},
            {"name": "What Drives Rates", "lines": "1250-1400"},
            {"name": "Fixed Mortgage Rates vs Treasury Yields", "lines": "1400-1500"},
            {"name": "Stocks and Bonds Relationship", "lines": "1500-1600"},
        ],
    },
    "C": {
        "name": "Understanding Economic Reports",
        "sections": [
            {"name": "Economic Reports Overview", "lines": "1600-1650"},
            {"name": "Employment Reports", "lines": "1650-1750"},
            {"name": "Housing Reports", "lines": "1750-1850"},
            {"name": "Other Economic Reports", "lines": "1850-2000"},
        ],
    },
    "D": {
        "name": "The Federal Reserve and Recession Indicators",
        "sections": [
            {"name": "Central Banking System", "lines": "2000-2150"},
            {"name": "Fed Members and Communication", "lines": "2150-2300"},
            {"name": "Money Printing and Inflation", "lines": "2300-2450"},
            {"name": "Recession Indicators", "lines": "2450-2600"},
            {"name": "Choosing the Best Loan Strategy", "lines": "2600-2700"},
        ],
    },
    "E": {
        "name": "Technical Analysis",
        "sections": [
            {"name": "Western Technical Signals", "lines": "2700-2900"},
            {"name": "Support and Resistance", "lines": "2900-3000"},
            {"name": "Eastern Technical Signals", "lines": "3000-3150"},
            {"name": "Candlestick Patterns", "lines": "3150-3300"},
            {"name": "Rate Lock Decisions", "lines": "3300-3554"},
        ],
    },
}

# Example categories and difficulty distribution
CATEGORIES = [
    "mortgage_basics",
    "client_advisory",
    "debt_management",
    "interest_rates_markets",
    "rate_lock_strategy",
    "economic_analysis",
    "fed_policy",
    "technical_analysis",
    "loan_comparison",
    "refinance_strategy",
]

DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]

# Target distribution
TARGET_TOTALS = {
    "beginner": 30,  # 30 examples
    "intermediate": 40,  # 40 examples
    "advanced": 50,  # 50 examples
    "expert": 30,  # 30 examples
}
# Total: 150 examples (including 7 seed examples)


def create_example_template(example_id, module, section, category, difficulty):
    """Create a template for a new training example"""
    return {
        "id": example_id,
        "prompt": "[QUESTION TO BE FILLED]",
        "chosen": "[DETAILED ADVISORY RESPONSE - Strategic, comprehensive, CMA voice]",
        "rejected": "[SURFACE-LEVEL RESPONSE - Generic, brief, non-advisory]",
        "metadata": {
            "category": category,
            "difficulty": difficulty,
            "cma_module": module,
            "source_section": section["name"],
            "line_reference": section["lines"],
            "debate_worthy": False,  # Set True if suitable for multi-agent debate
            "client_facing": False,  # Set True if direct client communication
            "requires_calculation": False,
            "requires_market_analysis": False,
            "visual_concept": False,
            "image_references": [],
        },
    }


def generate_extraction_plan():
    """Generate a plan for extracting examples from each module"""

    plan = {
        "total_needed": 143,  # To reach 150 total (including 7 seed)
        "by_module": {},
        "by_difficulty": TARGET_TOTALS,
    }

    # Distribute across modules
    module_distribution = {
        "A": 40,  # Most practical advisor content
        "B": 30,  # Markets and rates
        "C": 25,  # Economic reports
        "D": 25,  # Fed and recession indicators
        "E": 23,  # Technical analysis
    }

    for module, count in module_distribution.items():
        sections = MODULE_SECTIONS[module]["sections"]
        examples_per_section = count // len(sections)

        plan["by_module"][module] = {
            "total": count,
            "per_section": examples_per_section,
            "sections": sections,
        }

    return plan


def print_extraction_plan():
    """Print the extraction plan for reference"""
    plan = generate_extraction_plan()

    print("=== CMA Dataset Expansion Plan ===\n")
    print(f"Current: 7 seed examples")
    print(f"Target: 150 total examples")
    print(f"To Create: {plan['total_needed']} new examples\n")

    print("Distribution by Difficulty:")
    for diff, count in plan["by_difficulty"].items():
        print(f"  {diff.capitalize()}: {count} examples")

    print("\nDistribution by Module:")
    for module, details in plan["by_module"].items():
        print(f"\n  Module {module}: {MODULE_SECTIONS[module]['name']}")
        print(f"    Total: {details['total']} examples")
        print(f"    Per Section: ~{details['per_section']} examples")
        print(f"    Sections:")
        for section in details["sections"]:
            print(f"      - {section['name']} (lines {section['lines']})")


def create_batch_examples(module, section_index, num_examples, start_id=8):
    """
    Create a batch of example templates for a specific module section

    Args:
        module: Module letter (A, B, C, D, E)
        section_index: Index of section within module (0-based)
        num_examples: Number of examples to create
        start_id: Starting ID number

    Returns:
        List of example templates
    """
    section = MODULE_SECTIONS[module]["sections"][section_index]
    examples = []

    for i in range(num_examples):
        example_id = f"cma_{start_id + i:03d}"

        # Vary difficulty across batch
        difficulty = DIFFICULTY_LEVELS[i % 4]

        # Default category based on module
        category_map = {
            "A": "mortgage_basics",
            "B": "interest_rates_markets",
            "C": "economic_analysis",
            "D": "fed_policy",
            "E": "technical_analysis",
        }
        category = category_map[module]

        example = create_example_template(example_id, module, section, category, difficulty)
        examples.append(example)

    return examples


if __name__ == "__main__":
    print_extraction_plan()

    print("\n" + "=" * 50)
    print("Example Template for Module A, Section 0:")
    print("=" * 50)

    # Create one example template as demonstration
    examples = create_batch_examples("A", 0, 1, start_id=8)
    print(json.dumps(examples[0], indent=2))
