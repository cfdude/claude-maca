#!/usr/bin/env python3
"""
Convert markdown questions to JSON format for MACA debates.
"""

import json
import re
from pathlib import Path


def parse_markdown_questions(md_file: str) -> list:
    """Parse questions from markdown format."""
    questions = []

    with open(md_file, "r") as f:
        content = f.read()

    # Match pattern: "123. Question text here?"
    pattern = r"^\d+\.\s+(.+?)$"

    for line in content.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            question_text = match.group(1).strip()

            # Determine category from context (simple heuristic)
            category = "loan_origination"
            difficulty = "advanced"  # All these questions are strategic/advanced

            question_num = len(questions) + 1
            questions.append(
                {
                    "id": f"loan_orig_{question_num:03d}",
                    "prompt": question_text,
                    "metadata": {
                        "category": category,
                        "difficulty": difficulty,
                        "source": "loan_origination_questions",
                        "debate_worthy": True,
                        "client_facing": True,
                        "requires_consultation": True,
                    },
                }
            )

    return questions


def main():
    """Convert questions and create JSON files."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Input markdown file
    md_file = project_root / "proprietary/data/loan_origination_questions.md"

    # Output files
    full_output = project_root / "proprietary/data/loan_origination_questions_full.json"
    validation_output = project_root / "proprietary/data/loan_origination_questions_validation.json"

    print("Converting markdown questions to JSON...")

    # Parse all questions
    questions = parse_markdown_questions(md_file)

    print(f"✓ Parsed {len(questions)} questions")

    # Save full set (as array, not nested object)
    full_output.parent.mkdir(parents=True, exist_ok=True)
    with open(full_output, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"✓ Saved full set: {full_output}")
    print(f"  Total questions: {len(questions)}")

    # Create validation set (every 10th question)
    validation_questions = [questions[i] for i in range(0, len(questions), 10)]

    # Save validation set (as array, not nested object)
    with open(validation_output, "w") as f:
        json.dump(validation_questions, f, indent=2)

    print(f"✓ Saved validation set: {validation_output}")
    print(f"  Sample questions: {len(validation_questions)}")
    print(f"  Sampling: Every 10th question (1, 11, 21, 31...)")

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}\n")

    print("Next steps:")
    print(f"  1. Run validation: python scripts/run_batch_debates_scheduled.py \\")
    print(f"       --questions {validation_output} \\")
    print(f"       --output proprietary/data/validation_debate_results.json")
    print(
        f"\n  2. After validation, run full batch: python scripts/run_batch_debates_scheduled.py \\"
    )
    print(f"       --questions {full_output} \\")
    print(f"       --output proprietary/data/full_debate_results.json")
    print()


if __name__ == "__main__":
    main()
