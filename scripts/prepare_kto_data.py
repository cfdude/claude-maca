#!/usr/bin/env python3
"""
Prepare KTO (Kahneman-Tversky Optimization) training data from debate results.

KTO is an alternative to DPO that uses individual ratings (desirable/undesirable)
rather than pairwise comparisons. This can be useful when you have single responses
that can be clearly labeled as good or bad based on consensus.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def prepare_kto_data(
    debate_results_path: str,
    output_path: str,
    min_consensus: float = 0.6,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Convert debate results to KTO format.

    KTO Format:
    {
        "prompt": str,           # The question/prompt
        "completion": str,       # Agent response reasoning
        "label": bool,           # True = desirable, False = undesirable
        "metadata": dict         # Optional metadata
    }

    Args:
        debate_results_path: Path to batch debate results JSON
        output_path: Path to save KTO data (JSONL format)
        min_consensus: Minimum consensus strength to include (default 0.6)
        include_metadata: Whether to include debate metadata

    Returns:
        Dictionary with statistics about conversion
    """
    # Load debate results
    with open(debate_results_path) as f:
        data = json.load(f)

    results = data["results"]

    kto_data = []
    stats = {
        "total_debates": len(results),
        "debates_processed": 0,
        "desirable_responses": 0,
        "undesirable_responses": 0,
        "debates_skipped_low_consensus": 0,
        "total_kto_pairs": 0,
    }

    # Process each debate
    for debate in results:
        consensus = debate["final_consensus"]["consensus_strength"]

        # Skip debates with low consensus
        if consensus < min_consensus:
            stats["debates_skipped_low_consensus"] += 1
            continue

        stats["debates_processed"] += 1

        prompt = debate["question"]
        majority_answer = debate["final_consensus"]["majority_answer"]

        # Get final round responses (Round 2)
        if len(debate["rounds"]) < 2:
            continue

        final_round = debate["rounds"][-1]

        # Convert each agent's final response to KTO format
        for response in final_round:
            # Label based on consensus
            is_desirable = response["answer"] == majority_answer

            kto_entry = {
                "prompt": prompt,
                "completion": response["reasoning"],
                "label": is_desirable,
            }

            # Add metadata if requested
            if include_metadata:
                kto_entry["metadata"] = {
                    "debate_id": debate["debate_id"],
                    "agent_id": response["agent_id"],
                    "consensus_strength": consensus,
                    "answer": response["answer"],
                    "majority_answer": majority_answer,
                    "category": debate["metadata"].get("category", "unknown"),
                    "difficulty": debate["metadata"].get("difficulty", "unknown"),
                    "convergence": debate.get("convergence", "unknown"),
                }

            kto_data.append(kto_entry)

            if is_desirable:
                stats["desirable_responses"] += 1
            else:
                stats["undesirable_responses"] += 1

    stats["total_kto_pairs"] = len(kto_data)

    # Save to JSONL format
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for entry in kto_data:
            f.write(json.dumps(entry) + "\n")

    return stats


def main():
    """Main entry point for KTO data preparation."""
    parser = argparse.ArgumentParser(description="Prepare KTO training data from MACA debates")
    parser.add_argument(
        "--input",
        type=str,
        default="proprietary/data/batch_debate_results.json",
        help="Path to batch debate results JSON (default: proprietary/data/batch_debate_results.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="proprietary/data/kto_data.jsonl",
        help="Path to save KTO data (default: proprietary/data/kto_data.jsonl)",
    )
    parser.add_argument(
        "--min-consensus",
        type=float,
        default=0.6,
        help="Minimum consensus strength to include (default: 0.6)",
    )
    parser.add_argument(
        "--no-metadata", action="store_true", help="Exclude metadata from KTO entries"
    )

    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Resolve paths
    input_path = project_root / args.input
    output_path = project_root / args.output

    print(f"\n{'=' * 80}")
    print("MACA KTO DATA PREPARATION")
    print(f"{'=' * 80}\n")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Min consensus: {args.min_consensus}")
    print(f"Include metadata: {not args.no_metadata}")
    print()

    # Prepare KTO data
    stats = prepare_kto_data(
        str(input_path),
        str(output_path),
        min_consensus=args.min_consensus,
        include_metadata=not args.no_metadata,
    )

    # Print statistics
    print(f"{'─' * 80}")
    print("CONVERSION STATISTICS")
    print(f"{'─' * 80}")
    print(f"Total debates: {stats['total_debates']}")
    print(f"Debates processed: {stats['debates_processed']}")
    print(f"Debates skipped (low consensus): {stats['debates_skipped_low_consensus']}")
    print()
    print(f"Desirable responses: {stats['desirable_responses']}")
    print(f"Undesirable responses: {stats['undesirable_responses']}")
    print(f"Total KTO pairs: {stats['total_kto_pairs']}")
    print()

    if stats["total_kto_pairs"] > 0:
        desirable_ratio = stats["desirable_responses"] / stats["total_kto_pairs"]
        print(f"Desirable ratio: {desirable_ratio:.1%}")
        print(f"Undesirable ratio: {1 - desirable_ratio:.1%}")
        print()

        # Check for balance
        if 0.4 <= desirable_ratio <= 0.6:
            print("✓ Data is well-balanced")
        elif desirable_ratio > 0.8:
            print("⚠ Warning: Too many desirable responses (low diversity)")
            print("  Consider lowering min_consensus threshold")
        elif desirable_ratio < 0.2:
            print("⚠ Warning: Too many undesirable responses")
            print("  This is unusual - review debate configuration")
        else:
            print("✓ Data balance is acceptable")

    print(f"\n{'=' * 80}")
    print(f"KTO data saved to: {output_path}")
    print(f"{'=' * 80}\n")

    print("Next steps:")
    print("  1. Review the generated KTO data")
    print("  2. Split into train/val sets if needed")
    print("  3. Run KTO training with: python scripts/train_kto.py")
    print()


if __name__ == "__main__":
    main()
