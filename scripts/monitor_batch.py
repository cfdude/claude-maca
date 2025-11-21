#!/usr/bin/env python3
"""Monitor batch debate processing progress."""

import json
import time
from pathlib import Path

# Auto-detect project root and use proprietary/ directory
BASE_DIR = Path(__file__).parent.parent
RESULTS_FILE = BASE_DIR / "proprietary/data/debate_results.json"

def load_results():
    """Load current results if they exist."""
    if not RESULTS_FILE.exists():
        return None
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except:
        return None

def format_time(seconds):
    """Format seconds into readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

def print_progress():
    """Print current progress."""
    data = load_results()
    if not data:
        print("No results file found yet...")
        return

    metrics = data.get("metrics", {})
    results = data.get("results", [])

    print(f"\n{'='*80}")
    print(f"BATCH DEBATE PROGRESS")
    print(f"{'='*80}\n")

    total = metrics.get("total_debates", 49)
    completed = metrics.get("completed_debates", 0)
    progress = completed / total if total > 0 else 0

    print(f"Progress: {completed}/{total} debates ({progress:.1%})")

    if completed > 0:
        print(f"\nQuality Filtering:")
        print(f"  Kept: {metrics.get('kept_for_training', 0)}")
        print(f"  Unanimous: {metrics.get('filtered_out_unanimous', 0)}")
        print(f"  Ambiguous: {metrics.get('filtered_out_ambiguous', 0)}")

        print(f"\nConsensus:")
        avg_consensus = metrics.get('total_consensus_sum', 0) / completed
        print(f"  Average: {avg_consensus:.2f}")
        print(f"  Converged: {metrics.get('convergence_count', 0)} ({metrics.get('convergence_count', 0)/completed:.1%})")
        print(f"  Diverged: {metrics.get('divergence_count', 0)}")

        print(f"\nDPO Training Pairs:")
        print(f"  Generated: {metrics.get('dpo_pairs_generated', 0)}")

    if results:
        last = results[-1]
        print(f"\nLast Debate:")
        print(f"  ID: {last['debate_id']}")
        print(f"  Category: {last['metadata'].get('category', 'unknown')}")
        print(f"  Final consensus: {last['final_consensus']['consensus_strength']:.1%}")
        print(f"  Convergence: {last['convergence']}")
        print(f"  Filter: {last['filter_decision']}")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    print_progress()
