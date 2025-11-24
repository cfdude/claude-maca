#!/usr/bin/env python3
"""
Analyze completed batch debate results and generate comprehensive report.
Enhanced with per-agent metrics, convergence analysis, and quality scoring.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Auto-detect project root and use proprietary/ directory
BASE_DIR = Path(__file__).parent.parent
RESULTS_FILE = BASE_DIR / "proprietary/data/batch_debate_results.json"
TRAINING_PAIRS_FILE = BASE_DIR / "proprietary/data/dpo_training_pairs.json"


def compute_detailed_metrics(results: List[Dict], config: Dict = None) -> Dict[str, Any]:
    """
    Compute comprehensive metrics including per-agent performance.

    Args:
        results: List of debate results
        config: Optional debate configuration

    Returns:
        Dictionary containing detailed metrics
    """
    if not results:
        return {}

    # Get agent count from first debate
    first_debate = results[0]
    num_agents = len(first_debate["rounds"][0]) if first_debate["rounds"] else 5

    # Initialize per-agent stats
    agent_stats = {}
    for i in range(num_agents):
        agent_id = (
            first_debate["rounds"][0][i]["agent_id"] if first_debate["rounds"] else f"agent_{i}"
        )
        agent_stats[agent_id] = {
            "correct_responses": 0,  # Agreements with majority
            "total_responses": 0,
            "total_reasoning_length": 0,
            "consensus_agreements": 0,
            "answer_changes": 0,  # Round 1 → Round 2 changes
            "agreement_rate": 0.0,
            "avg_response_length": 0.0,
        }

    # Consensus strength distribution
    consensus_histogram = {
        "0.0-0.3": 0,  # Very weak
        "0.3-0.5": 0,  # Weak
        "0.5-0.7": 0,  # Moderate
        "0.7-0.9": 0,  # Strong
        "0.9-1.0": 0,  # Very strong
        "1.0": 0,  # Unanimous
    }

    # Convergence patterns
    convergence_stats = {
        "improved": 0,  # Round 2 consensus > Round 1
        "degraded": 0,  # Round 2 consensus < Round 1
        "stable": 0,  # No significant change
        "avg_improvement": 0.0,
        "total_improvement": 0.0,
    }

    # Process each debate
    for debate in results:
        # Update consensus histogram
        consensus = debate["final_consensus"]["consensus_strength"]
        if consensus == 1.0:
            consensus_histogram["1.0"] += 1
        elif consensus >= 0.9:
            consensus_histogram["0.9-1.0"] += 1
        elif consensus >= 0.7:
            consensus_histogram["0.7-0.9"] += 1
        elif consensus >= 0.5:
            consensus_histogram["0.5-0.7"] += 1
        elif consensus >= 0.3:
            consensus_histogram["0.3-0.5"] += 1
        else:
            consensus_histogram["0.0-0.3"] += 1

        # Track per-agent performance
        majority_answer = debate["final_consensus"]["majority_answer"]

        # Process each round
        for round_idx, round_responses in enumerate(debate["rounds"]):
            for response in round_responses:
                agent_id = response["agent_id"]
                if agent_id not in agent_stats:
                    continue

                stats = agent_stats[agent_id]
                stats["total_responses"] += 1
                stats["total_reasoning_length"] += len(response.get("reasoning", ""))

                # Check if agent agrees with final majority
                if response["answer"] == majority_answer:
                    stats["consensus_agreements"] += 1

                # Track answer changes between rounds (only for round 2+)
                if round_idx > 0 and len(debate["rounds"]) > 1:
                    prev_answer = debate["rounds"][round_idx - 1][round_responses.index(response)][
                        "answer"
                    ]
                    if prev_answer != response["answer"]:
                        stats["answer_changes"] += 1

        # Track convergence
        if len(debate.get("consensus_progression", [])) >= 2:
            round1_consensus = debate["consensus_progression"][0]["consensus_strength"]
            round2_consensus = debate["consensus_progression"][1]["consensus_strength"]

            improvement = round2_consensus - round1_consensus
            convergence_stats["total_improvement"] += improvement

            if improvement > 0.05:
                convergence_stats["improved"] += 1
            elif improvement < -0.05:
                convergence_stats["degraded"] += 1
            else:
                convergence_stats["stable"] += 1

    # Calculate averages for agents
    for agent_id, stats in agent_stats.items():
        if stats["total_responses"] > 0:
            stats["avg_response_length"] = (
                stats["total_reasoning_length"] / stats["total_responses"]
            )
            stats["agreement_rate"] = stats["consensus_agreements"] / stats["total_responses"]

    # Calculate average improvement
    num_debates = len(results)
    if num_debates > 0:
        convergence_stats["avg_improvement"] = convergence_stats["total_improvement"] / num_debates

    # Calculate quality score
    quality_score = calculate_quality_score(consensus_histogram)

    return {
        "agent_performance": agent_stats,
        "consensus_distribution": consensus_histogram,
        "convergence_patterns": convergence_stats,
        "quality_score": quality_score,
    }


def calculate_quality_score(histogram: Dict[str, int]) -> float:
    """
    Calculate overall training data quality score.

    Ideal consensus is in 0.6-0.8 range (strong signal, not unanimous).

    Args:
        histogram: Consensus distribution histogram

    Returns:
        Quality score between 0.0 and 1.0
    """
    ideal = histogram.get("0.7-0.9", 0) + histogram.get("0.5-0.7", 0)
    poor = histogram.get("0.0-0.3", 0) + histogram.get("0.3-0.5", 0) + histogram.get("1.0", 0)
    total = sum(histogram.values())

    if total == 0:
        return 0.0

    # Score based on ideal range percentage
    ideal_ratio = ideal / total
    poor_penalty = (poor / total) * 0.5  # Penalty for poor quality

    return max(0.0, ideal_ratio - poor_penalty)


def visualize_metrics(metrics: Dict[str, Any], output_dir: Path):
    """
    Create visual reports of metrics using matplotlib.

    Args:
        metrics: Detailed metrics dictionary
        output_dir: Directory to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
    except ImportError:
        print("\nℹ️  Matplotlib not installed. Skipping visualizations.")
        print("   Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MACA Debate Analysis - Detailed Metrics", fontsize=16, fontweight="bold")

    # 1. Agent agreement rates (top-left)
    agent_perf = metrics["agent_performance"]
    agents = list(agent_perf.keys())
    agreement_rates = [stats["agreement_rate"] for stats in agent_perf.values()]

    axes[0, 0].bar(agents, agreement_rates, color="steelblue")
    axes[0, 0].set_title("Agent Agreement with Majority", fontweight="bold")
    axes[0, 0].set_ylabel("Agreement Rate")
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].axhline(y=0.7, color="r", linestyle="--", alpha=0.5, label="Target (70%)")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Consensus distribution (top-right)
    consensus_dist = metrics["consensus_distribution"]
    ranges = list(consensus_dist.keys())
    counts = list(consensus_dist.values())

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    axes[0, 1].bar(ranges, counts, color=colors)
    axes[0, 1].set_title("Consensus Strength Distribution", fontweight="bold")
    axes[0, 1].set_ylabel("Number of Debates")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Highlight optimal range
    axes[0, 1].axvspan(1.5, 3.5, alpha=0.2, color="green", label="Optimal Range")
    axes[0, 1].legend()

    # 3. Convergence patterns (bottom-left)
    convergence = metrics["convergence_patterns"]
    patterns = ["Improved", "Stable", "Degraded"]
    pattern_counts = [convergence["improved"], convergence["stable"], convergence["degraded"]]
    pattern_colors = ["#2ca02c", "#ffbb33", "#d62728"]

    axes[1, 0].bar(patterns, pattern_counts, color=pattern_colors)
    axes[1, 0].set_title("Convergence Patterns (Round 1 → Round 2)", fontweight="bold")
    axes[1, 0].set_ylabel("Number of Debates")

    # Add average improvement text
    avg_imp = convergence["avg_improvement"]
    axes[1, 0].text(
        0.5,
        0.95,
        f"Avg Improvement: {avg_imp:+.1%}",
        transform=axes[1, 0].transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 4. Quality score gauge (bottom-right)
    quality = metrics["quality_score"]

    # Create a simple gauge chart
    axes[1, 1].barh(["Quality Score"], [quality], color="green" if quality > 0.6 else "orange")
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_title("Training Data Quality Score", fontweight="bold")
    axes[1, 1].set_xlabel("Score")

    # Add quality interpretation
    if quality >= 0.7:
        quality_text = "Excellent"
        quality_color = "green"
    elif quality >= 0.5:
        quality_text = "Good"
        quality_color = "yellowgreen"
    elif quality >= 0.3:
        quality_text = "Fair"
        quality_color = "orange"
    else:
        quality_text = "Poor"
        quality_color = "red"

    axes[1, 1].text(
        quality + 0.05,
        0,
        f"{quality:.1%} - {quality_text}",
        va="center",
        fontweight="bold",
        color=quality_color,
    )

    plt.tight_layout()

    # Save visualization
    output_file = output_dir / "debate_metrics_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n📊 Metrics visualization saved to: {output_file}")
    plt.close()


def analyze_results():
    """Generate comprehensive analysis report."""

    # Load data
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    results = data["results"]
    metrics = data["metrics"]

    with open(TRAINING_PAIRS_FILE) as f:
        training_pairs = json.load(f)

    print(f"\n{'=' * 80}")
    print("MACA BATCH DEBATE ANALYSIS - FINAL REPORT")
    print(f"{'=' * 80}\n")

    # Overall metrics
    print("OVERALL PERFORMANCE")
    print(f"{'─' * 80}")
    print(f"Total debates: {metrics['total_debates']}")
    print(
        f"Completed: {metrics['completed_debates']} ({metrics['completed_debates'] / metrics['total_debates']:.1%})"
    )
    print(f"Failed: {metrics['total_debates'] - metrics['completed_debates']}")

    # Quality filtering
    print(f"\nQUALITY FILTERING")
    print(f"{'─' * 80}")
    total = metrics["completed_debates"]
    print(
        f"Kept for training: {metrics['kept_for_training']} ({metrics['kept_for_training'] / total:.1%})"
    )
    print(
        f"Filtered (unanimous): {metrics['filtered_out_unanimous']} ({metrics['filtered_out_unanimous'] / total:.1%})"
    )
    print(
        f"Filtered (ambiguous <0.5): {metrics['filtered_out_ambiguous']} ({metrics['filtered_out_ambiguous'] / total:.1%})"
    )

    # Compute detailed metrics
    detailed_metrics = compute_detailed_metrics(results)

    # Per-agent performance
    print(f"\nPER-AGENT PERFORMANCE")
    print(f"{'─' * 80}")
    agent_perf = detailed_metrics["agent_performance"]

    for agent_id, stats in sorted(agent_perf.items()):
        print(f"\n{agent_id}:")
        print(f"  Total responses: {stats['total_responses']}")
        print(f"  Agreement rate: {stats['agreement_rate']:.1%}")
        print(f"  Avg response length: {stats['avg_response_length']:.0f} chars")
        print(f"  Answer changes (R1→R2): {stats['answer_changes']}")

    # Consensus analysis
    print(f"\nCONSENSUS ANALYSIS")
    print(f"{'─' * 80}")
    print(f"Average consensus: {metrics['avg_consensus']:.2f}")
    print(f"Target range: 0.6-0.8")

    consensus_dist = detailed_metrics["consensus_distribution"]
    print(f"\nConsensus distribution:")
    for range_label, count in sorted(consensus_dist.items()):
        percentage = (count / total * 100) if total > 0 else 0
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"  {range_label:12s}: {count:3d} ({percentage:5.1f}%) {bar}")

    # Convergence analysis
    print(f"\nCONVERGENCE ANALYSIS")
    print(f"{'─' * 80}")
    convergence = detailed_metrics["convergence_patterns"]
    print(f"Improved: {convergence['improved']} ({convergence['improved'] / total:.1%})")
    print(f"Stable: {convergence['stable']} ({convergence['stable'] / total:.1%})")
    print(f"Degraded: {convergence['degraded']} ({convergence['degraded'] / total:.1%})")
    print(f"Average improvement: {convergence['avg_improvement']:+.1%}")
    print(f"Target: >50% improved")

    # Quality score
    print(f"\nTRAINING DATA QUALITY SCORE")
    print(f"{'─' * 80}")
    quality = detailed_metrics["quality_score"]
    print(f"Quality score: {quality:.1%}")

    if quality >= 0.7:
        quality_rating = "Excellent - Ready for training"
    elif quality >= 0.5:
        quality_rating = "Good - Proceed with training"
    elif quality >= 0.3:
        quality_rating = "Fair - Consider generating more data"
    else:
        quality_rating = "Poor - Review debate configuration"

    print(f"Rating: {quality_rating}")
    print(f"\nQuality is based on:")
    print(f"  • Optimal consensus range (0.6-0.8): Higher is better")
    print(f"  • Avoiding unanimous (1.0) debates: Too easy, no signal")
    print(f"  • Avoiding ambiguous (<0.5) debates: Too hard, unclear")

    # DPO training pairs
    print(f"\nDPO TRAINING PAIRS")
    print(f"{'─' * 80}")
    print(f"Total pairs generated: {metrics['dpo_pairs_generated']}")
    if metrics["kept_for_training"] > 0:
        print(
            f"Pairs per kept debate: {metrics['dpo_pairs_generated'] / metrics['kept_for_training']:.1f}"
        )
    print(f"Generation rate: {metrics['dpo_generation_rate']:.1%}")
    print(f"Target generation rate: >60%")

    # Category breakdown
    print(f"\nCATEGORY BREAKDOWN")
    print(f"{'─' * 80}")
    category_stats = defaultdict(
        lambda: {"total": 0, "kept": 0, "unanimous": 0, "ambiguous": 0, "pairs": 0}
    )

    for result in results:
        category = result["metadata"].get("category", "unknown")
        category_stats[category]["total"] += 1

        if result["filter_decision"] == "keep":
            category_stats[category]["kept"] += 1
        elif result["filter_decision"] == "unanimous":
            category_stats[category]["unanimous"] += 1
        elif result["filter_decision"] == "ambiguous":
            category_stats[category]["ambiguous"] += 1

    for pair in training_pairs:
        category = pair["metadata"].get("category", "unknown")
        category_stats[category]["pairs"] += 1

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        print(f"\n{category}:")
        print(f"  Total: {stats['total']}")
        print(f"  Kept: {stats['kept']} ({stats['kept'] / stats['total']:.1%})")
        print(f"  Unanimous: {stats['unanimous']}")
        print(f"  Ambiguous: {stats['ambiguous']}")
        print(f"  DPO pairs: {stats['pairs']}")

    # Target achievement
    print(f"\n\nTARGET ACHIEVEMENT")
    print(f"{'─' * 80}")

    targets = [
        ("Quality score >60%", quality, 0.6, "higher"),
        (
            "Convergence (improved) >50%",
            convergence["improved"] / total if total > 0 else 0,
            0.5,
            "higher",
        ),
        ("DPO generation rate >60%", metrics["dpo_generation_rate"], 0.6, "higher"),
        ("Avg consensus 0.6-0.8", metrics["avg_consensus"], (0.6, 0.8), "range"),
    ]

    for name, actual, target, check_type in targets:
        if check_type == "higher":
            status = "✓ PASS" if actual >= target else "✗ FAIL"
            print(f"{status} {name}: {actual:.1%} (target: {target:.1%})")
        elif check_type == "range":
            if isinstance(target, tuple):
                status = "✓ PASS" if target[0] <= actual <= target[1] else "✗ FAIL"
                print(f"{status} {name}: {actual:.2f} (target: {target[0]}-{target[1]})")

    # Recommendations
    print(f"\n\nRECOMMENDATIONS FOR DPO TRAINING")
    print(f"{'─' * 80}")

    if metrics["dpo_pairs_generated"] >= 100:
        print(f"✓ Sufficient training pairs ({metrics['dpo_pairs_generated']}) for DPO fine-tuning")
    else:
        print(f"✗ Insufficient training pairs ({metrics['dpo_pairs_generated']}). Target: 100-150")

    if 0.6 <= metrics["avg_consensus"] <= 0.8:
        print(f"✓ Consensus strength in optimal range ({metrics['avg_consensus']:.2f})")
    else:
        print(f"⚠ Consensus outside optimal range. Consider adjusting temperature or filtering.")

    if quality >= 0.6:
        print(f"✓ High quality training data (score: {quality:.1%})")
    else:
        print(f"⚠ Quality score could be improved. Consider:")
        print(f"   • Adjusting agent temperature for more diversity")
        print(f"   • Filtering out more unanimous debates")
        print(f"   • Reviewing question difficulty")

    # Generate visualizations
    print(f"\n\nGENERATING VISUALIZATIONS")
    print(f"{'─' * 80}")
    visualize_metrics(detailed_metrics, BASE_DIR / "proprietary/data")

    # Save detailed metrics to JSON
    metrics_output = BASE_DIR / "proprietary/data/detailed_metrics.json"
    with open(metrics_output, "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"📄 Detailed metrics saved to: {metrics_output}")

    print(f"\n\n{'=' * 80}")
    print("Analysis complete. Ready for DPO training phase.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    analyze_results()
