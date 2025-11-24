#!/usr/bin/env python3
"""
Scheduled batch debate processing using job scheduler.

Provides efficient bulk processing with:
- Concurrent execution
- Automatic retries
- Progress tracking
- Resource management
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add scheduler to path
sys.path.insert(0, str(Path(__file__).parent))
# Import debate processor
from run_batch_debates import DebateBatchProcessor
from scheduler import DebateScheduler


async def run_single_debate_async(
    question: Dict[str, Any], agents: int = 5, rounds: int = 2
) -> Dict[str, Any]:
    """
    Async wrapper for single debate.

    Args:
        question: Question dictionary
        agents: Number of agents
        rounds: Number of rounds

    Returns:
        Debate result dictionary
    """
    # Create processor
    processor = DebateBatchProcessor()

    # Run debate synchronously (Ollama is sync)
    # In a real async implementation, you'd use async HTTP client
    result = processor.run_single_debate(question, index=0, total=1)

    return result


async def progress_callback(status: Dict[str, Any]):
    """
    Print progress updates.

    Args:
        status: Status dictionary from scheduler
    """
    print(
        f"\rProgress: {status['completed']}/{status['total_jobs']} "
        f"({status['progress_percent']:.1f}%) | "
        f"Active: {status['active_jobs']} | "
        f"Failed: {status['failed']}",
        end="",
        flush=True,
    )


async def main():
    """Main entry point for scheduled batch debates."""
    parser = argparse.ArgumentParser(description="Run batch debates with job scheduler")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions JSON file")
    parser.add_argument(
        "--output",
        type=str,
        default="proprietary/data/scheduled_debate_results.json",
        help="Path to save results (default: proprietary/data/scheduled_debate_results.json)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=4, help="Maximum concurrent debates (default: 4)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum retry attempts (default: 3)"
    )
    parser.add_argument(
        "--agents", type=int, default=5, help="Number of agents per debate (default: 5)"
    )
    parser.add_argument(
        "--rounds", type=int, default=2, help="Number of debate rounds (default: 2)"
    )

    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Resolve paths
    questions_path = project_root / args.questions
    output_path = project_root / args.output

    print(f"\n{'=' * 80}")
    print("MACA SCHEDULED BATCH DEBATE PROCESSING")
    print(f"{'=' * 80}\n")

    print(f"Questions: {questions_path}")
    print(f"Output: {output_path}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Max retries: {args.max_retries}")
    print(f"Agents per debate: {args.agents}")
    print(f"Rounds per debate: {args.rounds}")
    print()

    # Load questions
    with open(questions_path) as f:
        questions = json.load(f)

    if isinstance(questions, dict) and "questions" in questions:
        questions = questions["questions"]

    print(f"Loaded {len(questions)} questions")
    print()

    # Create scheduler
    scheduler = DebateScheduler(max_concurrent=args.max_concurrent, max_retries=args.max_retries)

    # Add all questions as jobs
    print("Adding jobs to scheduler...")
    job_ids = await scheduler.add_batch(questions, agents=args.agents, rounds=args.rounds)

    print(f"✓ Added {len(job_ids)} jobs")
    print()

    # Run scheduler
    print(f"{'─' * 80}")
    print("Starting debate processing...")
    print(f"{'─' * 80}\n")

    await scheduler.run(run_single_debate_async, progress_callback=progress_callback)

    print("\n")  # Clear progress line

    # Get results
    results = scheduler.get_results()
    failed_jobs = scheduler.get_failed_jobs()

    print(f"{'─' * 80}")
    print("Processing complete!")
    print(f"{'─' * 80}\n")

    print(f"Total debates: {len(questions)}")
    print(f"Completed: {len(results)}")
    print(f"Failed: {len(failed_jobs)}")
    print()

    if failed_jobs:
        print("Failed jobs:")
        for job in failed_jobs:
            print(f"  - {job['job_id']}: {job['error']}")
        print()

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "results": [r["result"] for r in results if r["result"]],
        "metadata": {
            "total_questions": len(questions),
            "completed_debates": len(results),
            "failed_debates": len(failed_jobs),
            "max_concurrent": args.max_concurrent,
            "agents_per_debate": args.agents,
            "rounds_per_debate": args.rounds,
        },
        "failed_jobs": failed_jobs,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved to: {output_path}")
    print()

    print(f"{'=' * 80}")
    print("Scheduled batch processing complete!")
    print(f"{'=' * 80}\n")

    print("Next steps:")
    print("  1. Review results and failed jobs")
    print("  2. Analyze with: python scripts/analyze_batch_results.py")
    print("  3. Generate DPO or KTO training data")
    print()


if __name__ == "__main__":
    asyncio.run(main())
