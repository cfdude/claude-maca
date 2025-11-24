#!/usr/bin/env python3
"""
MACA Debate Batch Processor
Orchestrates debates on 49 questions to generate DPO training pairs.

Based on validation recommendations:
- M=5 agents (increased from M=3)
- Temperature=0.9 (increased from 0.8)
- Filter consensus: 0.5-0.9 (discard unanimous and ambiguous)
"""

import json
import time
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any

# Import answer parser for improved consensus detection
from parser import AnswerParser

# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:3b"
TEMPERATURE = 0.9
NUM_AGENTS = 5
MAX_ROUNDS = 2
DOMAIN = "generic"  # Domain for answer normalization: generic, financial, legal, medical

# File paths (default to proprietary/ directory)
BASE_DIR = Path(__file__).parent.parent  # Auto-detect project root
QUESTIONS_FILE = BASE_DIR / "proprietary/data/training_questions.json"
RESULTS_FILE = BASE_DIR / "proprietary/data/debate_results.json"
TRAINING_PAIRS_FILE = BASE_DIR / "proprietary/data/dpo_training_pairs.json"

# Agent configuration
AGENTS = [
    {"id": "agent_alpha", "name": "Agent Alpha"},
    {"id": "agent_beta", "name": "Agent Beta"},
    {"id": "agent_gamma", "name": "Agent Gamma"},
    {"id": "agent_delta", "name": "Agent Delta"},
    {"id": "agent_epsilon", "name": "Agent Epsilon"},
]


class DebateBatchProcessor:
    """Orchestrates batch debate processing for MACA training data generation."""

    def __init__(self, domain: str = DOMAIN):
        self.results = []
        self.training_pairs = []
        self.domain = domain
        self.parser = AnswerParser(similarity_threshold=0.85)
        self.metrics = {
            "total_debates": 0,
            "completed_debates": 0,
            "filtered_out_unanimous": 0,
            "filtered_out_ambiguous": 0,
            "kept_for_training": 0,
            "dpo_pairs_generated": 0,
            "convergence_count": 0,
            "divergence_count": 0,
            "no_change_count": 0,
            "total_consensus_sum": 0,
            "avg_consensus": 0,
        }

    def call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API with configured temperature."""
        try:
            data = json.dumps(
                {
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                    },
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "tokens": result.get("eval_count", 0),
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_answer(self, response: str) -> str:
        """Extract a clear answer from agent response."""
        response_lower = response.lower()

        # Common answer patterns
        if "should wait" in response_lower or "recommend waiting" in response_lower:
            return "wait"
        if "should refinance" in response_lower or "recommend refinancing" in response_lower:
            return "refinance"
        if "yes" in response_lower[:100]:
            return "yes"
        if "no" in response_lower[:100]:
            return "no"

        # Default to analyzing first sentence
        first_sentence = response.split(".")[0].lower()
        if "wait" in first_sentence:
            return "wait"
        if "refinance" in first_sentence:
            return "refinance"
        if "lock" in first_sentence:
            return "lock"
        if "float" in first_sentence:
            return "float"

        return "unclear"

    def run_debate_round(
        self, question: str, round_num: int, peer_responses: List[Dict] = None
    ) -> List[Dict]:
        """Run a single debate round with all agents."""
        responses = []

        for agent in AGENTS:
            # Construct prompt
            if round_num == 1:
                # Round 1: Independent response
                prompt = f"""You are a domain expert analyzing this question:

{question}

Provide a clear, well-reasoned answer with your recommendation. Start with your clear position, then explain your reasoning."""
            else:
                # Round 2: With peer feedback
                peer_summary = "\n\n".join(
                    [
                        f'- {r["agent_name"]} said "{r["answer"]}": {r["reasoning"][:200]}...'
                        for r in peer_responses
                    ]
                )
                prompt = f"""You are a domain expert analyzing this question:

{question}

Round 1 responses from your peers:
{peer_summary}

Having seen your peers' reasoning, provide your refined answer. You may change your position or maintain it, but explain your reasoning clearly."""

            # Call Ollama
            print(f"  [{agent['name']}] Generating response...", end=" ", flush=True)
            result = self.call_ollama(prompt)

            if not result["success"]:
                print(f"ERROR: {result['error']}")
                return None

            reasoning = result["response"]
            answer = self.extract_answer(reasoning)

            responses.append(
                {
                    "agent_id": agent["id"],
                    "agent_name": agent["name"],
                    "answer": answer,
                    "reasoning": reasoning,
                    "tokens": result["tokens"],
                }
            )

            print(f"[{answer}]")

        return responses

    def calculate_consensus(self, responses: List[Dict]) -> Dict[str, Any]:
        """
        Calculate consensus from agent responses using fuzzy answer matching.

        Uses AnswerParser to group similar answers together, improving consensus
        detection accuracy by handling minor formatting differences.
        """
        answers = [r["answer"] for r in responses]

        # Group answers using fuzzy matching
        answer_groups = {}  # {canonical_answer: [equivalent_answers]}
        answer_counts = {}  # {canonical_answer: count}

        for ans in answers:
            # Check if this answer is equivalent to any existing group
            matched = False
            for canonical in answer_groups.keys():
                if self.parser.grade_answer(ans, canonical, domain=self.domain):
                    # This answer matches an existing group
                    answer_groups[canonical].append(ans)
                    answer_counts[canonical] += 1
                    matched = True
                    break

            if not matched:
                # Create new group with this answer as canonical
                answer_groups[ans] = [ans]
                answer_counts[ans] = 1

        # Find majority
        majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        majority_count = answer_counts[majority_answer]

        consensus_strength = majority_count / len(answers)

        return {
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "total_agents": len(answers),
            "consensus_strength": consensus_strength,
            "answer_distribution": answer_counts,
            "unique_answers": len(answer_counts),
            "answer_groups": answer_groups,  # For debugging/analysis
        }

    def generate_dpo_pairs(self, debate_result: Dict) -> List[Dict]:
        """Generate DPO training pairs from debate result."""
        # Only generate pairs if there's a clear majority and minority
        consensus = debate_result["final_consensus"]
        if consensus["consensus_strength"] >= 1.0:
            # Unanimous - no minority to learn from
            return []

        # Get final round responses
        final_round = debate_result["rounds"][-1]
        majority_answer = consensus["majority_answer"]

        # Find chosen (majority) and rejected (minority) responses
        chosen_responses = [r for r in final_round if r["answer"] == majority_answer]
        rejected_responses = [r for r in final_round if r["answer"] != majority_answer]

        if not rejected_responses:
            return []

        # Create DPO pairs
        pairs = []
        for rejected in rejected_responses:
            # Pick the best chosen response (longest reasoning as proxy for quality)
            chosen = max(chosen_responses, key=lambda x: len(x["reasoning"]))

            pairs.append(
                {
                    "prompt": debate_result["question"],
                    "chosen": chosen["reasoning"],
                    "rejected": rejected["reasoning"],
                    "metadata": {
                        "debate_id": debate_result["debate_id"],
                        "consensus_strength": consensus["consensus_strength"],
                        "category": debate_result["metadata"]["category"],
                        "difficulty": debate_result["metadata"]["difficulty"],
                        "majority_answer": majority_answer,
                        "rejected_answer": rejected["answer"],
                    },
                }
            )

        return pairs

    def run_single_debate(self, question_data: Dict, index: int, total: int) -> Dict:
        """Run a complete debate on a single question."""
        debate_id = question_data["id"]
        question = question_data["prompt"]
        metadata = question_data.get("metadata", {})

        print(f"\n[{index}/{total}] Debate: {debate_id}")
        print(f"Question: {question[:80]}...")
        print(
            f"Category: {metadata.get('category', 'unknown')} | Difficulty: {metadata.get('difficulty', 'unknown')}"
        )

        debate_result = {
            "debate_id": debate_id,
            "question": question,
            "metadata": metadata,
            "rounds": [],
            "consensus_progression": [],
            "start_time": time.time(),
        }

        # Round 1: Independent responses
        print("\nRound 1 (Independent):")
        round1_responses = self.run_debate_round(question, 1)
        if not round1_responses:
            return None

        round1_consensus = self.calculate_consensus(round1_responses)
        debate_result["rounds"].append(round1_responses)
        debate_result["consensus_progression"].append(round1_consensus)

        print(
            f"  Consensus: {round1_consensus['consensus_strength']:.1%} on '{round1_consensus['majority_answer']}'"
        )

        # Round 2: With peer feedback
        print("\nRound 2 (With Peer Feedback):")
        round2_responses = self.run_debate_round(question, 2, round1_responses)
        if not round2_responses:
            return None

        round2_consensus = self.calculate_consensus(round2_responses)
        debate_result["rounds"].append(round2_responses)
        debate_result["consensus_progression"].append(round2_consensus)

        print(
            f"  Consensus: {round2_consensus['consensus_strength']:.1%} on '{round2_consensus['majority_answer']}'"
        )

        # Calculate convergence
        consensus_change = (
            round2_consensus["consensus_strength"] - round1_consensus["consensus_strength"]
        )
        if consensus_change > 0.01:
            convergence = "converged"
        elif consensus_change < -0.01:
            convergence = "diverged"
        else:
            convergence = "stable"

        print(f"  Convergence: {convergence} ({consensus_change:+.1%})")

        debate_result["final_consensus"] = round2_consensus
        debate_result["convergence"] = convergence
        debate_result["consensus_change"] = consensus_change
        debate_result["end_time"] = time.time()
        debate_result["duration_seconds"] = debate_result["end_time"] - debate_result["start_time"]

        return debate_result

    def apply_quality_filtering(self, debate_result: Dict) -> str:
        """Apply quality filtering based on consensus strength."""
        consensus = debate_result["final_consensus"]["consensus_strength"]

        if consensus >= 1.0:
            return "unanimous"  # No training signal
        elif consensus < 0.5:
            return "ambiguous"  # Low quality
        else:
            return "keep"  # Good for training

    def update_metrics(self, debate_result: Dict, filter_decision: str):
        """Update running metrics."""
        self.metrics["completed_debates"] += 1
        self.metrics["total_consensus_sum"] += debate_result["final_consensus"][
            "consensus_strength"
        ]

        if filter_decision == "unanimous":
            self.metrics["filtered_out_unanimous"] += 1
        elif filter_decision == "ambiguous":
            self.metrics["filtered_out_ambiguous"] += 1
        else:
            self.metrics["kept_for_training"] += 1

        if debate_result["convergence"] == "converged":
            self.metrics["convergence_count"] += 1
        elif debate_result["convergence"] == "diverged":
            self.metrics["divergence_count"] += 1
        else:
            self.metrics["no_change_count"] += 1

    def save_progress(self):
        """Save current results and training pairs to disk."""
        # Save debate results
        with open(RESULTS_FILE, "w") as f:
            json.dump({"results": self.results, "metrics": self.metrics}, f, indent=2)

        # Save DPO training pairs
        with open(TRAINING_PAIRS_FILE, "w") as f:
            json.dump(self.training_pairs, f, indent=2)

        print(f"\n✓ Progress saved to {RESULTS_FILE}")

    def run_batch(self):
        """Run batch processing on all questions."""
        # Load questions
        with open(QUESTIONS_FILE) as f:
            questions = json.load(f)

        self.metrics["total_debates"] = len(questions)

        print(f"\n{'=' * 80}")
        print(f"MACA Batch Debate Processing")
        print(f"{'=' * 80}")
        print(f"Total questions: {len(questions)}")
        print(f"Agents: {NUM_AGENTS} (M=5)")
        print(f"Temperature: {TEMPERATURE}")
        print(f"Model: {MODEL}")
        print(f"{'=' * 80}\n")

        start_time = time.time()

        # Process each question
        for idx, question_data in enumerate(questions, 1):
            try:
                # Run debate
                debate_result = self.run_single_debate(question_data, idx, len(questions))
                if not debate_result:
                    print(f"✗ Debate failed, skipping...")
                    continue

                # Apply quality filtering
                filter_decision = self.apply_quality_filtering(debate_result)
                debate_result["filter_decision"] = filter_decision

                print(f"  Filter: {filter_decision}")

                # Generate DPO pairs if kept
                if filter_decision == "keep":
                    pairs = self.generate_dpo_pairs(debate_result)
                    self.training_pairs.extend(pairs)
                    self.metrics["dpo_pairs_generated"] += len(pairs)
                    print(f"  DPO pairs: {len(pairs)}")

                # Update metrics
                self.update_metrics(debate_result, filter_decision)

                # Save result
                self.results.append(debate_result)

                # Save progress every 5 debates
                if idx % 5 == 0:
                    self.save_progress()
                    self._print_interim_metrics(idx, len(questions), start_time)

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                self.save_progress()
                sys.exit(1)
            except Exception as e:
                print(f"\n✗ Error processing {question_data['id']}: {e}")
                continue

        # Final save
        self.save_progress()

        # Calculate final metrics
        if self.metrics["completed_debates"] > 0:
            self.metrics["avg_consensus"] = (
                self.metrics["total_consensus_sum"] / self.metrics["completed_debates"]
            )
            self.metrics["convergence_rate"] = (
                self.metrics["convergence_count"] / self.metrics["completed_debates"]
            )
            self.metrics["dpo_generation_rate"] = (
                self.metrics["kept_for_training"] / self.metrics["completed_debates"]
            )

        # Print final report
        self._print_final_report(time.time() - start_time)

    def _print_interim_metrics(self, current: int, total: int, start_time: float):
        """Print interim progress metrics."""
        elapsed = time.time() - start_time
        rate = current / elapsed
        remaining = (total - current) / rate if rate > 0 else 0

        print(f"\n{'─' * 80}")
        print(f"Progress: {current}/{total} ({current / total:.1%})")
        print(
            f"Time elapsed: {elapsed / 60:.1f} min | Estimated remaining: {remaining / 60:.1f} min"
        )
        print(f"Avg consensus: {self.metrics['total_consensus_sum'] / current:.2f}")
        print(
            f"Convergence: {self.metrics['convergence_count']}/{current} ({self.metrics['convergence_count'] / current:.1%})"
        )
        print(f"DPO pairs: {self.metrics['dpo_pairs_generated']}")
        print(f"{'─' * 80}\n")

    def _print_final_report(self, total_time: float):
        """Print comprehensive final report."""
        m = self.metrics

        print(f"\n\n{'=' * 80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'=' * 80}\n")

        print(f"Time & Performance:")
        print(f"  Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.1f} hours)")
        print(f"  Avg time per debate: {total_time / m['completed_debates']:.1f} seconds")
        print(f"  Debates per hour: {m['completed_debates'] / (total_time / 3600):.1f}")

        print(f"\nDebate Results:")
        print(f"  Total debates: {m['total_debates']}")
        print(f"  Completed: {m['completed_debates']}")
        print(f"  Failed: {m['total_debates'] - m['completed_debates']}")

        print(f"\nQuality Filtering:")
        print(
            f"  Kept for training: {m['kept_for_training']} ({m['kept_for_training'] / m['completed_debates']:.1%})"
        )
        print(
            f"  Filtered (unanimous): {m['filtered_out_unanimous']} ({m['filtered_out_unanimous'] / m['completed_debates']:.1%})"
        )
        print(
            f"  Filtered (ambiguous): {m['filtered_out_ambiguous']} ({m['filtered_out_ambiguous'] / m['completed_debates']:.1%})"
        )

        print(f"\nConsensus Analysis:")
        print(f"  Average consensus: {m['avg_consensus']:.2f}")
        print(f"  Converged: {m['convergence_count']} ({m['convergence_rate']:.1%})")
        print(
            f"  Diverged: {m['divergence_count']} ({m['divergence_count'] / m['completed_debates']:.1%})"
        )
        print(
            f"  No change: {m['no_change_count']} ({m['no_change_count'] / m['completed_debates']:.1%})"
        )

        print(f"\nDPO Training Pairs:")
        print(f"  Total pairs generated: {m['dpo_pairs_generated']}")
        print(f"  Pairs per kept debate: {m['dpo_pairs_generated'] / m['kept_for_training']:.1f}")
        print(f"  Generation rate: {m['dpo_generation_rate']:.1%}")

        print(f"\nTarget Achievement:")
        targets = [
            ("Convergence rate >50%", m["convergence_rate"], 0.5),
            ("DPO generation rate >60%", m["dpo_generation_rate"], 0.6),
            ("Avg consensus 0.6-0.8", m["avg_consensus"], None),
        ]

        for name, actual, target in targets:
            if target is None:
                status = "✓" if 0.6 <= actual <= 0.8 else "✗"
                print(f"  {status} {name}: {actual:.2f}")
            else:
                status = "✓" if actual >= target else "✗"
                print(f"  {status} {name}: {actual:.1%} (target: {target:.1%})")

        print(f"\nOutput Files:")
        print(f"  Debate results: {RESULTS_FILE}")
        print(f"  DPO training pairs: {TRAINING_PAIRS_FILE}")
        print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    processor = DebateBatchProcessor()
    processor.run_batch()
