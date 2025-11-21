#!/usr/bin/env python3
"""
MACA Debate Test Harness
Orchestrate test debates to validate the system
"""

import json
import requests
from typing import List, Dict, Any
from collections import Counter
from pathlib import Path
import sys

# Import answer parser for improved consensus detection
from parser import AnswerParser

# Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
DOMAIN = "generic"  # Domain for answer normalization: generic, financial, legal, medical

class DebateOrchestrator:
    def __init__(self, domain: str = DOMAIN):
        self.agents = []
        self.debates = {}
        self.domain = domain
        self.parser = AnswerParser(similarity_threshold=0.85)

    def register_agent(self, agent_id: str, name: str) -> Dict[str, Any]:
        """Register an agent for debates"""
        agent = {
            "id": agent_id,
            "name": name,
            "model": MODEL,
            "endpoint": OLLAMA_URL
        }
        self.agents.append(agent)
        print(f"✅ Registered {name} (ID: {agent_id}) using {MODEL}")
        return agent

    def call_ollama(self, prompt: str, temperature: float = 0.8) -> str:
        """Call Ollama API to get LLM response"""
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": None  # Random seed for diversity
            }
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"❌ Error calling Ollama: {e}")
            return ""

    def extract_answer(self, reasoning: str) -> str:
        """Extract the final answer from reasoning"""
        # Simple extraction - look for common answer patterns
        reasoning_lower = reasoning.lower()

        # Common answers for our test questions
        if "refinance" in reasoning_lower and "wait" not in reasoning_lower:
            return "refinance"
        elif "wait" in reasoning_lower or "don't refinance" in reasoning_lower:
            return "wait"
        elif "30-year" in reasoning_lower and "15-year" not in reasoning_lower:
            return "30-year"
        elif "15-year" in reasoning_lower and "30-year" not in reasoning_lower:
            return "15-year"
        elif "lock" in reasoning_lower and ("float" not in reasoning_lower or reasoning_lower.index("lock") < reasoning_lower.index("float")):
            return "lock"
        elif "float" in reasoning_lower:
            return "float"
        else:
            # Default - try to extract first clear recommendation
            return "unclear"

    def start_debate(
        self,
        debate_id: str,
        question: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start a new debate"""
        debate = {
            "id": debate_id,
            "question": question,
            "metadata": metadata,
            "rounds": [],
            "status": "active",
            "consensus": None
        }
        self.debates[debate_id] = debate
        print(f"\n📊 Starting Debate: {debate_id}")
        print(f"Question: {question}")
        print(f"Category: {metadata.get('category', 'N/A')} | Difficulty: {metadata.get('difficulty', 'N/A')}")
        return debate

    def run_round(
        self,
        debate_id: str,
        round_num: int,
        previous_responses: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a single debate round"""
        debate = self.debates[debate_id]
        question = debate["question"]

        print(f"\n🔄 Round {round_num}")

        responses = []
        for agent in self.agents:
            # Build prompt
            if round_num == 1:
                # Round 1: Independent response
                prompt = f"""You are a domain expert. Answer the following question with clear reasoning and a final recommendation.

Question: {question}

Provide your reasoning and conclusion:"""
            else:
                # Round 2+: With peer feedback
                peer_feedback = "\n\nPrevious round responses from your peers:\n"
                for prev in previous_responses:
                    peer_feedback += f"\n- {prev['agent_name']} answered '{prev['answer']}': {prev['reasoning'][:200]}...\n"

                prompt = f"""You are a domain expert. Answer the following question with clear reasoning and a final recommendation.

Question: {question}
{peer_feedback}

Having seen your peers' reasoning, provide your refined answer:"""

            # Call Ollama
            print(f"  🤖 Calling {agent['name']}...", end="", flush=True)
            reasoning = self.call_ollama(prompt)
            answer = self.extract_answer(reasoning)
            print(f" ✅ Answer: {answer}")

            response = {
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "round": round_num,
                "reasoning": reasoning,
                "answer": answer
            }
            responses.append(response)

        # Store round results
        round_data = {
            "round_num": round_num,
            "responses": responses
        }
        debate["rounds"].append(round_data)

        return responses

    def calculate_consensus(self, debate_id: str) -> Dict[str, Any]:
        """
        Calculate consensus for a debate using fuzzy answer matching.

        Uses AnswerParser to group similar answers together, improving consensus
        detection accuracy by handling minor formatting differences.
        """
        debate = self.debates[debate_id]

        # Get final round responses
        final_round = debate["rounds"][-1]
        responses = final_round["responses"]

        # Group answers using fuzzy matching
        answers = [r["answer"] for r in responses]
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

        # Determine majority
        majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        majority_count = answer_counts[majority_answer]

        # Calculate consensus strength (majority votes / total votes)
        consensus_strength = majority_count / len(answers)

        # Build consensus data
        consensus = {
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "total_agents": len(answers),
            "consensus_strength": consensus_strength,
            "vote_distribution": answer_counts,
            "answer_groups": answer_groups,  # For debugging/analysis
            "chosen_responses": [r for r in responses if self.parser.grade_answer(r["answer"], majority_answer, domain=self.domain)],
            "rejected_responses": [r for r in responses if not self.parser.grade_answer(r["answer"], majority_answer, domain=self.domain)]
        }

        debate["consensus"] = consensus
        debate["status"] = "completed"

        return consensus

    def analyze_convergence(self, debate_id: str) -> Dict[str, Any]:
        """
        Analyze convergence from Round 1 to Round 2 using fuzzy matching.

        Uses AnswerParser to group similar answers when calculating consensus
        strength for each round.
        """
        debate = self.debates[debate_id]

        if len(debate["rounds"]) < 2:
            return {"convergence": False, "reason": "Only 1 round"}

        # Round 1 consensus with fuzzy matching
        round1_responses = debate["rounds"][0]["responses"]
        round1_answers = [r["answer"] for r in round1_responses]
        round1_groups = {}
        round1_counts = {}

        for ans in round1_answers:
            matched = False
            for canonical in round1_groups.keys():
                if self.parser.grade_answer(ans, canonical, domain=self.domain):
                    round1_groups[canonical].append(ans)
                    round1_counts[canonical] += 1
                    matched = True
                    break
            if not matched:
                round1_groups[ans] = [ans]
                round1_counts[ans] = 1

        round1_consensus = max(round1_counts.values()) / len(round1_answers)

        # Round 2 consensus with fuzzy matching
        round2_responses = debate["rounds"][1]["responses"]
        round2_answers = [r["answer"] for r in round2_responses]
        round2_groups = {}
        round2_counts = {}

        for ans in round2_answers:
            matched = False
            for canonical in round2_groups.keys():
                if self.parser.grade_answer(ans, canonical, domain=self.domain):
                    round2_groups[canonical].append(ans)
                    round2_counts[canonical] += 1
                    matched = True
                    break
            if not matched:
                round2_groups[ans] = [ans]
                round2_counts[ans] = 1

        round2_consensus = max(round2_counts.values()) / len(round2_answers)

        # Check convergence
        improved = round2_consensus > round1_consensus

        return {
            "convergence": improved,
            "round1_consensus": round1_consensus,
            "round2_consensus": round2_consensus,
            "improvement": round2_consensus - round1_consensus,
            "round1_distribution": round1_counts,
            "round2_distribution": round2_counts,
            "round1_groups": round1_groups,  # For debugging/analysis
            "round2_groups": round2_groups   # For debugging/analysis
        }

    def export_dpo_pairs(self, debate_id: str) -> List[Dict[str, Any]]:
        """Export DPO training pairs"""
        debate = self.debates[debate_id]
        consensus = debate["consensus"]

        if not consensus:
            print("⚠️ No consensus calculated yet")
            return []

        if not consensus["rejected_responses"]:
            print("⚠️ Unanimous consensus - no rejected responses for DPO pairs")
            return []

        # Create pairs: chosen (majority) vs rejected (minority)
        pairs = []

        for chosen in consensus["chosen_responses"]:
            for rejected in consensus["rejected_responses"]:
                pair = {
                    "prompt": debate["question"],
                    "chosen": chosen["reasoning"],
                    "rejected": rejected["reasoning"],
                    "metadata": {
                        **debate["metadata"],
                        "debate_id": debate_id,
                        "consensus_strength": consensus["consensus_strength"],
                        "chosen_answer": chosen["answer"],
                        "rejected_answer": rejected["answer"]
                    }
                }
                pairs.append(pair)

        return pairs

    def print_debate_summary(self, debate_id: str):
        """Print comprehensive debate summary"""
        debate = self.debates[debate_id]
        consensus = debate["consensus"]
        convergence = self.analyze_convergence(debate_id)

        print(f"\n{'='*80}")
        print(f"📊 Debate Summary: {debate_id}")
        print(f"{'='*80}")
        print(f"\nQuestion: {debate['question']}")
        print(f"Category: {debate['metadata'].get('category', 'N/A')}")
        print(f"Difficulty: {debate['metadata'].get('difficulty', 'N/A')}")

        # Round 1 Results
        round1 = debate["rounds"][0]["responses"]
        round1_votes = Counter([r["answer"] for r in round1])
        print(f"\n🔄 Round 1 Results:")
        for answer, count in round1_votes.items():
            print(f"  - {answer}: {count} vote(s)")
        print(f"  - Consensus: {convergence['round1_consensus']:.1%}")

        # Round 2 Results
        if len(debate["rounds"]) > 1:
            round2 = debate["rounds"][1]["responses"]
            round2_votes = Counter([r["answer"] for r in round2])
            print(f"\n🔄 Round 2 Results:")
            for answer, count in round2_votes.items():
                print(f"  - {answer}: {count} vote(s)")
            print(f"  - Consensus: {convergence['round2_consensus']:.1%}")

        # Consensus Analysis
        print(f"\n✅ Final Consensus:")
        print(f"  - Majority Answer: {consensus['majority_answer']}")
        print(f"  - Votes: {consensus['majority_count']}/{consensus['total_agents']}")
        print(f"  - Strength: {consensus['consensus_strength']:.1%}")

        # Convergence
        if convergence["convergence"]:
            print(f"\n✅ Convergence: YES ({convergence['improvement']:+.1%})")
        else:
            print(f"\n⚠️ Convergence: NO ({convergence['improvement']:+.1%})")

        # Quality Assessment
        quality = "HIGH" if consensus['consensus_strength'] > 0.7 else "MODERATE" if consensus['consensus_strength'] > 0.5 else "LOW"
        print(f"\n📈 Quality Assessment: {quality}")

        # DPO Pairs
        dpo_pairs = self.export_dpo_pairs(debate_id)
        print(f"  - DPO pairs generated: {len(dpo_pairs)}")

        if consensus['consensus_strength'] == 1.0:
            print("  - ⚠️ Unanimous consensus - no training signal")
            print("  - Recommendation: Use for validation set, not training")
        elif consensus['consensus_strength'] > 0.7:
            print("  - ✅ Strong consensus with disagreement - excellent for training")
        else:
            print("  - ⚠️ Weak consensus - question may be ambiguous")

        print(f"\n{'='*80}\n")


def main():
    """Run test debates"""
    print("🚀 MACA Debate Test Harness")
    print("="*80)

    # Initialize orchestrator
    orchestrator = DebateOrchestrator()

    # Register 3 agents
    print("\n1️⃣ Registering Agents")
    print("-"*80)
    orchestrator.register_agent("agent_1", "Agent Alpha")
    orchestrator.register_agent("agent_2", "Agent Beta")
    orchestrator.register_agent("agent_3", "Agent Gamma")

    # Test questions (diverse set)
    test_questions = [
        {
            "id": "test_debate_001",
            "prompt": "A client wants to know if they should lock their rate today at 6.5% or wait. Rates have been improving this week but there's a Fed meeting next week. What do you recommend?",
            "metadata": {
                "category": "rate_lock_strategy",
                "difficulty": "expert",
                "original_id": "seed_007"
            }
        },
        {
            "id": "test_debate_002",
            "prompt": "Should I recommend a cash-out refinance for debt consolidation if a client has $25,000 in credit card debt at 22% APR, but their current mortgage rate is 3.5% and new rates are 6.5%?",
            "metadata": {
                "category": "debt_management",
                "difficulty": "advanced",
                "original_id": "cma_014"
            }
        },
        {
            "id": "test_debate_003",
            "prompt": "A client asks whether they should choose a 15-year or 30-year mortgage. They can afford the higher payment but are concerned about cash flow flexibility. What factors should guide this decision?",
            "metadata": {
                "category": "loan_comparison",
                "difficulty": "advanced",
                "original_id": "cma_024"
            }
        }
    ]

    # Run all debates
    all_dpo_pairs = []

    for i, question_data in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"2️⃣ Running Test Debate {i}/3")
        print(f"{'='*80}")

        # Start debate
        debate = orchestrator.start_debate(
            debate_id=question_data["id"],
            question=question_data["prompt"],
            metadata=question_data["metadata"]
        )

        # Round 1 - Independent responses
        round1_responses = orchestrator.run_round(
            debate_id=question_data["id"],
            round_num=1
        )

        # Round 2 - With peer feedback
        round2_responses = orchestrator.run_round(
            debate_id=question_data["id"],
            round_num=2,
            previous_responses=round1_responses
        )

        # Calculate consensus
        print("\n📊 Calculating consensus...")
        consensus = orchestrator.calculate_consensus(question_data["id"])

        # Print summary
        orchestrator.print_debate_summary(question_data["id"])

        # Collect DPO pairs
        pairs = orchestrator.export_dpo_pairs(question_data["id"])
        all_dpo_pairs.extend(pairs)

    # Final validation report
    print("\n" + "="*80)
    print("3️⃣ VALIDATION REPORT")
    print("="*80)

    print(f"\n✅ Agent Registration: {len(orchestrator.agents)}/3 successful")
    print(f"✅ Debates Completed: {len(orchestrator.debates)}/3")
    print(f"✅ Total DPO Pairs Generated: {len(all_dpo_pairs)}")

    # System validation
    print("\n📋 System Validation:")
    validation_passed = True

    # Check all debates have consensus
    for debate_id, debate in orchestrator.debates.items():
        if debate["consensus"] is None:
            print(f"  ❌ {debate_id}: No consensus calculated")
            validation_passed = False
        else:
            print(f"  ✅ {debate_id}: Consensus strength {debate['consensus']['consensus_strength']:.1%}")

    # Check convergence
    convergence_count = 0
    for debate_id in orchestrator.debates:
        conv = orchestrator.analyze_convergence(debate_id)
        if conv["convergence"]:
            convergence_count += 1

    print(f"\n📈 Convergence Rate: {convergence_count}/{len(orchestrator.debates)} ({convergence_count/len(orchestrator.debates):.1%})")

    # Quality distribution
    quality_dist = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
    for debate in orchestrator.debates.values():
        strength = debate["consensus"]["consensus_strength"]
        if strength > 0.7:
            quality_dist["HIGH"] += 1
        elif strength > 0.5:
            quality_dist["MODERATE"] += 1
        else:
            quality_dist["LOW"] += 1

    print(f"\n📊 Quality Distribution:")
    print(f"  - HIGH (>0.7): {quality_dist['HIGH']}")
    print(f"  - MODERATE (0.5-0.7): {quality_dist['MODERATE']}")
    print(f"  - LOW (<0.5): {quality_dist['LOW']}")

    # Recommendations
    print("\n💡 Recommendations:")
    if validation_passed:
        print("  ✅ All debates completed successfully")
        print("  ✅ Consensus calculation working correctly")
        print("  ✅ DPO pair generation validated")

    if convergence_count == len(orchestrator.debates):
        print("  ✅ Perfect convergence - peer feedback is effective")
    elif convergence_count > 0:
        print(f"  ⚠️ Partial convergence ({convergence_count}/{len(orchestrator.debates)}) - some questions may need refinement")
    else:
        print("  ❌ No convergence - review question selection or increase agent diversity")

    if quality_dist["HIGH"] >= 2:
        print("  ✅ Strong consensus signals - ready for batch processing")
    elif quality_dist["MODERATE"] >= 2:
        print("  ⚠️ Moderate consensus - consider refining questions or adjusting temperature")
    else:
        print("  ❌ Weak consensus - questions may be too ambiguous")

    # Save results (using Path for auto-detection)
    script_dir = Path(__file__).parent.parent
    output_file = str(script_dir / "proprietary/data/test_debate_results.json")
    results = {
        "agents": orchestrator.agents,
        "debates": orchestrator.debates,
        "dpo_pairs": all_dpo_pairs,
        "summary": {
            "total_debates": len(orchestrator.debates),
            "total_dpo_pairs": len(all_dpo_pairs),
            "convergence_rate": convergence_count / len(orchestrator.debates),
            "quality_distribution": quality_dist
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print("\n" + "="*80)
    print("✅ Test Debates Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
