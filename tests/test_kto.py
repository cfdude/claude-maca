#!/usr/bin/env python3
"""
Unit tests for KTO data preparation.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_kto_data import prepare_kto_data


class TestKTODataPreparation(unittest.TestCase):
    """Test KTO data preparation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample debate results
        self.sample_debates = {
            "results": [
                {
                    "debate_id": "test_001",
                    "question": "Should we use approach A or B?",
                    "metadata": {"category": "technical", "difficulty": "intermediate"},
                    "rounds": [
                        [
                            {
                                "agent_id": "agent_1",
                                "answer": "A",
                                "reasoning": "Approach A is better because...",
                            },
                            {
                                "agent_id": "agent_2",
                                "answer": "A",
                                "reasoning": "I agree with A...",
                            },
                            {
                                "agent_id": "agent_3",
                                "answer": "B",
                                "reasoning": "Actually B is better...",
                            },
                        ],
                        [
                            {
                                "agent_id": "agent_1",
                                "answer": "A",
                                "reasoning": "After review, A is still best...",
                            },
                            {
                                "agent_id": "agent_2",
                                "answer": "A",
                                "reasoning": "Confirmed, A works...",
                            },
                            {
                                "agent_id": "agent_3",
                                "answer": "B",
                                "reasoning": "Still prefer B...",
                            },
                        ],
                    ],
                    "final_consensus": {"majority_answer": "A", "consensus_strength": 0.67},
                    "convergence": "stable",
                },
                {
                    "debate_id": "test_002",
                    "question": "What is the best strategy?",
                    "metadata": {"category": "strategy", "difficulty": "advanced"},
                    "rounds": [
                        [
                            {
                                "agent_id": "agent_1",
                                "answer": "X",
                                "reasoning": "Strategy X is optimal...",
                            },
                            {
                                "agent_id": "agent_2",
                                "answer": "Y",
                                "reasoning": "Strategy Y is better...",
                            },
                            {
                                "agent_id": "agent_3",
                                "answer": "X",
                                "reasoning": "X is the right choice...",
                            },
                        ],
                        [
                            {"agent_id": "agent_1", "answer": "X", "reasoning": "Confirmed X..."},
                            {"agent_id": "agent_2", "answer": "X", "reasoning": "Changed to X..."},
                            {"agent_id": "agent_3", "answer": "X", "reasoning": "Still X..."},
                        ],
                    ],
                    "final_consensus": {"majority_answer": "X", "consensus_strength": 1.0},
                    "convergence": "improved",
                },
                {
                    "debate_id": "test_003",
                    "question": "Which option is better?",
                    "metadata": {"category": "general", "difficulty": "basic"},
                    "rounds": [
                        [
                            {
                                "agent_id": "agent_1",
                                "answer": "unclear",
                                "reasoning": "Not sure...",
                            },
                            {
                                "agent_id": "agent_2",
                                "answer": "maybe",
                                "reasoning": "Could be either...",
                            },
                            {
                                "agent_id": "agent_3",
                                "answer": "unclear",
                                "reasoning": "Ambiguous...",
                            },
                        ],
                        [
                            {
                                "agent_id": "agent_1",
                                "answer": "unclear",
                                "reasoning": "Still unclear...",
                            },
                            {
                                "agent_id": "agent_2",
                                "answer": "maybe",
                                "reasoning": "Still ambiguous...",
                            },
                            {
                                "agent_id": "agent_3",
                                "answer": "unclear",
                                "reasoning": "Cannot decide...",
                            },
                        ],
                    ],
                    "final_consensus": {"majority_answer": "unclear", "consensus_strength": 0.4},
                    "convergence": "stable",
                },
            ],
            "metrics": {},
        }

    def test_kto_data_conversion(self):
        """Test basic KTO data conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save sample debates
            input_file = Path(tmpdir) / "debates.json"
            with open(input_file, "w") as f:
                json.dump(self.sample_debates, f)

            # Prepare KTO data
            output_file = Path(tmpdir) / "kto_data.jsonl"
            stats = prepare_kto_data(str(input_file), str(output_file), min_consensus=0.6)

            # Check stats
            self.assertEqual(stats["total_debates"], 3)
            self.assertEqual(stats["debates_processed"], 2)  # Only 2 debates meet min_consensus
            self.assertEqual(stats["debates_skipped_low_consensus"], 1)  # test_003 skipped

            # Verify output file exists
            self.assertTrue(output_file.exists())

            # Load and verify KTO data
            kto_entries = []
            with open(output_file) as f:
                for line in f:
                    kto_entries.append(json.loads(line))

            # Should have 6 entries (2 debates × 3 agents)
            self.assertEqual(len(kto_entries), 6)

    def test_label_assignment(self):
        """Test correct label assignment (desirable/undesirable)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "debates.json"
            with open(input_file, "w") as f:
                json.dump(self.sample_debates, f)

            output_file = Path(tmpdir) / "kto_data.jsonl"
            _stats = prepare_kto_data(str(input_file), str(output_file), min_consensus=0.6)

            # Load KTO data
            kto_entries = []
            with open(output_file) as f:
                for line in f:
                    kto_entries.append(json.loads(line))

            # Test debate_001: 2 agents with "A" (desirable), 1 with "B" (undesirable)
            debate_001_entries = [
                e for e in kto_entries if e["metadata"]["debate_id"] == "test_001"
            ]
            self.assertEqual(len(debate_001_entries), 3)

            desirable = [e for e in debate_001_entries if e["label"] is True]
            undesirable = [e for e in debate_001_entries if e["label"] is False]

            self.assertEqual(len(desirable), 2)  # 2 agents answered "A"
            self.assertEqual(len(undesirable), 1)  # 1 agent answered "B"

    def test_consensus_filtering(self):
        """Test that low consensus debates are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "debates.json"
            with open(input_file, "w") as f:
                json.dump(self.sample_debates, f)

            output_file = Path(tmpdir) / "kto_data.jsonl"

            # Test with min_consensus=0.7
            stats = prepare_kto_data(str(input_file), str(output_file), min_consensus=0.7)

            # Only test_002 (consensus=1.0) should pass
            self.assertEqual(stats["debates_processed"], 1)

            # Test with min_consensus=0.5
            output_file2 = Path(tmpdir) / "kto_data_2.jsonl"
            stats2 = prepare_kto_data(str(input_file), str(output_file2), min_consensus=0.5)

            # test_001 (0.67) and test_002 (1.0) should pass
            self.assertEqual(stats2["debates_processed"], 2)

    def test_metadata_preservation(self):
        """Test that metadata is correctly preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "debates.json"
            with open(input_file, "w") as f:
                json.dump(self.sample_debates, f)

            output_file = Path(tmpdir) / "kto_data.jsonl"
            _stats = prepare_kto_data(
                str(input_file), str(output_file), min_consensus=0.6, include_metadata=True
            )

            # Load and check metadata
            with open(output_file) as f:
                first_entry = json.loads(f.readline())

            self.assertIn("metadata", first_entry)
            self.assertIn("debate_id", first_entry["metadata"])
            self.assertIn("agent_id", first_entry["metadata"])
            self.assertIn("consensus_strength", first_entry["metadata"])
            self.assertIn("category", first_entry["metadata"])

    def test_no_metadata_option(self):
        """Test KTO data without metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "debates.json"
            with open(input_file, "w") as f:
                json.dump(self.sample_debates, f)

            output_file = Path(tmpdir) / "kto_data.jsonl"
            _stats = prepare_kto_data(
                str(input_file), str(output_file), min_consensus=0.6, include_metadata=False
            )

            # Load and check no metadata
            with open(output_file) as f:
                first_entry = json.loads(f.readline())

            self.assertNotIn("metadata", first_entry)
            self.assertIn("prompt", first_entry)
            self.assertIn("completion", first_entry)
            self.assertIn("label", first_entry)


if __name__ == "__main__":
    unittest.main()
