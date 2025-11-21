# Data Directory

This directory is for your training data and debate results.

## Structure

Recommended structure for your data:

```
data/
├── training_questions.json     # Questions for debate generation
├── debate_results.json         # Output from run_batch_debates.py
├── dpo_training_pairs.json     # Processed DPO pairs
├── test_set.json               # Validation/test questions
└── README.md                   # This file
```

## File Formats

### Training Questions (`training_questions.json`)
```json
[
  {
    "id": "q001",
    "question": "Your question here",
    "category": "domain_category",
    "difficulty": "easy|medium|hard",
    "ground_truth": "Expected answer (optional)"
  }
]
```

### Debate Results (`debate_results.json`)
Generated automatically by `run_batch_debates.py`

### DPO Training Pairs (`dpo_training_pairs.json`)
Generated automatically from debate results with consensus filtering.

## Privacy Note

This directory is **gitignored** by default. Your data will not be committed to version control.

To use the MACA pipeline:
1. Create your question dataset in the appropriate format
2. Run `python scripts/run_batch_debates.py`
3. Results will be saved here automatically

For examples, see `proprietary.example/data/` directory.
