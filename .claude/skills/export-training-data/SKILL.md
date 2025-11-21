# Export Training Data Skill

## Purpose

Export DPO training pairs from completed debates with quality filtering and dataset preparation for fine-tuning.

## Usage

```bash
# Export all high-quality pairs
/export-training-data --min-consensus 0.7

# Export to JSONL format
/export-training-data --format jsonl --output data/training_pairs.jsonl

# Merge with seed dataset
/export-training-data --merge-seed --output data/full_training_set.json

# Prepare train/val/test splits
/export-training-data --split 70/15/15 --output-dir data/splits/
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--min-consensus` | Minimum consensus strength (0-1) | 0.7 |
| `--format` | Output format (json, jsonl, huggingface) | json |
| `--merge-seed` | Include seed dataset | false |
| `--split` | Train/val/test ratio | none |
| `--output` | Output file path | stdout |
| `--output-dir` | Directory for split files | ./data |

## Quality Filtering

Automatically filters by:
- ✅ Consensus strength ≥ threshold
- ✅ Convergence (Round 2 > Round 1)
- ❌ Unanimous (no training signal)
- ❌ Duplicate questions

## Output Formats

### JSON (default)
```json
[
  {
    "id": "debate_001_pair_1",
    "prompt": "Question",
    "chosen": "Majority reasoning",
    "rejected": "Minority reasoning",
    "metadata": {...}
  }
]
```

### JSONL
```
{"id":"debate_001_pair_1","prompt":"...","chosen":"...","rejected":"..."}
{"id":"debate_002_pair_1","prompt":"...","chosen":"...","rejected":"..."}
```

### HuggingFace Dataset
Direct compatibility with `datasets.load_dataset()`

## See Also
- `/run-debate` - Run debates
- `/analyze-consensus` - Quality analysis
