# Proprietary Directory - Your Domain-Specific Content

This directory structure shows where to place your domain-specific training data, models, and results.

## Purpose

The `proprietary/` directory (which you create) is for:
- **Your training questions and datasets**
- **Your trained models**
- **Your training results and evaluations**
- **Your source documents (PDFs, images, etc.)**
- **Your domain-specific documentation**

This directory is **gitignored** by default, so your proprietary content stays private.

---

## Quick Start

1. **Copy this example structure:**
   ```bash
   cp -r proprietary.example proprietary
   ```

2. **Add your training questions:**
   ```bash
   # Edit proprietary/data/training_questions.json
   # Format: Same as examples/datasets/example_questions.json
   ```

3. **Run debates on your questions:**
   ```bash
   python scripts/run_batch_debates.py \
     --config proprietary/configs/debate_config.json \
     --questions proprietary/data/training_questions.json \
     --output proprietary/data/debate_results.json
   ```

4. **Train your model:**
   ```bash
   python scripts/train_dpo.py \
     --config proprietary/configs/training_config.json \
     --train proprietary/data/dpo_train.jsonl \
     --val proprietary/data/dpo_val.jsonl \
     --output proprietary/models/your-domain-model
   ```

---

## Directory Structure

```
proprietary/
├── README.md                    # This file (customize for your domain)
├── data/                        # Your training data
│   ├── training_questions.json  # Your domain-specific questions
│   ├── debate_results.json      # Generated debate outputs
│   ├── dpo_training_pairs.json  # Generated DPO pairs
│   ├── dpo_train.jsonl          # Training split
│   └── dpo_val.jsonl            # Validation split
├── models/                      # Your trained models
│   ├── your-domain-model/       # LoRA adapters
│   └── your-domain-model-merged/ # Merged model
├── results/                     # Training artifacts
│   ├── dpo_training_v1/         # Checkpoints, logs
│   └── evaluation/              # Evaluation results
├── docs/                        # Your documentation
│   ├── training_report.md       # Training results
│   ├── evaluation_report.md     # Model evaluation
│   └── deployment_notes.md      # Deployment info
└── configs/                     # Your configurations
    ├── debate_config.json       # Debate parameters
    └── training_config.json     # Training parameters
```

---

## What Goes Here?

### `data/` - Your Training Data

**training_questions.json**
- Your domain-specific questions in DPO format
- Can be hand-written or generated from your knowledge base
- Format:
  ```json
  [
    {
      "id": "your_001",
      "prompt": "Your domain question here",
      "chosen": "High-quality detailed response",
      "rejected": "Lower-quality superficial response",
      "metadata": {
        "category": "your_category",
        "difficulty": "advanced"
      }
    }
  ]
  ```

**Generated Files** (created by scripts):
- `debate_results.json` - Full debate outputs with all agent responses
- `dpo_training_pairs.json` - Extracted DPO pairs from debates
- `dpo_train.jsonl` - Training split (80%)
- `dpo_val.jsonl` - Validation split (20%)

### `models/` - Your Trained Models

After training, you'll have:
- `your-domain-model/` - LoRA adapter weights (~100-200MB)
- `your-domain-model-merged/` - Full merged model (~3-7GB depending on base)

These can be:
- Loaded with HuggingFace Transformers
- Exported to Ollama for local serving
- Deployed to cloud inference services

### `results/` - Training Artifacts

Checkpoints and logs from training:
- `dpo_training_v1/checkpoint-N/` - Model checkpoints
- `dpo_training_v1/runs/` - TensorBoard logs
- `evaluation/` - Evaluation outputs and comparisons

### `docs/` - Your Documentation

Keep notes on:
- Training methodology and hyperparameters
- Evaluation results and model performance
- Deployment instructions for your team
- Lessons learned and future improvements

### `configs/` - Your Configurations

Domain-specific configurations:
- Number of agents, temperature, rounds
- LoRA rank, learning rate, epochs
- Consensus thresholds, filtering criteria

---

## Example Workflow

### 1. Prepare Your Questions

Create `proprietary/data/training_questions.json`:

```json
[
  {
    "id": "legal_001",
    "prompt": "What are the key considerations in a merger agreement?",
    "chosen": "[Detailed legal analysis with specific clauses...]",
    "rejected": "[Brief generic response...]",
    "metadata": {
      "category": "corporate_law",
      "difficulty": "expert"
    }
  }
]
```

### 2. Run Debates

```bash
python scripts/run_batch_debates.py \
  --config proprietary/configs/debate_config.json \
  --questions proprietary/data/training_questions.json \
  --output proprietary/data/debate_results.json
```

This generates:
- `debate_results.json` - All debate outputs
- `dpo_training_pairs.json` - Extracted DPO pairs
- Console summary of consensus and quality

### 3. Train Your Model

```bash
python scripts/train_dpo.py \
  --config proprietary/configs/training_config.json \
  --train proprietary/data/dpo_train.jsonl \
  --val proprietary/data/dpo_val.jsonl \
  --output proprietary/models/legal-advisor-v1
```

### 4. Evaluate

```bash
python scripts/evaluate_model.py \
  --model proprietary/models/legal-advisor-v1 \
  --test proprietary/data/dpo_val.jsonl \
  --output proprietary/results/evaluation/
```

### 5. Export to Ollama

```bash
python scripts/export_to_ollama.py \
  --model proprietary/models/legal-advisor-v1 \
  --name legal-advisor:v1

ollama run legal-advisor:v1 "Your domain question"
```

---

## Best Practices

### Data Quality

- **Start with 50-100 high-quality questions**
- Each question should have:
  - Clear, specific prompt
  - Detailed "chosen" response (expert-level)
  - Superficial "rejected" response (novice-level)
  - Accurate metadata (category, difficulty)

### Debate Configuration

- **Agents (M)**: 3-7 agents (5 recommended)
- **Rounds (R)**: 2 rounds (independent → peer feedback)
- **Temperature**: 0.7-0.9 for diversity
- **Consensus**: Filter 0.5-0.9 (exclude unanimous and ambiguous)

### Training Configuration

**Small Dataset (30-50 pairs)**:
- Learning rate: 1e-6 (very low)
- Epochs: 2-3
- LoRA rank: 16
- Early stopping: patience=1

**Large Dataset (100+ pairs)**:
- Learning rate: 5e-6
- Epochs: 3-5
- LoRA rank: 32
- Early stopping: patience=2

### Model Evaluation

- **Quantitative**: Validation loss, accuracy, reward margin
- **Qualitative**: Generate on test prompts, compare to base model
- **Domain Experts**: Have experts evaluate response quality
- **A/B Testing**: Deploy alongside base model, measure user satisfaction

---

## Security Considerations

### What to Protect

- Your training questions (domain expertise)
- Your trained models (competitive advantage)
- Your evaluation results (business metrics)
- Your source documents (proprietary materials)

### How It's Protected

- `proprietary/` is in `.gitignore` by default
- Never committed to version control
- Stays local to your machine or private storage
- Only you and your team have access

### Sharing with Team

If you need to share with your team:

```bash
# Create encrypted archive
tar -czf proprietary.tar.gz proprietary/
gpg -c proprietary.tar.gz  # Enter passphrase

# Share proprietary.tar.gz.gpg with team
# They decrypt: gpg -d proprietary.tar.gz.gpg | tar -xz
```

Or use:
- Private cloud storage (S3, GCS with access controls)
- Private git repository (separate from public MACA repo)
- Encrypted file sharing services

---

## Troubleshooting

### "Scripts can't find my data"

Make sure your file paths match:
- `proprietary/data/training_questions.json` (not `proprietary/questions.json`)
- `proprietary/configs/debate_config.json` (not `debate.json`)

### "Not enough training data"

Quality > quantity. Start with:
- 30-50 high-quality questions
- Run debates to generate ~100-150 DPO pairs
- Train with conservative hyperparameters
- Iterate and expand

### "Model not improving"

Check:
- Validation loss - should decrease across epochs
- Chosen vs rejected - chosen should win 80-95% of time
- Response quality - generate examples and compare
- Overfitting - if train loss << val loss, reduce epochs/LR

---

## Examples by Domain

### Legal
- Contract analysis questions
- Case law interpretation
- Compliance scenarios
- Legal reasoning challenges

### Medical
- Diagnosis scenarios
- Treatment recommendations
- Medical literature interpretation
- Clinical decision-making

### Financial
- Investment analysis
- Risk assessment
- Portfolio strategies
- Market analysis

### Technical
- Architecture decisions
- Code review scenarios
- Debugging strategies
- Performance optimization

### Customer Support
- Response quality
- Empathy and tone
- Problem-solving approaches
- Escalation handling

---

## Getting Help

- **Documentation**: See `docs/` in main repository
- **Examples**: See `examples/` for reference implementations
- **Issues**: GitHub Issues for framework questions
- **Community**: GitHub Discussions for domain-specific advice

---

## License

Your content in `proprietary/` is yours. The MACA framework is open source (see LICENSE in root directory).

---

**Remember**: The quality of your training data determines the quality of your model. Invest time in crafting excellent chosen/rejected pairs!
