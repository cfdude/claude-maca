# Configuration Files

This directory contains configurations for debate generation and model training.

## debate_config.json

Controls multi-agent debate parameters:

### Key Parameters

**agents.count** (M)
- Number of agents participating in debate
- Recommended: 3-7
- More agents = more diverse reasoning, but slower

**agents.temperature**
- Controls response diversity
- Range: 0.0-1.0
- Recommended: 0.7-0.9 for debates

**rounds.total** (R)
- Number of debate rounds
- Standard: 2 (independent → peer feedback)
- More rounds rarely improve quality

**consensus.optimal_range**
- Filter debates by consensus strength
- Recommended: 0.6-0.8
- Too high (0.9-1.0) = unanimous, no training signal
- Too low (<0.5) = poor quality, ambiguous

## training_config.json

Controls DPO fine-tuning parameters:

### For Small Datasets (<50 pairs)

```json
{
  "lora.r": 16,
  "training_args.learning_rate": 1e-06,
  "training_args.num_train_epochs": 2,
  "dpo_args.beta": 0.1
}
```

Conservative settings to prevent overfitting.

### For Large Datasets (>100 pairs)

```json
{
  "lora.r": 32,
  "training_args.learning_rate": 5e-06,
  "training_args.num_train_epochs": 3,
  "dpo_args.beta": 0.2
}
```

More aggressive settings for better performance.

### Key Parameters

**lora.r**
- LoRA rank (number of trainable parameters)
- 16 = ~30M params, 32 = ~60M params
- Higher = more capacity, but needs more data

**learning_rate**
- How fast model learns
- Too high = unstable, too low = slow
- 1e-6 is very conservative (safe for small data)

**num_train_epochs**
- How many times to see training data
- Too many = overfitting
- Monitor validation loss

**beta**
- Strength of preference signal in DPO
- 0.05 = subtle preferences
- 0.1 = standard
- 0.2 = strong preferences

## Customization Tips

### Start Conservative

1. Begin with small dataset settings
2. Train for 2 epochs
3. Evaluate results
4. Increase epochs/LR if validation loss still decreasing
5. Stop if validation loss increases (overfitting)

### Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir proprietary/results/dpo_training_v1/runs

# Watch for:
# - Training loss should decrease smoothly
# - Validation loss should decrease (not increase!)
# - Gap between train/val should stay small
```

### Iteration Strategy

1. **First run**: Conservative settings, see baseline
2. **Second run**: If no overfitting, increase epochs by 1
3. **Third run**: If still good, increase LR slightly
4. **Fourth run**: Generate more data if needed

## Domain-Specific Tuning

### Legal (Complex Reasoning)
- Higher lora_r (32)
- Lower temperature (0.7)
- Higher beta (0.2)
- Reason: Need strong logical consistency

### Medical (Factual Accuracy)
- Lower temperature (0.6-0.7)
- Standard lora_r (16)
- Lower beta (0.05)
- Reason: Prefer subtle corrections, not dramatic shifts

### Customer Support (Tone & Empathy)
- Higher temperature (0.8-0.9)
- Standard settings
- Higher beta (0.2)
- Reason: Capture nuanced differences in tone

### Technical (Precision)
- Lower temperature (0.7)
- Higher lora_r (32)
- Standard beta (0.1)
- Reason: Technical accuracy over diversity

## See Also

- `../../examples/configs/` - Reference configurations
- `../../docs/usage/training.md` - Training guide
- [TRL Documentation](https://huggingface.co/docs/trl/) - DPO trainer details
