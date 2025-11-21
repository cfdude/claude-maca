# Results Directory

This directory contains training artifacts: checkpoints, logs, and evaluation results.

## Structure

```
results/
├── dpo_training_v1/          # First training run
│   ├── checkpoint-9/         # Epoch 1 checkpoint
│   ├── checkpoint-18/        # Epoch 2 checkpoint (best)
│   ├── training_config.json  # Hyperparameters used
│   ├── training_metrics.json # Final metrics
│   └── runs/                 # TensorBoard logs
├── dpo_training_v2/          # Second training run
└── evaluation/               # Evaluation results
    ├── comparison_results.json
    └── evaluation_metrics.json
```

## Training Checkpoints

Each training run creates:

**checkpoint-N/**
- Model state at that training step
- Optimizer state
- Can resume training from here
- Automatically saved at each epoch

**Best checkpoint** (selected by lowest validation loss):
- Automatically loaded at end of training
- Becomes the final model

## TensorBoard Logs

View training progress in real-time:

```bash
tensorboard --logdir proprietary/results/dpo_training_v1/runs
```

Open http://localhost:6006 to see:
- **Loss curves**: Training vs validation loss
- **Learning rate**: How LR changes over time
- **Gradients**: Check for exploding/vanishing gradients
- **Rewards**: DPO chosen vs rejected rewards

## Training Metrics

**training_metrics.json**
```json
{
  "train_loss": 0.234,
  "eval_loss": 0.289,
  "train_runtime": 518.5,
  "train_samples_per_second": 0.13,
  "epoch": 2.0
}
```

## Evaluation Results

After training, run evaluation:

```bash
python scripts/evaluate_model.py \
  --model proprietary/models/your-model \
  --test proprietary/data/dpo_val.jsonl \
  --output proprietary/results/evaluation/
```

Generates:

**comparison_results.json**
- Side-by-side: base model vs fine-tuned vs ground truth
- Shows improvement on each test example

**evaluation_metrics.json**
```json
{
  "accuracy": 0.92,
  "avg_chosen_reward": 2.45,
  "avg_rejected_reward": -1.23,
  "reward_margin": 3.68
}
```

## What to Monitor

### During Training

✅ **Good Signs**:
- Training loss decreasing smoothly
- Validation loss decreasing
- Small gap between train and validation loss
- Rewards: chosen > rejected consistently

❌ **Warning Signs**:
- Validation loss increasing (overfitting)
- Train loss much lower than val loss (overfitting)
- Rewards: chosen < rejected (model confused)
- Loss spiking or oscillating (LR too high)

### After Training

✅ **Good Results**:
- Validation accuracy > 80%
- Reward margin > 2.0
- Generated responses better than base model
- No catastrophic forgetting on general questions

⚠️ **Needs Improvement**:
- Accuracy < 70%
- Reward margin < 1.0
- Responses not noticeably better
- Model forgot basic knowledge

## Troubleshooting

### Overfitting

**Symptoms**:
- Train loss keeps decreasing
- Val loss starts increasing
- Gap between train/val grows

**Solutions**:
- Reduce epochs
- Reduce learning rate
- Increase LoRA dropout
- Get more training data

### Underfitting

**Symptoms**:
- Both train and val loss stay high
- Model not improving

**Solutions**:
- Increase epochs
- Increase learning rate
- Increase LoRA rank
- Check data quality

### Unstable Training

**Symptoms**:
- Loss spiking or oscillating
- NaN values
- Gradients exploding

**Solutions**:
- Reduce learning rate
- Enable gradient clipping (max_grad_norm=1.0)
- Reduce batch size
- Check for bad data points

## Storage Management

Checkpoints are large. Consider:

**Keep**:
- Best checkpoint (automatically selected)
- Final merged model

**Delete**:
- Intermediate checkpoints
- Failed runs
- Old experimental runs

**Archive** (to cloud storage):
- Successful training runs with good metrics
- For future reference or comparison

## See Also

- `../models/` - Final trained models
- `../../docs/usage/training.md` - Training guide
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization tool
