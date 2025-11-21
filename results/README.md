# Results Directory

This directory is for storing experiment results, metrics, and analyses.

## Structure

Recommended structure for your results:

```
results/
├── experiment_001/
│   ├── metrics.json            # Training metrics
│   ├── evaluation.json         # Model evaluation results
│   ├── config.json             # Experiment configuration
│   └── plots/                  # Visualization plots
│       ├── loss_curve.png
│       └── accuracy_plot.png
├── experiment_002/
└── README.md                   # This file
```

## Metrics Tracking

### Training Metrics (`metrics.json`)
```json
{
  "experiment_id": "experiment_001",
  "timestamp": "2025-11-19T13:00:00Z",
  "hyperparameters": {
    "learning_rate": 1e-6,
    "num_epochs": 2,
    "batch_size": 4
  },
  "results": {
    "final_train_loss": 0.23,
    "final_eval_loss": 0.31,
    "validation_accuracy": 0.89
  }
}
```

### Evaluation Results (`evaluation.json`)
```json
{
  "model_version": "maca-v1.0",
  "test_set_size": 50,
  "accuracy": 0.89,
  "hallucination_rate": 0.08,
  "comparison_to_base": {
    "base_accuracy": 0.64,
    "improvement": 0.25
  }
}
```

## Privacy Note

This directory is **gitignored** by default. Your results will not be committed to version control.

## Visualization

Use the results to create plots and track progress over time:
- Training loss curves
- Validation accuracy trends
- Comparison to baseline models
- Consensus strength distributions

For examples, see `proprietary.example/results/` directory.
