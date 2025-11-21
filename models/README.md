# Models Directory

This directory is for storing trained models and checkpoints.

## Structure

Recommended structure for your models:

```
models/
├── maca-v1.0/                  # First trained model
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_args.json
├── maca-v1.1/                  # Improved version
└── README.md                   # This file
```

## Model Naming Convention

Use semantic versioning for model names:
- `maca-v1.0` - First production model
- `maca-v1.1` - Incremental improvement
- `maca-v2.0` - Major methodology change

Include metadata in each model directory:
- Training hyperparameters
- Validation metrics
- Training date
- Base model used

## Storage

This directory is **gitignored** by default. Models will not be committed to version control.

For large models, consider:
- Uploading to HuggingFace Hub
- Using Git LFS
- Storing on S3 or similar cloud storage

## Loading Models

To load a trained model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("base-model-name")
model = PeftModel.from_pretrained(base_model, "models/maca-v1.0")
```

For examples, see `proprietary.example/models/` directory.
