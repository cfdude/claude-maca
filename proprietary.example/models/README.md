# Models Directory

This directory will contain your trained models after running the training pipeline.

## What Gets Saved Here

After training, you'll have two versions of your model:

### 1. LoRA Adapters (`your-model-name/`)

**Size**: ~100-200MB
**Contents**:
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA weights
- `tokenizer_config.json` - Tokenizer settings
- `special_tokens_map.json` - Special tokens
- `tokenizer.json` - Tokenizer model

**Usage**:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
model = PeftModel.from_pretrained(base_model, "proprietary/models/your-model-name")
```

**Advantages**:
- Small file size (easy to share with team)
- Can be loaded on top of any compatible base model
- Easy to version control (if using private git)

### 2. Merged Model (`your-model-name-merged/`)

**Size**: ~3-7GB (depends on base model)
**Contents**:
- Full model with LoRA weights merged into base weights
- All config and tokenizer files
- Ready for deployment

**Usage**:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("proprietary/models/your-model-name-merged")
```

**Advantages**:
- Standalone model (no need for base model)
- Can be deployed directly
- Can be exported to Ollama

## Ollama Export

After training, export to Ollama for easy local serving:

```bash
python scripts/export_to_ollama.py \
  --model proprietary/models/your-model-name-merged \
  --name your-domain-model:v1

# Then use:
ollama run your-domain-model:v1 "Your question"
```

## Model Versioning

Recommended naming convention:

```
proprietary/models/
├── domain-advisor-v1/           # First version
├── domain-advisor-v1-merged/
├── domain-advisor-v2/           # After retraining with more data
├── domain-advisor-v2-merged/
└── domain-advisor-v2.1/         # After hyperparameter tuning
```

## Model Comparison

Keep notes on each version:

**v1** (baseline)
- Training data: 50 pairs
- Validation accuracy: 85%
- Notes: Good starting point, but struggles with edge cases

**v2** (expanded dataset)
- Training data: 150 pairs
- Validation accuracy: 92%
- Notes: Much better on edge cases, added category X questions

**v2.1** (hyperparameter tuning)
- Training data: 150 pairs (same as v2)
- Validation accuracy: 93%
- Notes: Increased beta to 0.2, slightly better preferences

## Deployment

### Option 1: Ollama (Local)
```bash
ollama run your-domain-model:v2
```

### Option 2: HuggingFace Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "proprietary/models/your-model-v2-merged",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("proprietary/models/your-model-v2-merged")
```

### Option 3: Cloud Deployment
- AWS SageMaker
- Google Vertex AI
- Azure ML
- Runpod, vast.ai, etc.

## Storage Management

Models are large. Consider:

**Keep**:
- Latest merged model (for deployment)
- Latest LoRA adapters (for retraining)
- Best performing version

**Archive** (to cloud storage):
- Older versions
- Experimental versions
- Checkpoints

**Delete**:
- Failed training runs
- Intermediate checkpoints (keep only best)

## See Also

- `../results/` - Training checkpoints and logs
- `../../docs/usage/deployment.md` - Deployment guide
- `../../scripts/export_to_ollama.py` - Export script
