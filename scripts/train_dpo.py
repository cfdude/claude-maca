#!/usr/bin/env python3
"""
DPO Training Script for MACA (Multi-Agent Consensus Alignment)

This script implements Direct Preference Optimization (DPO) training with LoRA
for parameter-efficient fine-tuning using debate-generated preference pairs.

Conservative configuration designed for small datasets.
"""

import os
import json
import torch
import sys
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Configuration
class TrainingConfig:
    # Paths (default to proprietary/ directory)
    BASE_DIR = Path(__file__).parent.parent  # Auto-detect project root
    DATA_DIR = BASE_DIR / "proprietary/data"
    OUTPUT_DIR = BASE_DIR / "proprietary/results" / "dpo_training_v1"
    MODEL_DIR = BASE_DIR / "proprietary/models" / "maca-trained-model"

    # Model
    MODEL_NAME = "Qwen/Qwen2.5-3B"  # Using HuggingFace model directly

    # Training files
    TRAIN_FILE = str(DATA_DIR / "dpo_train.jsonl")
    VAL_FILE = str(DATA_DIR / "dpo_val.jsonl")

    # LoRA Configuration (Conservative for small dataset)
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # DPO Hyperparameters (Very conservative to prevent overfitting)
    NUM_EPOCHS = 2
    LEARNING_RATE = 1e-6  # Very low LR for small dataset
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
    BETA = 0.1  # DPO temperature
    MAX_LENGTH = 1024
    MAX_PROMPT_LENGTH = 512

    # Optimization
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Early Stopping (Critical for small dataset)
    EARLY_STOPPING_PATIENCE = 1
    EARLY_STOPPING_THRESHOLD = 0.01

    # Misc
    SEED = 42
    SAVE_TOTAL_LIMIT = 2
    LOGGING_STEPS = 5


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def load_model_and_tokenizer(config):
    """Load base model and tokenizer."""
    logger.info(f"Loading base model: {config.MODEL_NAME}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Model loaded successfully")
    logger.info(f"Model dtype: {model.dtype}")
    logger.info(f"Device: {model.device}")

    return model, tokenizer


def apply_lora(model, config):
    """Apply LoRA adapters to the model."""
    logger.info("Applying LoRA configuration...")

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA applied with r={config.LORA_R}, alpha={config.LORA_ALPHA}")
    print_trainable_parameters(model)

    return model


def load_datasets(config):
    """Load training and validation datasets."""
    logger.info("Loading datasets...")

    dataset = load_dataset(
        "json", data_files={"train": config.TRAIN_FILE, "validation": config.VAL_FILE}
    )

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")

    # Log sample
    sample = dataset["train"][0]
    logger.info("Sample training example:")
    logger.info(f"  Prompt: {sample['prompt'][:100]}...")
    logger.info(f"  Chosen length: {len(sample['chosen'])}")
    logger.info(f"  Rejected length: {len(sample['rejected'])}")

    return dataset


def create_training_config(config):
    """Create DPO training configuration."""
    training_args = DPOConfig(
        output_dir=str(config.OUTPUT_DIR),
        # Training schedule
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        # Learning rate
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=config.WARMUP_RATIO,
        # Optimization
        optim="adamw_torch",
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        # DPO-specific
        beta=config.BETA,
        loss_type="sigmoid",
        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=config.LOGGING_STEPS,
        logging_first_step=True,
        report_to=["tensorboard"],
        # Memory optimization
        gradient_checkpointing=True,
        bf16=torch.backends.mps.is_available(),  # Use bf16 on Apple Silicon
        fp16=not torch.backends.mps.is_available(),  # Use fp16 on other hardware
        # Reproducibility
        seed=config.SEED,
        data_seed=config.SEED,
        # Misc
        remove_unused_columns=False,
        # Max lengths for DPO
        max_length=config.MAX_LENGTH,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
    )

    return training_args


def train(config):
    """Main training function."""
    # Set seed
    set_seed(config.SEED)

    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 80)
    logger.info("MACA DPO Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Base model: {config.MODEL_NAME}")
    logger.info(f"Training samples: {config.TRAIN_FILE}")
    logger.info(f"Validation samples: {config.VAL_FILE}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Epochs: {config.NUM_EPOCHS}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info(
        f"Batch size (effective): {config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}"
    )
    logger.info(f"LoRA r: {config.LORA_R}, alpha: {config.LORA_ALPHA}")
    logger.info(f"DPO beta: {config.BETA}")
    logger.info("=" * 80)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Apply LoRA
    model = apply_lora(model, config)

    # Load datasets
    dataset = load_datasets(config)

    # Create training config
    training_args = create_training_config(config)

    # Initialize trainer
    logger.info("Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    logger.info("Monitor progress with: tensorboard --logdir results/dpo_maca_v1")

    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()

    training_duration = end_time - start_time
    logger.info(f"Training completed in {training_duration}")

    # Log training metrics
    logger.info("=" * 80)
    logger.info("Training Results")
    logger.info("=" * 80)
    for key, value in train_result.metrics.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 80)

    # Save final model
    logger.info(f"Saving final model to {config.MODEL_DIR}...")
    trainer.save_model(str(config.MODEL_DIR))
    tokenizer.save_pretrained(str(config.MODEL_DIR))

    # Save training metrics
    metrics_file = config.OUTPUT_DIR / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    # Save config
    config_file = config.OUTPUT_DIR / "training_config.json"
    config_dict = {
        "model_name": config.MODEL_NAME,
        "num_epochs": config.NUM_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS,
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "beta": config.BETA,
        "training_duration_seconds": training_duration.total_seconds(),
    }
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Config saved to {config_file}")

    logger.info("Training complete!")
    return trainer


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = train(config)
