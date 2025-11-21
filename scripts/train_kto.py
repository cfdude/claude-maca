#!/usr/bin/env python3
"""
Train model using KTO (Kahneman-Tversky Optimization).

KTO is an alternative to DPO that uses individual ratings (desirable/undesirable)
instead of pairwise comparisons. This can be beneficial when you have single
responses that can be clearly labeled rather than pairs.

Based on: "KTO: Model Alignment as Prospect Theoretic Optimization"
"""

import json
import argparse
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import KTOTrainer, KTOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_kto_dataset(jsonl_path: str) -> Dataset:
    """
    Load KTO dataset from JSONL file.

    Expected format:
    {
        "prompt": str,
        "completion": str,
        "label": bool  # True = desirable, False = undesirable
    }

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        HuggingFace Dataset
    """
    data = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)

    return Dataset.from_list(data)

def train_kto(config_path: str):
    """
    Run KTO training pipeline.

    Args:
        config_path: Path to training configuration JSON
    """
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print(f"\n{'='*80}")
    print("MACA KTO TRAINING")
    print(f"{'='*80}\n")

    # Extract config
    base_model = config['base_model']
    kto_data_path = config['kto_data_path']
    output_dir = config['output_dir']

    print(f"Base model: {base_model}")
    print(f"KTO data: {kto_data_path}")
    print(f"Output dir: {output_dir}")
    print()

    # Load model and tokenizer
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"✓ Model loaded: {base_model}")
    print(f"  Parameters: {model.num_parameters():,}")
    print()

    # Apply LoRA
    if config.get('use_lora', True):
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['target_modules'],
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Prepare model for k-bit training if using quantization
        if config.get('load_in_8bit', False) or config.get('load_in_4bit', False):
            model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"✓ LoRA applied")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        print(f"  LoRA rank: {config['lora']['r']}")
        print(f"  LoRA alpha: {config['lora']['alpha']}")
        print()

    # Load KTO dataset
    print("Loading KTO dataset...")
    dataset = load_kto_dataset(kto_data_path)

    # Split train/val
    train_val_split = dataset.train_test_split(
        test_size=config.get('val_split', 0.2),
        seed=config.get('seed', 42)
    )
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']

    print(f"✓ Dataset loaded")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples: {len(val_dataset)}")
    print()

    # Count desirable vs undesirable
    desirable_count = sum(1 for ex in dataset if ex['label'])
    undesirable_count = len(dataset) - desirable_count
    print(f"  Desirable: {desirable_count} ({desirable_count/len(dataset):.1%})")
    print(f"  Undesirable: {undesirable_count} ({undesirable_count/len(dataset):.1%})")
    print()

    # KTO training config
    training_args = KTOConfig(
        output_dir=output_dir,
        num_train_epochs=config['training_args']['num_train_epochs'],
        per_device_train_batch_size=config['training_args'].get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=config['training_args'].get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=config['training_args'].get('gradient_accumulation_steps', 4),
        learning_rate=config['training_args']['learning_rate'],
        logging_steps=config['training_args'].get('logging_steps', 5),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),

        # KTO-specific parameters
        beta=config['kto_args'].get('beta', 0.1),
        desirable_weight=config['kto_args'].get('desirable_weight', 1.0),
        undesirable_weight=config['kto_args'].get('undesirable_weight', 1.0),
    )

    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  KTO beta: {training_args.beta}")
    print(f"  Desirable weight: {training_args.desirable_weight}")
    print(f"  Undesirable weight: {training_args.undesirable_weight}")
    print()

    # Initialize KTO trainer
    print("Initializing KTO trainer...")
    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("✓ Trainer initialized")
    print()

    # Train
    print(f"{'─'*80}")
    print("Starting KTO training...")
    print(f"{'─'*80}\n")

    trainer.train()

    print(f"\n{'─'*80}")
    print("Training complete!")
    print(f"{'─'*80}\n")

    # Save final model
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Model saved to: {output_dir}")
    print()

    # Save training metrics
    metrics_file = Path(output_dir) / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "final_train_loss": trainer.state.log_history[-1].get('train_loss', None),
            "final_eval_loss": trainer.state.log_history[-1].get('eval_loss', None),
            "total_steps": trainer.state.global_step,
            "config": config
        }, f, indent=2)

    print(f"✓ Metrics saved to: {metrics_file}")
    print()

    print(f"{'='*80}")
    print("KTO training pipeline complete!")
    print(f"{'='*80}\n")

    print("Next steps:")
    print(f"  1. Review TensorBoard logs: tensorboard --logdir {output_dir}")
    print(f"  2. Evaluate model performance")
    print(f"  3. Export to Ollama: python scripts/export_to_ollama.py --model {output_dir}")
    print()

def main():
    """Main entry point for KTO training."""
    parser = argparse.ArgumentParser(
        description='Train model using KTO from MACA debates'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='proprietary/configs/kto_training_config.json',
        help='Path to training configuration JSON'
    )

    args = parser.parse_args()

    # Resolve config path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_path = project_root / args.config

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print(f"\nCreate config at: {config_path}")
        print("See: proprietary.example/configs/training_config.json for template")
        return

    train_kto(str(config_path))

if __name__ == "__main__":
    main()
