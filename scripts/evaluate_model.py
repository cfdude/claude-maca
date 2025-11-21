#!/usr/bin/env python3
"""
Model Evaluation Script for MACA DPO Training

Compares base model vs fine-tuned model on validation set.
Generates qualitative comparisons and computes metrics.
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationConfig:
    BASE_DIR = Path(__file__).parent.parent  # Auto-detect project root
    DATA_DIR = BASE_DIR / "proprietary/data"
    MODEL_DIR = BASE_DIR / "proprietary/models" / "maca-trained-model"
    RESULTS_DIR = BASE_DIR / "proprietary/results" / "evaluation"

    BASE_MODEL_NAME = "Qwen/Qwen2.5-3B"
    VAL_FILE = str(DATA_DIR / "dpo_val.jsonl")

    # Generation parameters
    MAX_NEW_TOKENS = 500
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 40

    # Number of examples to generate
    NUM_EXAMPLES = 5

def load_base_model(config):
    """Load the base (untuned) model."""
    logger.info(f"Loading base model: {config.BASE_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_NAME,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    logger.info("Base model loaded successfully")
    return model, tokenizer

def load_finetuned_model(config):
    """Load the fine-tuned LoRA model."""
    logger.info(f"Loading fine-tuned model from: {config.MODEL_DIR}")

    # Load base model first
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, str(config.MODEL_DIR))

    tokenizer = AutoTokenizer.from_pretrained(str(config.MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Fine-tuned model loaded successfully")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, config):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (excluding the prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def evaluate(config):
    """Main evaluation function."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load validation dataset
    logger.info("Loading validation dataset...")
    dataset = load_dataset("json", data_files={"validation": config.VAL_FILE})
    val_data = dataset["validation"]
    logger.info(f"Validation samples: {len(val_data)}")

    # Load models
    base_model, base_tokenizer = load_base_model(config)
    finetuned_model, finetuned_tokenizer = load_finetuned_model(config)

    # Evaluate on subset of validation data
    num_examples = min(config.NUM_EXAMPLES, len(val_data))
    results = []

    logger.info(f"Generating responses for {num_examples} validation examples...")

    for i in range(num_examples):
        example = val_data[i]
        prompt = example['prompt']

        logger.info(f"\nExample {i+1}/{num_examples}")
        logger.info(f"Prompt: {prompt[:100]}...")

        # Generate from base model
        logger.info("Generating from base model...")
        base_response = generate_response(base_model, base_tokenizer, prompt, config)

        # Generate from fine-tuned model
        logger.info("Generating from fine-tuned model...")
        finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt, config)

        # Store results
        result = {
            'prompt': prompt,
            'base_response': base_response,
            'finetuned_response': finetuned_response,
            'chosen_response': example['chosen'],
            'rejected_response': example['rejected'],
            'base_length': len(base_response),
            'finetuned_length': len(finetuned_response),
            'chosen_length': len(example['chosen']),
            'rejected_length': len(example['rejected']),
        }
        results.append(result)

        # Print comparison
        logger.info("=" * 80)
        logger.info(f"PROMPT: {prompt}")
        logger.info("-" * 80)
        logger.info(f"BASE MODEL ({len(base_response)} chars):")
        logger.info(base_response[:300] + "..." if len(base_response) > 300 else base_response)
        logger.info("-" * 80)
        logger.info(f"FINE-TUNED MODEL ({len(finetuned_response)} chars):")
        logger.info(finetuned_response[:300] + "..." if len(finetuned_response) > 300 else finetuned_response)
        logger.info("-" * 80)
        logger.info(f"CHOSEN RESPONSE ({len(example['chosen'])} chars):")
        logger.info(example['chosen'][:300] + "..." if len(example['chosen']) > 300 else example['chosen'])
        logger.info("=" * 80)

    # Save results
    results_file = config.RESULTS_DIR / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Compute metrics
    avg_base_length = sum(r['base_length'] for r in results) / len(results)
    avg_finetuned_length = sum(r['finetuned_length'] for r in results) / len(results)
    avg_chosen_length = sum(r['chosen_length'] for r in results) / len(results)

    metrics = {
        'num_examples': num_examples,
        'avg_base_length': avg_base_length,
        'avg_finetuned_length': avg_finetuned_length,
        'avg_chosen_length': avg_chosen_length,
        'finetuned_vs_base_length_ratio': avg_finetuned_length / avg_base_length if avg_base_length > 0 else 0,
        'finetuned_vs_chosen_length_ratio': avg_finetuned_length / avg_chosen_length if avg_chosen_length > 0 else 0,
    }

    metrics_file = config.RESULTS_DIR / "evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Examples evaluated: {num_examples}")
    logger.info(f"Average base response length: {avg_base_length:.1f} chars")
    logger.info(f"Average fine-tuned response length: {avg_finetuned_length:.1f} chars")
    logger.info(f"Average chosen response length: {avg_chosen_length:.1f} chars")
    logger.info(f"Fine-tuned vs Base ratio: {metrics['finetuned_vs_base_length_ratio']:.2f}x")
    logger.info(f"Fine-tuned vs Chosen ratio: {metrics['finetuned_vs_chosen_length_ratio']:.2f}x")
    logger.info("=" * 80)

    logger.info("\nEvaluation complete!")
    logger.info(f"Review detailed comparisons in: {results_file}")

if __name__ == "__main__":
    config = EvaluationConfig()
    evaluate(config)
