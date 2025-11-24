#!/usr/bin/env python3
"""
Export Fine-Tuned Model to Ollama Format

Merges LoRA adapters with base model and creates Ollama-compatible model.
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExportConfig:
    BASE_DIR = Path(__file__).parent.parent  # Auto-detect project root
    MODEL_DIR = BASE_DIR / "proprietary/models" / "maca-trained-model"
    MERGED_MODEL_DIR = BASE_DIR / "proprietary/models" / "maca-trained-model-merged"

    BASE_MODEL_NAME = "Qwen/Qwen2.5-3B"
    OLLAMA_MODEL_NAME = "maca-model:v1"


def merge_lora_weights(config):
    """Merge LoRA adapters with base model."""
    logger.info("=" * 80)
    logger.info("Merging LoRA weights with base model")
    logger.info("=" * 80)

    # Load base model
    logger.info(f"Loading base model: {config.BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapters
    logger.info(f"Loading LoRA adapters from: {config.MODEL_DIR}")
    model = PeftModel.from_pretrained(base_model, str(config.MODEL_DIR))

    # Merge and unload
    logger.info("Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    # Save merged model
    config.MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged model to: {config.MERGED_MODEL_DIR}")
    merged_model.save_pretrained(str(config.MERGED_MODEL_DIR))

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config.MODEL_DIR))
    tokenizer.save_pretrained(str(config.MERGED_MODEL_DIR))

    logger.info("Model merged successfully!")
    return config.MERGED_MODEL_DIR


def create_modelfile(config):
    """Create Ollama Modelfile."""
    modelfile_path = config.BASE_DIR / "Modelfile.maca"

    modelfile_content = f"""FROM {config.MERGED_MODEL_DIR}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are a domain expert trained via MACA (Multi-Agent Consensus Alignment) to provide strategic, well-reasoned guidance.

Your training included:
- Multi-agent debate-generated preference pairs
- Consensus-based learning from diverse perspectives
- Direct Preference Optimization (DPO) fine-tuning

When answering questions:
- Provide detailed strategic reasoning, not surface-level answers
- Ground recommendations in relevant data and specific analysis
- Consider multiple perspectives and tradeoffs
- Include both pros and cons for important decisions
- Offer practical, actionable guidance

Always prioritize detailed, thoughtful advice over quick, generic responses.
\"\"\"
"""

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    logger.info(f"Modelfile created at: {modelfile_path}")
    return modelfile_path


def create_ollama_model(modelfile_path, config):
    """Create Ollama model from Modelfile."""
    logger.info("=" * 80)
    logger.info("Creating Ollama model")
    logger.info("=" * 80)

    # Create model
    cmd = f"ollama create {config.OLLAMA_MODEL_NAME} -f {modelfile_path}"
    logger.info(f"Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.info(f"Ollama model '{config.OLLAMA_MODEL_NAME}' created successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create Ollama model: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise


def test_ollama_model(config):
    """Test the created Ollama model."""
    logger.info("=" * 80)
    logger.info("Testing Ollama model")
    logger.info("=" * 80)

    test_prompt = "What are the key factors to consider when making a complex decision?"
    cmd = f'ollama run {config.OLLAMA_MODEL_NAME} "{test_prompt}"'

    logger.info(f"Test prompt: {test_prompt}")
    logger.info(f"Running: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True, timeout=60
        )
        logger.info("\nModel response:")
        logger.info("-" * 80)
        logger.info(result.stdout)
        logger.info("-" * 80)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to test model: {e}")
        logger.error(f"stderr: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Model test timed out after 60 seconds")


def export(config):
    """Main export function."""
    logger.info("Starting model export to Ollama format...")

    # Step 1: Merge LoRA weights
    merged_model_dir = merge_lora_weights(config)

    # Step 2: Create Modelfile
    modelfile_path = create_modelfile(config)

    # Step 3: Create Ollama model
    create_ollama_model(modelfile_path, config)

    # Step 4: Test model
    test_ollama_model(config)

    logger.info("=" * 80)
    logger.info("Export complete!")
    logger.info("=" * 80)
    logger.info(f"Merged model location: {config.MERGED_MODEL_DIR}")
    logger.info(f"Ollama model name: {config.OLLAMA_MODEL_NAME}")
    logger.info(f"Usage: ollama run {config.OLLAMA_MODEL_NAME}")
    logger.info("=" * 80)


if __name__ == "__main__":
    config = ExportConfig()
    export(config)
