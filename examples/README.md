# MACA Examples

This directory contains example datasets and configurations to help you get started with the MACA (Multi-Agent Consensus Alignment) framework.

## Example Datasets

### `datasets/example_questions.json`

A small dataset of 5 generic questions demonstrating the DPO (Direct Preference Optimization) format:

- **example_001**: Technical decision-making
- **example_002**: Stakeholder communication
- **example_003**: Metrics and measurement
- **example_004**: Refactor vs rebuild decisions
- **example_005**: Feature prioritization

Each example includes:
- `prompt`: The question or scenario
- `chosen`: High-quality, detailed response (what the model should learn to prefer)
- `rejected`: Lower-quality, superficial response (what the model should learn to avoid)
- `metadata`: Category, difficulty, and debate-worthiness flags

## Example Configurations

### `configs/debate_config.json`

Configuration for running multi-agent debates:
- **Agents**: 5 agents using qwen2.5:3b model
- **Rounds**: 2 rounds (independent → peer feedback)
- **Consensus**: Majority voting with 0.5-0.9 optimal range
- **Filtering**: Excludes unanimous and ambiguous debates
- **DPO Generation**: Automatically generates training pairs

### `configs/training_config.json`

Configuration for DPO fine-tuning:
- **Model**: Qwen2.5-3B base model
- **LoRA**: r=16, alpha=32, dropout=0.1 (conservative for small datasets)
- **Training**: 2 epochs, learning rate 1e-6
- **DPO**: Beta=0.1, sigmoid loss
- **Early Stopping**: Enabled with patience=1

## Using These Examples

### 1. Run a Test Debate

```bash
# Start Ollama server
ollama serve

# Pull the base model
ollama pull qwen2.5:3b

# Run debate on example questions
python scripts/run_batch_debates.py \
  --config examples/configs/debate_config.json \
  --questions examples/datasets/example_questions.json
```

### 2. Train with Example Data

```bash
# Convert example dataset to training format
python scripts/prepare_dataset.py \
  --input examples/datasets/example_questions.json \
  --output data/

# Train DPO model
python scripts/train_dpo.py \
  --config examples/configs/training_config.json
```

### 3. Adapt to Your Domain

Replace the example questions with your own domain-specific questions:

1. Create a JSON file with your questions in the same format
2. Update the `category` field to match your domain
3. Ensure `chosen` responses are high-quality, detailed answers
4. Ensure `rejected` responses are superficial or incorrect
5. Run debates and training using your dataset

## Domain Examples

The MACA framework can be applied to any domain requiring preference-based fine-tuning:

- **Legal**: Legal reasoning, case analysis, contract review
- **Medical**: Clinical decision-making, diagnosis support, treatment recommendations
- **Technical**: Software architecture, code review, debugging strategies
- **Customer Support**: Response quality, empathy, problem-solving
- **Education**: Explanation quality, pedagogical approaches, tutoring strategies

## Next Steps

1. Review the [documentation](../docs/) for detailed setup instructions
2. Check the [MCP server](../mcp-server/) for debate orchestration tools
3. Explore the [plugin](../plugin/) for Claude Code integration
4. Join our community to share your domain-specific applications!

---

**Note**: These examples use generic decision-making scenarios. Replace with your own domain-specific content to create value in your particular use case.
