# MACA Claude Code Integration

**Complete guide to using MACA with Claude Code**

This document covers everything you need to integrate MACA (Multi-Agent Consensus Alignment) into your Claude Code workflows, including plugin installation, MCP server setup, and agent/skill usage.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [MCP Server Setup](#mcp-server-setup)
- [Plugin Components](#plugin-components)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Overview

The MACA Claude Code integration provides:

1. **MCP Server**: Tools for debate orchestration, consensus calculation, and training data export
2. **Specialized Agents**: Pre-configured agents for different MACA workflows
3. **Skills**: Reusable workflows for common tasks
4. **Hooks**: Automation hooks for post-debate processing

**Architecture**:
```
Claude Code
    ↓
MCP Server (debate tools)
    ↓
Ollama (local LLM serving)
    ↓
MACA Scripts (training & evaluation)
```

---

## Prerequisites

### Required Software

- **Claude Code**: Latest version from [claude.ai/code](https://claude.ai/code)
- **Node.js**: 18.0.0 or higher
- **Python**: 3.8 or higher
- **Ollama**: Latest version from [ollama.ai](https://ollama.ai/)

### Verify Installation

```bash
# Check Node.js version
node --version  # Should be >= v18.0.0

# Check Python version
python --version  # Should be >= 3.8

# Check Ollama
ollama --version
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/maca.git
cd maca
```

### 2. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build MCP Server

```bash
cd mcp-server
npm install
npm run build
cd ..
```

### 4. Install Claude Code Plugin

```bash
cd plugin
./.claude-plugin/install.sh
cd ..
```

The installer will:
- Register the plugin with Claude Code
- Configure MCP server connection
- Set up agents, skills, and hooks

### 5. Verify Installation

Start Claude Code and type:
```
You: /maca-help
```

If the plugin is installed correctly, you'll see the MACA help menu.

---

## MCP Server Setup

### Manual Configuration (If Needed)

If automatic installation fails, add this to your Claude Code MCP configuration:

**Location**: `~/.claude/mcp.json`

```json
{
  "mcpServers": {
    "maca-debate": {
      "command": "node",
      "args": ["/path/to/maca/mcp-server/dist/index.js"],
      "env": {
        "OLLAMA_URL": "http://localhost:11434",
        "DEBUG": "false"
      }
    }
  }
}
```

### Environment Variables

**`OLLAMA_URL`**: Ollama API endpoint (default: `http://localhost:11434`)

**`DEBUG`**: Enable debug logging (default: `false`)

### Start Ollama

```bash
ollama serve
```

### Pull Base Model

```bash
ollama pull qwen2.5:3b
```

### Test MCP Connection

In Claude Code:
```
You: Test the MACA MCP server connection

Claude: I'll use the mcp__maca__connect_llm tool to test the connection...
```

---

## Plugin Components

### Agents

**Specialized agents for MACA workflows**

#### 1. `debate-orchestrator`

**Purpose**: Run multi-agent debates and generate training data

**When to use**:
- Starting a new debate on a question
- Processing a batch of questions
- Testing debate configurations

**Example**:
```
You: Run a MACA debate on "Should we refactor or rebuild this authentication module?"

Claude: I'll use the debate-orchestrator agent to run a multi-agent debate on this question...
```

**Configuration**:
```json
{
  "num_agents": 5,
  "max_rounds": 2,
  "temperature": 0.9,
  "model": "qwen2.5:3b"
}
```

#### 2. `dpo-trainer`

**Purpose**: Train models using Direct Preference Optimization

**When to use**:
- Training a model on debate results
- Fine-tuning with LoRA
- Evaluating trained models

**Example**:
```
You: Train a DPO model on the debate results in data/debates.json

Claude: I'll use the dpo-trainer agent to set up DPO training with optimal hyperparameters...
```

**Configuration**:
```json
{
  "lora_r": 16,
  "lora_alpha": 32,
  "learning_rate": 1e-6,
  "num_epochs": 2,
  "beta": 0.1
}
```

#### 3. `dataset-curator`

**Purpose**: Manage training dataset quality and selection

**When to use**:
- Filtering debates by consensus strength
- Analyzing dataset quality
- Selecting optimal training pairs

**Example**:
```
You: Analyze the quality of debates in data/batch_results.json and filter for optimal training

Claude: I'll use the dataset-curator agent to analyze consensus distribution and filter for the optimal 0.6-0.8 range...
```

**Quality Metrics**:
- Consensus strength distribution
- Convergence patterns
- Answer diversity
- Training data quality score

### Skills

**Reusable workflows for common tasks**

#### 1. `run-debate`

**Purpose**: Execute a single debate with full orchestration

**Usage**:
```
You: /run-debate "Should we prioritize feature A or B?"

Claude: I'll orchestrate a debate with 5 agents across 2 rounds...
```

**Parameters**:
- `question`: The debate question (required)
- `agents`: Number of agents (default: 5)
- `rounds`: Number of rounds (default: 2)
- `temperature`: Sampling temperature (default: 0.9)

**Output**:
```json
{
  "question": "Should we prioritize feature A or B?",
  "rounds": [...],
  "consensus": {
    "majority_answer": "Feature A",
    "consensus_strength": 0.8,
    "convergence": "improved"
  }
}
```

#### 2. `export-training-data`

**Purpose**: Convert debate results into DPO training pairs

**Usage**:
```
You: /export-training-data data/debates.json

Claude: I'll export debate results to DPO training format...
```

**Parameters**:
- `input`: Path to debate results JSON (required)
- `output`: Path to output JSONL (default: `data/dpo_training.jsonl`)
- `min_consensus`: Minimum consensus strength (default: 0.6)
- `max_consensus`: Maximum consensus strength (default: 0.8)

**Output Format**:
```jsonl
{"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}
{"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}
```

#### 3. `analyze-consensus`

**Purpose**: Detailed analysis of consensus patterns

**Usage**:
```
You: /analyze-consensus data/batch_results.json

Claude: I'll analyze consensus patterns and convergence rates...
```

**Output**:
- Per-agent agreement rates
- Consensus strength distribution
- Convergence analysis (improved/stable/degraded)
- Training data quality score
- Visualizations (if matplotlib available)

### Hooks

**Automation hooks for post-debate processing**

#### `post-debate-export.sh`

**Trigger**: After debate completion

**Actions**:
1. Export debate results to training data
2. Analyze consensus quality
3. Generate summary report

**Configuration**: `plugin/hooks/hooks.json`

```json
{
  "post-debate": {
    "enabled": true,
    "script": "./hooks/post-debate-export.sh",
    "auto_export": true,
    "min_consensus": 0.6,
    "max_consensus": 0.8
  }
}
```

---

## Usage Examples

### Example 1: Single Debate

```
You: I need help deciding whether to use PostgreSQL or MongoDB for our new feature. Run a MACA debate.

Claude: I'll orchestrate a multi-agent debate on this database choice question.

[Uses debate-orchestrator agent]
[Runs 5 agents across 2 rounds]
[Calculates consensus]

Result:
- Majority consensus: PostgreSQL (80% agreement)
- Key reasoning: ACID compliance, relational data structure, team expertise
- Minority view: MongoDB (20% agreement)
- Reasoning: Document flexibility, JSON-native storage

Training pair generated:
- Chosen: PostgreSQL reasoning (consensus path)
- Rejected: MongoDB reasoning (minority path)
```

### Example 2: Batch Processing

```
You: Process all questions in examples/datasets/architecture_decisions.json

Claude: I'll process the batch of 15 architecture questions using the debate-orchestrator agent.

[Runs debates sequentially]
[Generates progress updates]
[Exports training data automatically via hook]

Summary:
- 15 debates completed
- Average consensus: 0.72
- Convergence rate: 47% improved
- Training pairs exported: 12 (filtered for optimal range)
- Quality score: 68.3% (Good)
```

### Example 3: Training Pipeline

```
You: Train a model on the debate results, then evaluate it

Claude: I'll use the dpo-trainer agent to set up a complete training pipeline.

[Prepares training data]
[Configures LoRA with r=16]
[Trains for 2 epochs with lr=1e-6]
[Evaluates on validation set]

Results:
- Validation accuracy: 87.3% (chosen > rejected)
- Training loss: 0.23
- Validation loss: 0.31
- No catastrophic forgetting detected
- Model exported to: models/maca-architecture-v1
```

### Example 4: Quality Analysis

```
You: Analyze the quality of debates in data/week1_debates.json

Claude: I'll use the dataset-curator agent to perform comprehensive quality analysis.

[Analyzes 47 debates]
[Calculates per-agent metrics]
[Generates consensus distribution]

Findings:
- Agent Alpha: 73.5% agreement rate (best performer)
- Agent Delta: 54.2% agreement rate (needs investigation)
- Optimal consensus range (0.6-0.8): 68% of debates
- Convergence: 42% improved, 51% stable, 7% degraded
- Quality score: 68.3% (Good - proceed with training)

Recommendations:
- Filter out 15 debates below 0.6 consensus
- Investigate Agent Delta's performance
- Consider adding 10-15 more debates for better distribution
```

### Example 5: Custom Domain Parser

```
You: I want to use MACA for legal reasoning. How do I add a legal domain parser?

Claude: I'll guide you through creating a legal domain parser.

[Shows parser.py structure]
[Explains normalization requirements]
[Provides example code]

Here's the implementation for legal citation normalization:

```python
class LegalParser(BaseDomainParser):
    def normalize(self, answer: str) -> str:
        # Normalize U.S.C. citations
        answer = re.sub(r'(\d+)\s+U\.?S\.?C\.?\s*§?\s*(\d+)',
                       r'\1 USC \2', answer)
        # Normalize case citations
        answer = re.sub(r'\d+\s+F\.\s*\d+d\s+\d+',
                       lambda m: m.group(0).replace(' ', ''), answer)
        return answer.lower().strip()
```

Would you like me to:
1. Create test cases for this parser?
2. Add it to the main parser.py file?
3. Create example legal questions for testing?
```

---

## Troubleshooting

### Common Issues

#### 1. MCP Server Connection Failed

**Error**: `Failed to connect to MACA MCP server`

**Solutions**:
```bash
# Check Ollama is running
ollama serve

# Verify MCP server build
cd mcp-server
npm run build

# Check MCP configuration
cat ~/.claude/mcp.json

# Test Ollama connection
curl http://localhost:11434/api/tags
```

#### 2. Agent Not Found

**Error**: `Unknown agent: debate-orchestrator`

**Solutions**:
```bash
# Reinstall plugin
cd plugin
./.claude-plugin/install.sh

# Verify agent files exist
ls plugin/agents/

# Check Claude Code plugin directory
ls ~/.claude/plugins/
```

#### 3. Debate Fails with Empty Responses

**Error**: `All agents returned empty responses`

**Solutions**:
```bash
# Check model is available
ollama list

# Pull model if missing
ollama pull qwen2.5:3b

# Test model directly
ollama run qwen2.5:3b "Test question"

# Check temperature isn't too high
# Edit debate config: temperature should be 0.7-0.9, not >1.0
```

#### 4. Training Fails with CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size in training_config.json
{
  "training_args": {
    "per_device_train_batch_size": 2,  # Reduce from 4
    "gradient_accumulation_steps": 8   # Increase from 4
  }
}

# Or use CPU training (slower but no memory limit)
{
  "training_args": {
    "no_cuda": true
  }
}
```

#### 5. Consensus Calculation Seems Wrong

**Error**: Consensus shows 100% but agents gave different answers

**Solutions**:
```python
# Check similarity threshold in debate_config.json
{
  "parser": {
    "similarity_threshold": 0.85  # Lower if too aggressive (try 0.75)
  }
}

# Enable debug mode to see normalization
{
  "debug": {
    "show_normalized_answers": true
  }
}
```

### Debug Mode

Enable detailed logging:

**MCP Server**:
```json
{
  "mcpServers": {
    "maca-debate": {
      "env": {
        "DEBUG": "true"
      }
    }
  }
}
```

**Python Scripts**:
```bash
export MACA_DEBUG=1
python scripts/run_batch_debates.py --debug ...
```

---

## Advanced Configuration

### Custom Agent Configuration

Create a custom agent by adding a new file to `plugin/agents/`:

**Example**: `plugin/agents/my-custom-agent.md`

```markdown
# My Custom Agent

You are a specialized agent for [your use case].

## Primary Responsibilities

1. [Responsibility 1]
2. [Responsibility 2]

## Tools Available

- mcp__maca__start_debate
- mcp__maca__get_agent_response
- mcp__maca__calculate_consensus

## Workflow

When the user asks you to [trigger], follow these steps:

1. Validate input parameters
2. Configure debate with custom settings
3. Run debate orchestration
4. Analyze and present results

## Configuration

Use these optimal settings:
- num_agents: 7 (higher diversity for complex questions)
- temperature: 0.95 (maximum exploration)
- rounds: 3 (extra round for refinement)
```

### Custom Skill

Create a custom skill in `plugin/skills/my-skill/`:

**Structure**:
```
plugin/skills/my-skill/
├── skill.md          # Main skill definition
└── config.json       # Skill configuration
```

**Example**: `plugin/skills/my-skill/skill.md`

```markdown
# My Custom Skill

## Description

This skill performs [custom workflow].

## Usage

/my-skill [parameter1] [parameter2]

## Steps

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Output

Returns [description of output].
```

### Custom Hook

Add custom automation in `plugin/hooks/`:

**Example**: `plugin/hooks/my-hook.sh`

```bash
#!/bin/bash

# Custom post-processing hook
# Trigger: [when this runs]

INPUT_FILE=$1
OUTPUT_DIR=$2

# Your custom logic here
echo "Processing: $INPUT_FILE"

# Example: Upload results to S3
aws s3 cp $INPUT_FILE s3://my-bucket/debates/

# Example: Send Slack notification
curl -X POST $SLACK_WEBHOOK \
  -d "{\"text\": \"Debate completed: $INPUT_FILE\"}"
```

**Register hook** in `plugin/hooks/hooks.json`:

```json
{
  "my-hook": {
    "enabled": true,
    "script": "./hooks/my-hook.sh",
    "trigger": "post-debate",
    "async": true
  }
}
```

---

## Performance Tips

### 1. Optimize Debate Configuration

```json
{
  "performance": {
    "parallel_agents": true,        // Run agents in parallel (faster)
    "cache_responses": true,        // Cache LLM responses
    "batch_size": 10                // Process questions in batches
  }
}
```

### 2. Model Selection

```bash
# Faster models for development
ollama pull qwen2.5:3b         # 3B params, ~2GB RAM

# Balanced models for production
ollama pull qwen2.5:7b         # 7B params, ~5GB RAM

# Higher quality (slower)
ollama pull llama3:8b          # 8B params, ~6GB RAM
```

### 3. Training Optimization

```python
# Use quantization for faster training
{
  "quantization": {
    "load_in_8bit": true,
    "load_in_4bit": false
  }
}

# Use flash attention if available
{
  "training_args": {
    "use_flash_attention_2": true
  }
}
```

---

## Resources

- **Plugin README**: `plugin/README.md`
- **MCP Server README**: `mcp-server/README.md`
- **Example Configurations**: `examples/configs/`
- **Example Datasets**: `examples/datasets/`

---

## Contributing

Want to improve the Claude Code integration? See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Adding new agents
- Creating custom skills
- Developing hooks
- Testing guidelines

---

## Support

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Discord**: [Join our community](#) (coming soon)

---

**Happy debating! 🎯**

*Last Updated: November 2025*
