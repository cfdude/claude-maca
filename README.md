# MACA: Multi-Agent Consensus Alignment

**A framework for training language models through multi-agent debate and preference learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.15172-b31b1b.svg)](https://arxiv.org/abs/2509.15172)

---

## 📖 Overview

**MACA** (Multi-Agent Consensus Alignment) is a complete implementation of the research paper "Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment" (Meta AI, 2024). It provides tools for improving LLM reasoning through multi-agent debate and preference-based fine-tuning.

### Key Features

- 🤖 **Multi-Agent Debates**: Orchestrate M agents across R rounds to generate diverse reasoning
- 🎯 **Consensus Calculation**: Majority voting to identify preferred vs rejected responses
- 📊 **DPO Training**: Direct Preference Optimization using consensus-derived training pairs
- 🔌 **Claude Code Integration**: Complete plugin with agents, skills, hooks, and MCP server
- 🚀 **End-to-End Pipeline**: From debate generation to model fine-tuning and evaluation
- 🌐 **Domain-Agnostic**: Apply to any domain requiring preference-based alignment

### Research Results (from original MACA paper)

- **+27.6%** improvement on GSM8K (self-consistency)
- **+23.7%** improvement on MATH (single-agent reasoning)
- **+22.4%** improvement on MATH (Pass@20 sampling)
- **+42.7%** improvement on MathQA (multi-agent decision-making)

---

## 🧠 How It Works

### 1. Multi-Agent Debate

**M agents** (LLM clones) engage in **R rounds** of structured debate:

```
Round 1 (Independent):
  Agent 1: [independent response]
  Agent 2: [independent response]
  ...
  Agent M: [independent response]

Round 2 (Peer Feedback):
  Agent 1: [revised response after seeing peers]
  Agent 2: [revised response after seeing peers]
  ...
  Agent M: [revised response after seeing peers]
```

### 2. Answer Parsing & Normalization

Before consensus calculation, answers are normalized to handle domain-specific formats:

```python
# Financial domain
"$1,000" == "1000" == "1K" == "one thousand"  # All equivalent

# Legal domain
"42 U.S.C. § 1983" == "42 USC 1983" == "42 USC Section 1983"  # Citation variations

# Medical domain
"E11.9" == "E11.9 (Type 2 diabetes)" == "e11.9"  # ICD code variations

# Generic domain
"Yes, I agree" == "yes" == "Yes"  # Fuzzy matching (85% similarity)
```

**How it works:**
- Domain-specific normalizers handle format variations
- Fuzzy string matching groups semantically equivalent answers
- Configurable similarity threshold (default: 85%)
- Reduces false disagreements in consensus calculation

**Configuration:**
```json
{
  "parser": {
    "domain": "financial",  // or "legal", "medical", "generic"
    "similarity_threshold": 0.85
  }
}
```

### 3. Consensus Calculation

Majority voting determines consensus using normalized answers:

```
Question: "Should we prioritize feature A or B?"

Round 2 Results:
  Agent 1: "A" ← Majority (3/5)
  Agent 2: "A" ← Majority (3/5)
  Agent 3: "B" ← Minority (2/5)
  Agent 4: "A" ← Majority (3/5)
  Agent 5: "B" ← Minority (2/5)

Consensus Strength: 60% (3/5 agents agree)
```

### 4. DPO Pair Generation

Convert debates into training data:

```json
{
  "prompt": "Should we prioritize feature A or B?",
  "chosen": "[Reasoning from Agent 1, 2, or 4 - majority consensus]",
  "rejected": "[Reasoning from Agent 3 or 5 - minority opinion]",
  "metadata": {
    "consensus_strength": 0.6,
    "convergence": "improved"
  }
}
```

### 5. Fine-Tuning with DPO

Train model to prefer consensus reasoning:

```
Base Model → MACA Debates → DPO Training → Aligned Model
  (generic)    (diverse       (preference    (consensus-
                reasoning)     learning)      aligned)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) (for local LLM serving)
- Node.js 18+ (for MCP server)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/maca.git
cd maca

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build MCP server
cd mcp-server
npm install
npm run build
cd ..
```

### Run Your First Debate

```bash
# Start Ollama
ollama serve

# Pull a base model
ollama pull qwen2.5:3b

# Run debate on example questions
python scripts/run_batch_debates.py \
  --config examples/configs/debate_config.json \
  --questions examples/datasets/example_questions.json \
  --output data/debate_results.json
```

### Train with DPO or KTO

MACA supports two training methods:

**DPO (Direct Preference Optimization)** - Recommended for debate pairs:
```bash
python scripts/train_dpo.py \
  --config examples/configs/training_config.json \
  --train data/dpo_train.jsonl \
  --val data/dpo_val.jsonl \
  --output models/your-model-dpo
```

**KTO (Kahneman-Tversky Optimization)** - Alternative using individual ratings:
```bash
# Prepare KTO data from debates
python scripts/prepare_kto_data.py \
  --input data/batch_debate_results.json \
  --output data/kto_data.jsonl \
  --min-consensus 0.6

# Train with KTO
python scripts/train_kto.py \
  --config examples/configs/kto_training_config.json
```

**When to use each:**
- **DPO**: Use when you have natural pairs (chosen vs rejected) from debates. Better for MACA's multi-agent setup.
- **KTO**: Use when you have individual responses with clear good/bad labels. Simpler data format, similar results.

### Export to Ollama

```bash
# Export fine-tuned model
python scripts/export_to_ollama.py \
  --model models/your-model \
  --name your-model:latest

# Use your model
ollama run your-model:latest "Your question here"
```

---

## 🛠️ Architecture

### Components

```
maca/
├── mcp-server/              # MCP server for debate orchestration
│   ├── connect_llm          # Register agents
│   ├── start_debate         # Initialize debate
│   ├── get_agent_response   # Get agent reasoning
│   ├── calculate_consensus  # Majority voting
│   └── export_training_data # Generate DPO pairs
│
├── plugin/                  # Claude Code plugin
│   ├── agents/              # Specialized agents
│   │   ├── debate-orchestrator.md
│   │   ├── dpo-trainer.md
│   │   └── dataset-curator.md
│   ├── skills/              # Reusable workflows
│   │   ├── run-debate/
│   │   ├── export-training-data/
│   │   └── analyze-consensus/
│   └── hooks/               # Automation hooks
│
├── scripts/                 # Python utilities
│   ├── run_batch_debates.py
│   ├── train_dpo.py
│   ├── evaluate_model.py
│   └── export_to_ollama.py
│
└── examples/                # Example data & configs
    ├── datasets/
    └── configs/
```

### Claude Code Integration

Install the MACA plugin:

```bash
cd plugin
./.claude-plugin/install.sh
```

Use in Claude Code:

```
You: Run a MACA debate on whether to refactor or rebuild this module

Claude: I'll use the debate-orchestrator agent to run a multi-agent debate...
```

---

## 📚 Documentation

- **[Setup Guide](docs/setup/)** - Installation and configuration
- **[Usage Guide](docs/usage/)** - Running debates and training models
- **[API Reference](docs/api/)** - MCP tools and Python APIs
- **[Examples](examples/)** - Sample datasets and configurations

### Analyzing Results

After running debates, analyze the results with detailed metrics:

```bash
python scripts/analyze_batch_results.py
```

**Metrics provided:**

1. **Per-Agent Performance**
   - Agreement rate with majority consensus
   - Average response length
   - Answer changes between rounds
   - Individual agent quality assessment

2. **Consensus Distribution**
   - Histogram of consensus strengths
   - Optimal range targeting (0.6-0.8)
   - Quality filtering breakdown

3. **Convergence Analysis**
   - Improved/stable/degraded patterns
   - Average improvement from Round 1 to Round 2
   - Convergence rate tracking

4. **Quality Score**
   - Overall training data quality (0.0-1.0)
   - Based on optimal consensus distribution
   - Recommendations for improvement

5. **Visualizations** (requires matplotlib)
   - Agent agreement charts
   - Consensus distribution histogram
   - Convergence pattern analysis
   - Quality score gauge

**Example output:**
```
PER-AGENT PERFORMANCE
────────────────────────────────────────────────────────────────────────────────

agent_alpha:
  Total responses: 98
  Agreement rate: 73.5%
  Avg response length: 1247 chars
  Answer changes (R1→R2): 12

TRAINING DATA QUALITY SCORE
────────────────────────────────────────────────────────────────────────────────
Quality score: 68.3%
Rating: Good - Proceed with training

Quality is based on:
  • Optimal consensus range (0.6-0.8): Higher is better
  • Avoiding unanimous (1.0) debates: Too easy, no signal
  • Avoiding ambiguous (<0.5) debates: Too hard, unclear
```

---

## 🎯 Use Cases

MACA can be applied to any domain requiring preference-based alignment:

### Professional Domains
- **Legal**: Legal reasoning, case analysis, contract review
- **Medical**: Clinical decision-making, diagnosis support, treatment recommendations
- **Financial**: Investment analysis, risk assessment, portfolio strategies
- **Technical**: Software architecture, debugging strategies, code review
- **Customer Support**: Response quality, empathy training, problem-solving

### Development Workflows
- **Code Review**: Multi-agent validation of code changes
- **Architecture Decisions**: Consensus-based design choices
- **Test Generation**: Diverse test case creation
- **Documentation**: Quality assessment and improvement
- **Refactoring**: Evaluating refactor vs rebuild decisions

---

## 🧪 Research Background

### Original Paper

**Title**: Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment
**Authors**: Ankur Samanta, Akshayaa Magesh, Youliang Yu, et al.
**Institution**: Meta AI, Meta Superintelligence Labs, Columbia University, Cornell Tech
**Date**: September 19, 2024
**arXiv**: [2509.15172v2](https://arxiv.org/abs/2509.15172)
**Code**: [github.com/facebookresearch/maca](https://github.com/facebookresearch/maca)

### Key Contributions

1. **Self-Consistency as Intrinsic Property**: Formalizes self-consistency as a learnable trait
2. **Multi-Agent Debate Framework**: M agents × R rounds generates diverse reasoning paths
3. **Training Objectives**: Supports MV-SFT, MV-GRPO, MV-DPO, MV-KTO
4. **Convergence Analysis**: Tracks reasoning improvement across debate rounds

### Implementation Details

This implementation focuses on:
- **MV-DPO** (Majority-Vote Direct Preference Optimization) as primary training method
- **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Local deployment** via Ollama for privacy and control
- **Claude Code integration** for seamless development workflows

---

## 🔧 Configuration

### Debate Configuration

Key parameters in `examples/configs/debate_config.json`:

- **agents.count**: Number of agents (M) - recommended: 3-7
- **agents.temperature**: Response diversity - recommended: 0.7-0.9
- **rounds.total**: Number of debate rounds (R) - recommended: 2
- **consensus.optimal_range**: Filter debates by consensus strength - recommended: 0.6-0.8

### Training Configuration

Key parameters in `examples/configs/training_config.json`:

- **lora.r**: LoRA rank - recommended: 16 (small datasets), 32 (large datasets)
- **training_args.learning_rate**: Learning rate - recommended: 1e-6 (small), 5e-6 (large)
- **training_args.num_train_epochs**: Training epochs - recommended: 2-3
- **dpo_args.beta**: DPO temperature - recommended: 0.1

---

## 📊 Expected Results

Based on our implementation and the original research:

### Quality Metrics

- **Consensus Strength**: Target 0.6-0.8 (sweet spot for training signal)
- **Convergence Rate**: 30-50% of debates should show Round 1 → Round 2 improvement
- **DPO Pair Quality**: Chosen responses should be 15-25% longer and more detailed

### Training Outcomes

With 50-100 high-quality debate pairs:
- **Validation Accuracy**: 80-95% (chosen > rejected)
- **Response Quality**: +10-30% improvement in detail and reasoning
- **Zero Catastrophic Forgetting**: Model retains general knowledge

### Small Dataset Performance

Our conservative approach works with as few as 30-40 pairs:
- Use learning_rate ≤ 1e-6
- Limit to 2-3 epochs
- Apply LoRA with r=16
- Enable early stopping
- Monitor train vs validation loss

---

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- **Domain Examples**: Share your domain-specific applications
- **Training Recipes**: Optimize hyperparameters for different dataset sizes
- **Evaluation Methods**: Improve quality assessment metrics
- **Documentation**: Expand guides and tutorials
- **Bug Fixes**: Report and fix issues

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📖 Citation

This project implements the Multi-Agent Consensus Alignment (MACA) approach from Meta AI research.

### Original Research

**Paper**: Samanta et al., "Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment" (2024)

**Developed by**: Meta AI in collaboration with Meta Superintelligence Labs and the LIINC Lab at Columbia University

If you use this framework in your research or application, please cite the original paper:

```bibtex
@misc{samanta2024maca,
  title={Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment},
  author={Ankur Samanta and Akshayaa Magesh and Youliang Yu and Runzhe Wu and Ayush Jain and Daniel Jiang and Boris Vidolov and Paul Sajda and Yonathan Efroni and Kaveh Hassani},
  year={2024},
  eprint={2509.15172},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://doi.org/10.48550/arXiv.2509.15172}
}
```

### Related Resources

- **Original Paper**: https://arxiv.org/abs/2509.15172
- **Meta MACA Repository**: https://github.com/facebookresearch/maca (MIT License)

### Implementation Differences

This implementation differs from Meta's research code in several key ways:

| Aspect | Meta MACA (Research) | This Implementation (Practical) |
|--------|---------------------|--------------------------------|
| **Focus** | Mathematical reasoning benchmarks (GSM8K, MATH) | Domain-agnostic business applications |
| **Infrastructure** | Multi-GPU HuggingFace clusters | Local Ollama + single GPU |
| **Training Methods** | MV-SFT, MV-GRPO, MV-KTO, MV-DPO | MV-DPO (Direct Preference Optimization) |
| **Complexity** | Research-grade comprehensive implementation | Production-ready minimal dependencies |
| **Deployment** | HuggingFace checkpoints | Ollama models (instant local use) |
| **Target Users** | ML researchers | Domain experts and developers |

Our implementation prioritizes accessibility, simplicity, and practical deployment while maintaining the core MACA methodology.

---

## 🙏 Acknowledgments

- **Meta AI** for the original MACA research and implementation
- **Hugging Face** for Transformers and TRL libraries
- **Ollama** for local LLM serving
- **Anthropic** for Claude Code and MCP framework

---

## 📮 Contact & Community

- **Issues**: [GitHub Issues](https://github.com/yourusername/maca/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/maca/discussions)
- **Email**: your.email@example.com

---

## ⚠️ Disclaimer

This is an independent implementation of the MACA research paper. Results may vary based on:
- Base model quality
- Dataset size and quality
- Domain complexity
- Training configuration
- Hardware resources

Always evaluate thoroughly before production use.

---

**Built with ❤️ for the open-source AI community**

*Last Updated: November 2025*
