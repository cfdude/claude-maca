# MACA Debate Plugin

> **Multi-Agent Consensus Alignment for LLM Training**
>
> Generate high-quality training data through multi-agent debates with automated DPO pair generation

---

## 🎯 Overview

The MACA Debate plugin implements the Multi-Agent Consensus Alignment framework for generating training data through orchestrated debates between LLM agents. Based on the research paper *"Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment"* (Meta AI, 2025), this plugin enables:

- **Multi-agent debates** with 3-5 LLM clones via Ollama
- **Consensus-based preference learning** using majority voting
- **Automated DPO pair generation** (chosen/rejected responses)
- **Quality filtering** by consensus strength and convergence
- **End-to-end training pipeline** from debates to fine-tuned models

Research shows **+27.6%** improvement on reasoning tasks after MACA training.

---

## ✨ Key Features

### 🎭 Complete Plugin Architecture
- **3 Specialized Agents**: Debate Orchestrator, DPO Trainer, Dataset Curator
- **3 Workflow Skills**: run-debate, export-training-data, analyze-consensus
- **Automated Hooks**: Post-debate export tracking and metrics
- **MCP Server**: 10 tools for debate orchestration

### 🤖 Multi-Agent Debates
- Register M agents (typically 3-5) using same base model
- Run R rounds (typically 2) with peer feedback
- Calculate consensus via majority voting
- Generate DPO pairs: majority = chosen, minority = rejected

### 📊 Quality Management
- Consensus strength metrics (0-1 scale)
- Convergence analysis (Round 1 → Round 2 improvement)
- Automatic quality filtering (>0.7 recommended)
- Training signal optimization

### 🚀 End-to-End Pipeline
- Debate orchestration → DPO pair generation → Fine-tuning → Evaluation
- Supports HuggingFace TRL, PEFT, LoRA
- Export formats: JSON, JSONL, HuggingFace Dataset

---

## 📦 Installation

### Automatic Installation

```bash
# From your project directory
cd /path/to/your/project

# Install the plugin
claude plugin install maca-debate
```

The installer will:
1. ✅ Copy agents to `.claude/agents/`
2. ✅ Install skills to `.claude/skills/`
3. ✅ Install hooks to `.claude/hooks/`
4. ✅ Install MCP server to `mcp-server/`
5. ✅ Configure `.mcp.json`
6. ✅ Create data directories
7. ✅ Validate installation

### Manual Installation

```bash
# 1. Copy plugin files
cp -r maca-debate/plugin/agents/* /path/to/project/.claude/agents/
cp -r maca-debate/plugin/skills/* /path/to/project/.claude/skills/
cp -r maca-debate/plugin/hooks/* /path/to/project/.claude/hooks/

# 2. Install MCP server
cp -r maca-debate/mcp-server /path/to/project/
cd /path/to/project/mcp-server
npm install
npm run build

# 3. Configure MCP
cp maca-debate/plugin/.mcp.json /path/to/project/.mcp.json
```

---

## ⚙️ Prerequisites

### 1. Ollama

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Download model
ollama pull qwen2.5:3b

# Start Ollama server
ollama serve
```

### 2. Node.js (for MCP server)

```bash
# Check Node version (need 18+)
node --version

# Install if needed
brew install node
```

### 3. Claude Code

Ensure you're using Claude Code with MCP support.

---

## 🚀 Quick Start

### 1. Register Agents

```bash
# Use debate-orchestrator agent
@debate-orchestrator

# Register 3 agents with qwen2.5:3b
connect_llm(id="agent_1", name="Agent Alpha", model="qwen2.5:3b")
connect_llm(id="agent_2", name="Agent Beta", model="qwen2.5:3b")
connect_llm(id="agent_3", name="Agent Gamma", model="qwen2.5:3b")
```

### 2. Run Your First Debate

```bash
/run-debate "Should a client with a 4.5% rate refinance at 6.5%?"
```

This executes:
- Round 1: Agents respond independently
- Round 2: Agents refine based on peer responses
- Consensus calculation via majority vote
- DPO pair export (if not unanimous)

### 3. Analyze Results

```bash
/analyze-consensus --summary
```

Shows:
- Consensus strength distribution
- Convergence trends (Round 1 → Round 2)
- Training signal quality
- Recommendations

### 4. Export Training Data

```bash
/export-training-data --min-consensus 0.7 --format jsonl
```

Filters and exports high-quality DPO pairs ready for fine-tuning.

---

## 📚 Plugin Components

### Agents

| Agent | Description |
|-------|-------------|
| **debate-orchestrator** | Coordinates multi-agent debates, manages rounds, calculates consensus |
| **dpo-trainer** | Handles DPO fine-tuning workflow with LoRA and HuggingFace TRL |
| **dataset-curator** | Manages dataset quality, balance, gap analysis |

### Skills

| Skill | Description |
|-------|-------------|
| **run-debate** | Execute complete debate workflow from start to export |
| **export-training-data** | Export DPO pairs with quality filtering and formatting |
| **analyze-consensus** | Analyze debate quality, convergence, dataset statistics |

### Hooks

| Hook | Description |
|------|-------------|
| **post-debate-export** | Auto-tracks export metrics, suggests next actions |

### MCP Server Tools

| Tool | Purpose |
|------|---------|
| `connect_llm` | Register Ollama agents |
| `start_debate` | Initialize debate session |
| `submit_response` | Record agent responses |
| `advance_round` | Progress to next round |
| `calculate_consensus` | Determine majority answer |
| `get_debate_history` | Retrieve all responses |
| `export_training_data` | Generate DPO pairs |
| `list_debates` | View all sessions |
| `get_debate_stats` | Analyze quality metrics |
| `get_all_training_pairs` | Get complete dataset |

---

## 💼 Use Cases

### Financial Advisory LLM (Example Domain)
Train a financial advisor chatbot with expert-level guidance:
- Refinance timing decisions
- Debt consolidation strategies
- Rate lock recommendations
- Product selection

### General Domain Training
Apply MACA to any domain:
- Legal reasoning
- Medical diagnosis
- Financial planning
- Technical support
- Code review

### Research & Experimentation
- Test MACA framework on new domains
- Experiment with debate parameters (M agents, R rounds)
- Analyze consensus patterns
- Compare to baseline models

---

## 🔬 Training Workflow

### Phase 1: Dataset Preparation (Week 1)
1. Create seed dataset (7-20 hand-crafted gold examples)
2. Generate additional examples (100-150 from domain material)
3. Add visual concepts and statistical context
4. Current: 168 seed examples ready

### Phase 2: MACA Debates (Week 2-3)
1. Select debate-worthy questions (advanced/expert difficulty)
2. Run 100-200 debates (3 agents, 2 rounds each)
3. Filter by consensus strength (keep >0.7)
4. Target: 300-500 DPO pairs

### Phase 3: DPO Fine-Tuning (Week 3-4)
1. Merge seed + debate pairs
2. Split train/val/test (70/15/15)
3. Configure LoRA + DPO (r=16, α=32, β=0.1)
4. Train for 3 epochs
5. Expected: +8-15% accuracy, +15-25% consistency

### Phase 4: Evaluation & Deployment (Week 4-6)
1. Baseline vs trained comparison
2. Statistical significance testing
3. Manual quality review
4. Export to Ollama format
5. Deploy to production

---

## 📊 Expected Results

Conservative estimates based on MACA research:

| Metric | Baseline | Post-MACA | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | 55-65% | 70-80% | +8-15 pts |
| Consistency | 40-50% | 60-75% | +15-25 pts |
| Debate Agreement | 50-60% | 70-85% | +15-25 pts |

---

## ⚙️ Configuration

### Debate Parameters

Recommended settings:

```yaml
agents: 3-5          # Balance diversity vs compute
rounds: 2            # Independent → peer feedback
temperature: 0.8     # Higher = more diverse responses
model: qwen2.5:3b    # Or llama3.2:3b
```

### Quality Thresholds

```yaml
min_consensus: 0.7   # High-quality training pairs
export_unanimous: false   # Skip 100% consensus (no signal)
convergence_required: true   # Round 2 > Round 1
```

### Training Hyperparameters

```yaml
lora_r: 16
lora_alpha: 32
dpo_beta: 0.1
learning_rate: 5e-5
batch_size: 16       # Effective (4 × 4 accumulation)
epochs: 3
```

---

## 🛠️ Troubleshooting

### Ollama Not Responding
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Low Consensus (<0.5)
- Question may be ambiguous → rephrase
- Temperature too high → reduce to 0.7
- Domain too complex → mark as expert-level

### All Debates Unanimous
- Questions too easy → use advanced/expert only
- Temperature too low → increase to 0.8-0.9
- Agents not seeing peer responses → check Round 2 prompts

### MCP Server Not Found
```bash
# Check MCP configuration
cat .mcp.json

# Rebuild MCP server
cd mcp-server && npm run build
```

---

## 📖 Documentation

- **Agents**: See `.claude/agents/*.md` for detailed instructions
- **Skills**: See `.claude/skills/*/SKILL.md` for usage guides
- **MCP Server**: See `mcp-server/README.md` for API documentation
- **Research Paper**: See `docs/concepts/01_core_framework.md`

---

## 🤝 Contributing

This plugin is part of the rob-sherman-claude-plugins marketplace.

### Reporting Issues
- GitHub: https://github.com/robsherman/claude-maca/issues

### Contributing Code
1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Based on the research paper:
**"Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment"**
*Ankur Samanta, Akshayaa Magesh, Youliang Yu, et al.*
*Meta AI, Meta Superintelligence Labs*
*arXiv:2509.15172v2, September 2025*

Developed as an open-source research implementation of the MACA framework.

---

## 📈 Roadmap

### v1.0 (Current)
- ✅ Complete plugin architecture
- ✅ 3 agents, 3 skills, 1 hook
- ✅ MCP server with 10 tools
- ✅ Automated installation

### v1.1 (Planned)
- [ ] Parallel debate execution
- [ ] Advanced convergence algorithms
- [ ] Cross-project training data sharing
- [ ] Visualization dashboards

### v2.0 (Future)
- [ ] Multi-model debates (different LLMs)
- [ ] Active learning (identify best questions)
- [ ] Automated hyperparameter tuning
- [ ] Integration with knowledge-store plugin

---

**Version**: 1.0.0
**Last Updated**: 2025-11-07
**Maintained By**: Rob Sherman (@robsherman)

**Happy Debating! 🎯**
