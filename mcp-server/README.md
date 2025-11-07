# MACA Debate MCP Server

Model Context Protocol (MCP) server implementing Multi-Agent Consensus Alignment (MACA) for generating high-quality training data through multi-agent debates.

## What is This?

This MCP server orchestrates debates between multiple LLM agents to create preference pairs for DPO (Direct Preference Optimization) training. It implements the MACA framework from the research paper:

**"Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment"**
Meta AI, 2025 (arXiv:2509.15172v2)

## How MACA Works

1. **M agents** (clones of the same base model) debate a question
2. Debate proceeds in **R rounds** (typically 2-3)
3. Round 1: Agents respond independently
4. Round 2+: Agents see peer responses and refine reasoning
5. **Consensus** is calculated via majority voting
6. **DPO pairs** are generated:
   - Majority response = "chosen"
   - Minority responses = "rejected"
7. Fine-tune model on these pairs to prefer consensus reasoning

Research shows **+27.6%** improvement on reasoning tasks after MACA training.

## Installation

```bash
cd mcp-server
npm install
npm run build
```

## Quick Start

### 1. Start the Server

```bash
npm start
```

### 2. Configure Claude Code

Add to your Claude Code MCP settings (`~/.claude/mcp_settings.json`):

```json
{
  "mcpServers": {
    "maca-debate": {
      "command": "node",
      "args": ["/path/to/maca/mcp-server/dist/index.js"]
    }
  }
}
```

### 3. Use in Claude Code

```
# Register 3 agents (all using same model for fair comparison)
connect_llm(id="agent_1", name="Agent Alpha", model="qwen2.5:3b")
connect_llm(id="agent_2", name="Agent Beta", model="qwen2.5:3b")
connect_llm(id="agent_3", name="Agent Gamma", model="qwen2.5:3b")

# Start a debate
start_debate(
  id="debate_001",
  question="Should this client refinance at 6.5% if their current rate is 4.5%?",
  agentIds=["agent_1", "agent_2", "agent_3"],
  maxRounds=2,
  metadata={category: "refinance_strategy", difficulty: "advanced"}
)

# For each agent, get a response from Ollama and submit
submit_response(
  debateId="debate_001",
  agentId="agent_1",
  reasoning="[Full LLM response]",
  answer="refinance" # or "wait" or other extracted answer
)

# After all agents respond
advance_round(debateId="debate_001")

# Calculate consensus
calculate_consensus(debateId="debate_001")

# Export to DPO format
export_training_data(debateId="debate_001", format="json")
```

## Available Tools

| Tool | Description |
|------|-------------|
| `connect_llm` | Register an Ollama agent for debates |
| `start_debate` | Initialize a new debate session |
| `submit_response` | Submit agent reasoning for current round |
| `advance_round` | Move to next round (or complete) |
| `calculate_consensus` | Determine majority via voting |
| `get_debate_history` | Retrieve all responses |
| `export_training_data` | Generate DPO training pairs |
| `list_debates` | List all debate sessions |
| `get_debate_stats` | Analyze debate quality |
| `get_all_training_pairs` | Get complete training dataset |

## Architecture

```
mcp-server/
├── src/
│   ├── index.ts                    # Main MCP server
│   ├── types/
│   │   └── debate.ts              # Type definitions
│   └── services/
│       └── debate-manager.ts      # Debate orchestration logic
├── dist/                          # Compiled JavaScript (generated)
├── package.json
├── tsconfig.json
└── README.md
```

## Educational Comments

This codebase is heavily commented with educational notes explaining:

- How MACA works conceptually
- Why each design decision was made
- How DPO training pairs are created
- Consensus calculation mechanics
- Best practices for debate parameters

Read the source code to learn! Start with:
1. `src/types/debate.ts` - Understand the data structures
2. `src/services/debate-manager.ts` - See the core logic
3. `src/index.ts` - See how tools are exposed via MCP

## Typical Workflow

### For Mortgage Training Data:

1. **Setup** (once)
   - Install Ollama
   - Download a model (e.g., `ollama pull qwen2.5:3b`)
   - Register 3-5 agents via `connect_llm`

2. **Run debates** (100+ questions)
   - Start debate with mortgage question
   - Round 1: Each agent responds independently
   - Round 2: Each agent refines based on peer feedback
   - Calculate consensus
   - Export DPO pairs

3. **Accumulate dataset**
   - After all debates, call `get_all_training_pairs`
   - Save to JSON file
   - Use with HuggingFace TRL for DPO training

4. **Fine-tune model**
   - Export Ollama model to HuggingFace format
   - Run DPO training with TRL library
   - Import back to Ollama
   - Repeat: trained model → new debates → better data → better model

## Key Parameters

### Number of Agents (M)
- **M = 3**: Fast, minimal compute, decent consensus
- **M = 5**: Good balance (recommended)
- **M = 7**: Strong consensus signals, higher compute

### Number of Rounds (R)
- **R = 1**: No peer feedback (baseline)
- **R = 2**: One round of refinement (recommended)
- **R = 3**: Diminishing returns, more compute

### Consensus Strength Filtering
- **> 0.7**: High agreement, keep for training
- **0.5 - 0.7**: Moderate agreement, review manually
- **< 0.5**: Low agreement, discard or flag as ambiguous

## Integration with Highway.ai

This MCP server generates training data for Highway.ai's mortgage advisory LLM.

**Source Material**: CMA (Certified Mortgage Advisor) Book
**Training Categories**:
- Mortgage basics
- Refinancing decisions
- Debt management strategies
- Rate lock timing
- Economic analysis

**Target Model**: qwen2.5:3b or llama3.2:3b
**Training Method**: DPO (Direct Preference Optimization) with LoRA
**Deployment**: Ollama server → Highway.ai SaaS API

## Research Citation

```bibtex
@article{samanta2025maca,
  title={Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment},
  author={Samanta, Ankur and Magesh, Akshayaa and Yu, Youliang and others},
  journal={arXiv preprint arXiv:2509.15172},
  year={2025}
}
```

## License

MIT License - See project root LICENSE file

## Development

```bash
# Build
npm run build

# Watch mode
npm run dev

# Run with inspector (debugging)
npm run inspect
```

## Troubleshooting

### Error: "Agent X not found"
→ Register agents with `connect_llm` before starting debates

### Error: "Cannot advance round - waiting for N responses"
→ All agents must submit responses before advancing

### Warning: "Low consensus strength (<0.5)"
→ Question may be ambiguous, consider rephrasing or discarding

### No training pairs exported
→ Ensure you called `calculate_consensus` and `export_training_data` after completing debate

## Next Steps

1. Run sample debates with mortgage questions
2. Analyze consensus patterns and convergence
3. Generate 100-200 DPO pairs for training dataset
4. Fine-tune qwen2.5:3b with DPO
5. Evaluate improvement on test set
6. Deploy to Highway.ai production

---

For questions about MACA or this implementation, see:
- `docs/concepts/01_core_framework.md` - MACA research summary
- `docs/concepts/02_claude_code_implementation_plan.md` - Implementation plan
- `LEARNING.md` - ML training guide for beginners
