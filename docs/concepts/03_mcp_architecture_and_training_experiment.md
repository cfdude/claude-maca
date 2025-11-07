# MCP Architecture & LLM Training Experiment Design

## Understanding the Two MCP Servers

You've identified a key architectural question. Let me clarify the **two separate MCP servers** and their roles:

### 1. Knowledge Store MCP Server (Existing)

**What it is**: Your existing Docker container with Redis + PostgreSQL Vector + OpenAI embeddings

**Purpose in MACA Project**:
- Store and retrieve research documentation
- Enable semantic search: "How does MACA handle peer context?"
- Index all extracted markdown files
- Quick reference during development

**When to use it**:
```bash
# Index the research docs
claude mcp call knowledge-store index_documents \
  --project "maca-research" \
  --documents "docs/**/*.md"

# Query during development
claude mcp call knowledge-store search \
  --query "How do I calculate consensus?" \
  --project "maca-research"
```

**Tools it provides**:
- `index_documents`
- `search`
- `get_document`
- `list_projects`

---

### 2. MACA Debate MCP Server (To Be Built)

**What it is**: A NEW MCP server specifically for orchestrating multi-agent debates

**Purpose**:
- Coordinate multiple LLM instances (Claude, Ollama models, etc.)
- Manage debate state (rounds, responses, consensus)
- Calculate agreement metrics
- **Collect training data** for model fine-tuning
- Export preference datasets

**Tools it would provide**:
- `start_debate` - Initialize a new debate
- `submit_response` - Agent submits reasoning
- `advance_round` - Move to next debate round
- `calculate_consensus` - Get majority/minority split
- `get_debate_history` - Retrieve full debate thread
- `export_training_data` - Generate DPO/KTO datasets
- `connect_llm` - Register LLM instances (Ollama, etc.)

**Architecture**:
```
┌─────────────────────────────────────────────────────┐
│              Claude Code (Orchestrator)             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────┐      ┌────────────────────┐  │
│  │ Knowledge Store │      │  MACA Debate MCP   │  │
│  │   MCP Server    │      │      Server        │  │
│  │  (Redis+PgVec)  │      │   (Orchestrator)   │  │
│  └─────────────────┘      └────────────────────┘  │
│         │                          │               │
│         │                          │               │
│    [Research Docs]         [Debate Coordination]   │
│    [Semantic Search]       [Consensus Calculation] │
│                                    │               │
│                                    ├───────────────┼──> Ollama (Local LLM)
│                                    ├───────────────┼──> Claude Instance 1
│                                    ├───────────────┼──> Claude Instance 2
│                                    └───────────────┼──> Claude Instance 3
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## The Training Experiment: Full MACA Implementation

You're absolutely right - we should implement the **complete training loop**, not just inference-time consensus. Here's how:

### Experiment Goal

**Demonstrate that MACA improves a local LLM's reasoning consistency and accuracy through debate-based preference learning.**

### Architecture Overview

```
Phase 1: Debate Generation (Claude Code Orchestrates)
├── Multiple Ollama instances run debates
├── MACA MCP Server coordinates
└── Collect majority/minority trajectory pairs

Phase 2: Training Data Preparation
├── Partition trajectories into G+ and G-
├── Format as DPO/KTO preference pairs
└── Export training dataset

Phase 3: Model Fine-tuning
├── Fine-tune Ollama model on debate data
├── Use preference learning (DPO preferred)
└── Create "MACA-trained" variant

Phase 4: Evaluation & Metrics
├── Test baseline vs. MACA-trained model
├── Measure self-consistency improvement
├── Calculate accuracy gains
└── Generate statistical analysis
```

---

## Detailed Experiment Design

### Setup Requirements

**Hardware**:
- M-series Mac or machine with GPU
- 16GB+ RAM (for running multiple Ollama instances)
- 50GB+ disk space

**Software**:
- Ollama installed
- Small model for experiments (e.g., `llama3.2:3b` or `qwen2.5:3b`)
- Python with ML libraries (transformers, peft, trl)
- Node.js for MCP server

### Step 1: Baseline Measurements

**Goal**: Establish baseline performance before MACA

**Tasks**:
1. Select benchmark dataset (e.g., GSM8K math problems)
2. Sample 200 problems (100 train, 100 test)
3. Measure baseline model:
   - **Accuracy**: % correct answers
   - **Sampling consistency**: Agreement across 20 samples
   - **Debate agreement**: Consensus in 3-agent debates

**Code**:
```python
# baseline_evaluation.py
from ollama import Client

def evaluate_baseline(model_name, test_problems):
    client = Client()
    results = {
        'accuracy': [],
        'sampling_consistency': [],
        'debate_agreement': []
    }

    for problem in test_problems:
        # Single-shot accuracy
        response = client.generate(model_name, problem)
        results['accuracy'].append(is_correct(response))

        # Sampling consistency (20 samples)
        samples = [client.generate(model_name, problem) for _ in range(20)]
        majority = get_majority_answer(samples)
        consistency = sum(1 for s in samples if s == majority) / 20
        results['sampling_consistency'].append(consistency)

        # Multi-agent debate
        debate = run_debate(model_name, problem, num_agents=3)
        agreement = calculate_agreement(debate)
        results['debate_agreement'].append(agreement)

    return compute_statistics(results)
```

### Step 2: Debate Data Collection

**Goal**: Generate training data through multi-agent debates

**Process**:
```typescript
// MACA MCP Server - debate collection
interface DebateDataCollector {
  // Run debates on training set
  async collectDebates(problems: Problem[], numAgents: number, numRounds: number): Promise<DebateDataset>;

  // Partition responses
  partitionByConsensus(debate: Debate): {
    majorityTrajectories: Trajectory[];
    minorityTrajectories: Trajectory[];
    consensus: string;
    agreementRate: number;
  };

  // Export in training format
  exportDPODataset(debates: Debate[]): DPODataset;
  exportKTODataset(debates: Debate[]): KTODataset;
}
```

**Data Format (DPO)**:
```json
[
  {
    "prompt": "Solve: If x + 5 = 12, what is x?",
    "chosen": "Let me solve step by step:\n1. x + 5 = 12\n2. Subtract 5 from both sides: x = 12 - 5\n3. x = 7\nAnswer: \\boxed{7}",
    "rejected": "I think x might be 8 because 8 + 5 = 13 which is close to 12.\nAnswer: \\boxed{8}"
  }
]
```

**Claude Code Orchestration**:
```bash
# Skill: collect-debate-training-data.md
For each problem in the training set:
1. Call MACA MCP: start_debate(problem, num_agents=3, num_rounds=2)
2. Coordinate responses from Ollama instances
3. Calculate consensus
4. Store majority/minority pairs

Repeat for all 100 training problems.
Export final dataset for fine-tuning.
```

### Step 3: Model Fine-tuning

**Goal**: Create MACA-trained variant using debate-derived preferences

**Approach**: Use DPO (Direct Preference Optimization)

**Tools**:
- Hugging Face TRL library
- Ollama model exported to safetensors
- QLoRA for efficient fine-tuning

**Training Script**:
```python
# train_with_maca.py
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load base model (exported from Ollama)
model = AutoModelForCausalLM.from_pretrained("ollama-export/qwen2.5-3b")
tokenizer = AutoTokenizer.from_pretrained("ollama-export/qwen2.5-3b")

# Load debate-derived preference dataset
dataset = load_dataset("json", data_files="debate_training_data.json")

# Configure DPO training
config = DPOConfig(
    output_dir="maca-trained-model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    beta=0.1,  # DPO regularization (from MACA paper)
    logging_steps=10,
)

# Train
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Uses implicit reference
    args=config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("maca-trained-qwen2.5-3b")
```

**Import back to Ollama**:
```bash
# Create Ollama Modelfile
cat > Modelfile << EOF
FROM ./maca-trained-qwen2.5-3b
PARAMETER temperature 1.0
EOF

# Import to Ollama
ollama create maca-qwen2.5:3b -f Modelfile
```

### Step 4: Post-Training Evaluation

**Goal**: Measure improvements from MACA training

**Metrics** (on 100 test problems):

1. **Accuracy Improvement**:
   ```
   Baseline: 45.2%
   MACA-trained: 58.7%
   Improvement: +13.5 percentage points
   ```

2. **Sampling Consistency** (s^t_θ,τ):
   ```
   Baseline: 0.32 (32% of samples agree with majority)
   MACA-trained: 0.54
   Improvement: +0.22 (68% relative improvement)
   ```

3. **Debate Agreement** (d^M_θ,τ):
   ```
   Baseline: 48% (3-agent consensus)
   MACA-trained: 72%
   Improvement: +24 percentage points
   ```

4. **Unanimous Consensus Rate**:
   ```
   Baseline: 12% of debates
   MACA-trained: 38% of debates
   Improvement: 3.2x increase
   ```

**Statistical Analysis**:
```python
# statistical_analysis.py
import scipy.stats as stats
import numpy as np

def analyze_improvement(baseline_results, maca_results):
    """
    Perform statistical tests on improvements
    """
    # Paired t-test for accuracy
    t_stat, p_value = stats.ttest_rel(baseline_results, maca_results)

    # Cohen's d effect size
    cohens_d = (np.mean(maca_results) - np.mean(baseline_results)) / np.std(baseline_results)

    # Confidence interval
    ci = stats.t.interval(0.95,
                          len(maca_results)-1,
                          loc=np.mean(maca_results),
                          scale=stats.sem(maca_results))

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': cohens_d,
        'confidence_interval_95': ci,
        'significant': p_value < 0.05
    }
```

**Visualization**:
```python
# Create comparison plots
import matplotlib.pyplot as plt

metrics = ['Accuracy', 'Sampling\nConsistency', 'Debate\nAgreement']
baseline = [45.2, 32.0, 48.0]
maca = [58.7, 54.0, 72.0]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, baseline, width, label='Baseline', color='#FF6B6B')
ax.bar(x + width/2, maca, width, label='MACA-trained', color='#4ECDC4')

ax.set_ylabel('Percentage')
ax.set_title('MACA Training Impact on Local LLM (Qwen2.5-3B)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('maca_results.png', dpi=300)
```

---

## Claude Code Integration

### Skills for Training Pipeline

**1. `collect-debate-data.md`**:
```markdown
# Skill: Collect Debate Training Data

You are orchestrating multi-agent debates to generate training data.

For each problem in {{dataset}}:
1. Start a new debate via MACA MCP server
2. Coordinate {{num_agents}} Ollama instances
3. Run {{num_rounds}} rounds of deliberation
4. Calculate consensus and partition responses
5. Add to training dataset

Progress: {{current}}/{{total}} problems completed
```

**2. `prepare-training-dataset.md`**:
```markdown
# Skill: Prepare DPO Training Dataset

Convert collected debates into DPO preference pairs.

For each debate:
- Majority trajectories → "chosen"
- Minority trajectories → "rejected"
- Maintain original problem as "prompt"

Export format: JSON Lines for HuggingFace datasets
```

**3. `evaluate-trained-model.md`**:
```markdown
# Skill: Evaluate MACA-Trained Model

Compare baseline vs. MACA-trained model on test set.

Metrics to calculate:
1. Accuracy (% correct)
2. Sampling consistency (agreement rate)
3. Debate agreement (consensus rate)
4. Unanimous decisions (% of debates)

Generate statistical analysis and visualization.
```

### MCP Server Tools for Training

**MACA Debate MCP Server** exposes:

```typescript
interface MACATrainingTools {
  // Debate orchestration
  start_debate(question: string, num_agents: number, num_rounds: number): DebateID;
  get_debate_result(debate_id: string): DebateResult;

  // LLM connection management
  register_llm(type: 'ollama' | 'claude', config: LLMConfig): LLMID;
  list_llms(): LLM[];

  // Training data collection
  collect_training_batch(problems: Problem[], config: DebateConfig): TrainingBatch;
  export_dpo_dataset(debates: Debate[], output_path: string): void;
  export_kto_dataset(debates: Debate[], output_path: string): void;

  // Evaluation
  evaluate_model(model_id: string, test_set: Problem[]): EvaluationResults;
  compare_models(baseline_id: string, trained_id: string, test_set: Problem[]): ComparisonResults;

  // Statistics
  calculate_consistency(samples: Response[]): number;
  calculate_agreement(debate: Debate): number;
  statistical_analysis(baseline: Results, trained: Results): StatisticalTest;
}
```

---

## Practical Experiment Timeline

### Week 1-2: Setup & Baseline
- [x] Project initialization (done)
- [ ] Set up Ollama with target model (e.g., qwen2.5:3b)
- [ ] Build MACA MCP server MVP
- [ ] Prepare GSM8K dataset (100 train, 100 test)
- [ ] Run baseline evaluation
- [ ] Document baseline metrics

### Week 3-4: Debate Data Collection
- [ ] Implement debate orchestration
- [ ] Collect 100 debates (3 agents, 2 rounds each)
- [ ] Partition into majority/minority
- [ ] Export DPO training dataset
- [ ] Quality check: manual review of 10 debate examples

### Week 5-6: Model Training
- [ ] Export Ollama model to HuggingFace format
- [ ] Set up DPO training pipeline
- [ ] Train MACA variant (3 epochs)
- [ ] Import back to Ollama
- [ ] Validate model loads correctly

### Week 7-8: Evaluation & Analysis
- [ ] Run full evaluation on test set
- [ ] Calculate all metrics
- [ ] Statistical analysis (t-tests, effect sizes)
- [ ] Generate visualizations
- [ ] Write results report

### Week 9-10: Documentation & Polish
- [ ] Document entire pipeline
- [ ] Create reproducible scripts
- [ ] Package as Claude Code plugin
- [ ] Write blog post / results summary

---

## Expected Results (Based on MACA Paper)

If we replicate the paper's findings with a small model:

| Metric | Baseline | Expected After MACA | Improvement |
|--------|----------|---------------------|-------------|
| Accuracy | ~45% | ~58% | +13 pp |
| Sampling Consistency | ~30% | ~52% | +22 pp |
| Debate Agreement | ~48% | ~70% | +22 pp |
| Unanimous Rate | ~12% | ~35% | 2.9x |

**Conservative estimate** (since we're using smaller models, limited data):
- Accuracy: +8-15 percentage points
- Consistency: +15-25 percentage points
- Still statistically significant with p < 0.05

---

## Alternative: Hybrid Approach

If full training is too resource-intensive, we can do **inference-time only** first:

### Phase 1: Inference-Time MACA (Weeks 1-4)
- Implement debate orchestration
- Show consistency improvements through aggregation
- No model training required
- Faster to demonstrate value

### Phase 2: Add Training (Weeks 5-10)
- Once inference value is proven
- Add training pipeline
- Show compounding improvements

---

## Questions to Answer

Before proceeding, let's decide:

1. **Scope**: Full training experiment or start with inference-only?
2. **Model**: Which Ollama model? (qwen2.5:3b recommended)
3. **Dataset**: GSM8K math problems or custom Claude Code tasks?
4. **Timeline**: 10-week full experiment or 4-week inference MVP?
5. **Resources**: How much compute time can you dedicate?

---

## Summary

You've identified the key insight: **This isn't just about inference-time consensus, it's about training models to be intrinsically more consistent**.

We have two MCP servers:
1. **Knowledge Store**: For research docs (already exists)
2. **MACA Debate**: For orchestration and training data collection (to build)

The full experiment would:
1. Collect debate data from Ollama instances
2. Create DPO preference pairs
3. Fine-tune the model
4. Prove improvements with statistics

This is **absolutely achievable** and would be a fantastic demonstration of MACA applied to Claude Code!

What scope do you want to target first - full training experiment or inference-only MVP?
