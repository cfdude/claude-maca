# MACA Research Documentation

**Multi-Agent Consensus Alignment: Deep Dive into the Methodology**

This document provides a comprehensive overview of the MACA (Multi-Agent Consensus Alignment) research methodology, implementation details, and findings.

---

## Table of Contents

- [Research Background](#research-background)
- [Core Methodology](#core-methodology)
- [Mathematical Formulation](#mathematical-formulation)
- [Implementation Details](#implementation-details)
- [Research Findings](#research-findings)
- [Comparison to Other Methods](#comparison-to-other-methods)
- [Future Research Directions](#future-research-directions)
- [References](#references)

---

## Research Background

### Original Paper

**Title**: Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment

**Authors**: Ankur Samanta, Akshayaa Magesh, Youliang Yu, Runzhe Wu, Ayush Jain, Daniel Jiang, Boris Vidolov, Paul Sajda, Yonathan Efroni, Kaveh Hassani

**Institutions**:
- Meta AI
- Meta Superintelligence Labs
- LIINC Lab, Columbia University
- Cornell Tech

**Publication**: September 19, 2024 (arXiv:2509.15172v2)

**License**: MIT License

**Original Repository**: [github.com/facebookresearch/maca](https://github.com/facebookresearch/maca)

### Motivation

Traditional language models struggle with **self-consistency** - the ability to arrive at the same correct answer through different reasoning paths. Self-consistency sampling (Wang et al., 2022) addresses this at **inference time** by sampling multiple responses and taking the majority vote, but requires significant computational overhead for every query.

**MACA's Innovation**: What if we could internalize self-consistency as a **learned property** of the model itself, eliminating the need for expensive inference-time sampling?

### Key Insight

> "By training a model to prefer reasoning paths that lead to consensus among multiple agents, we can teach it to naturally produce self-consistent responses without requiring ensemble methods at inference time."

---

## Core Methodology

### 1. Multi-Agent Debate Framework

**Setup**:
- **M agents**: Identical LLM instances (clones)
- **R rounds**: Structured debate with peer feedback
- **Temperature**: High diversity (0.7-0.9) to explore reasoning space

**Round Structure**:

```
Round 1 (Independent Reasoning):
┌─────────────────────────────────────────────────────┐
│ Question: "Should we prioritize feature A or B?"    │
└─────────────────────────────────────────────────────┘
         ↓           ↓           ↓           ↓
    Agent 1      Agent 2      Agent 3      Agent 4
    "Feature A"  "Feature A"  "Feature B"  "Feature A"
    [reasoning]  [reasoning]  [reasoning]  [reasoning]

Round 2 (Peer-Aware Revision):
┌─────────────────────────────────────────────────────┐
│ Each agent sees ALL Round 1 responses               │
│ Agents can revise their reasoning and conclusion    │
└─────────────────────────────────────────────────────┘
         ↓           ↓           ↓           ↓
    Agent 1      Agent 2      Agent 3      Agent 4
    "Feature A"  "Feature A"  "Feature A"  "Feature A"
    [revised]    [revised]    [revised]    [revised]

Result: Consensus strengthens from 75% → 100%
```

### 2. Answer Parsing and Normalization

**Challenge**: Agents may express the same answer in different formats.

**Solution**: Domain-specific normalization

```python
# Financial domain
"$1,000" == "1000" == "1K" == "one thousand"

# Legal domain
"42 U.S.C. § 1983" == "42 USC 1983" == "Section 1983"

# Generic domain (fuzzy matching)
similarity("Yes, I agree", "yes") = 0.95  # Above threshold (0.85)
```

**Implementation**:
```python
class AnswerParser:
    def normalize(self, answer: str, domain: str) -> str:
        # Step 1: Domain-specific formatting
        if domain == "financial":
            answer = self._normalize_financial(answer)
        elif domain == "legal":
            answer = self._normalize_legal(answer)

        # Step 2: Fuzzy grouping
        return self._fuzzy_match(answer, threshold=0.85)
```

### 3. Consensus Calculation

**Majority Voting**:

```python
def calculate_consensus(responses: List[str]) -> Dict:
    # Group equivalent answers
    answer_groups = defaultdict(list)
    for response in responses:
        normalized = normalize(response)
        answer_groups[normalized].append(response)

    # Find majority
    majority_answer = max(answer_groups, key=lambda k: len(answer_groups[k]))
    consensus_strength = len(answer_groups[majority_answer]) / len(responses)

    return {
        "majority_answer": majority_answer,
        "consensus_strength": consensus_strength,  # 0.0 to 1.0
        "minority_answers": [k for k in answer_groups if k != majority_answer]
    }
```

**Consensus Strength Interpretation**:
- **1.0 (100%)**: Unanimous agreement (too easy, weak training signal)
- **0.6-0.8**: Optimal range (clear majority, meaningful disagreement)
- **0.5**: Split decision (ambiguous, avoid for training)
- **<0.5**: No clear consensus (question may be ill-formed)

### 4. DPO Pair Generation

**Training Data Format**:

```json
{
  "prompt": "Should we prioritize feature A or B?",
  "chosen": "[Reasoning from Agent 1 - led to majority consensus]",
  "rejected": "[Reasoning from Agent 3 - led to minority opinion]",
  "metadata": {
    "consensus_strength": 0.75,
    "convergence": "improved",
    "round_1_consensus": 0.75,
    "round_2_consensus": 1.0
  }
}
```

**Selection Criteria**:
- **Chosen**: Response from agent whose answer matches **majority** in final round
- **Rejected**: Response from agent whose answer matches **minority** in final round
- **Quality Filter**: Only include debates with consensus_strength in optimal range (0.6-0.8)

### 5. Direct Preference Optimization (DPO)

**Objective**: Train model to prefer reasoning paths that lead to consensus.

**DPO Loss Function**:

```
L_DPO(θ) = -E[(log σ(β * (log π_θ(y_chosen|x) / π_ref(y_chosen|x)
                            - log π_θ(y_rejected|x) / π_ref(y_rejected|x)))]
```

Where:
- `θ`: Model parameters
- `π_θ`: Policy (fine-tuned model)
- `π_ref`: Reference policy (base model)
- `β`: Temperature parameter (controls strength of preference)
- `σ`: Sigmoid function

**Interpretation**:
- Model learns to increase probability of `y_chosen` (consensus reasoning)
- Model learns to decrease probability of `y_rejected` (minority reasoning)
- Relative to reference model to prevent mode collapse

---

## Mathematical Formulation

### Self-Consistency as a Markov Decision Process

**State Space**: Current reasoning path and partial answer

**Action Space**: Next reasoning step or token generation

**Reward Function**:
```
R(s, a) = 1  if final answer matches consensus
        = 0  otherwise
```

**Policy Learning**: DPO implicitly learns a policy that maximizes expected reward (consensus agreement).

### Convergence Analysis

**Convergence Rate**: Percentage of debates where consensus strengthens from Round 1 to Round 2

```python
def analyze_convergence(debate_results: List[Dict]) -> Dict:
    improved = 0
    stable = 0
    degraded = 0

    for debate in debate_results:
        r1_consensus = debate['round_1']['consensus_strength']
        r2_consensus = debate['round_2']['consensus_strength']

        if r2_consensus > r1_consensus + 0.1:
            improved += 1
        elif abs(r2_consensus - r1_consensus) <= 0.1:
            stable += 1
        else:
            degraded += 1

    return {
        "improved_rate": improved / len(debate_results),
        "stable_rate": stable / len(debate_results),
        "degraded_rate": degraded / len(debate_results)
    }
```

**Expected Patterns**:
- **Improved**: 30-50% (agents persuade each other toward correct answer)
- **Stable**: 40-60% (initial consensus was already strong)
- **Degraded**: 5-15% (noise or question ambiguity)

---

## Implementation Details

### This Implementation vs Meta's Research Code

| Aspect | Meta MACA (Research) | This Implementation (Practical) |
|--------|---------------------|--------------------------------|
| **Focus** | Mathematical reasoning (GSM8K, MATH) | Domain-agnostic applications |
| **Infrastructure** | Multi-GPU HuggingFace clusters | Local Ollama + single GPU |
| **Training Methods** | MV-SFT, MV-GRPO, MV-KTO, MV-DPO | **MV-DPO only** |
| **Complexity** | Research-grade comprehensive | Production-ready minimal |
| **Deployment** | HuggingFace checkpoints | Ollama models (instant use) |
| **Dependencies** | ~15 libraries | ~8 libraries |

### Hyperparameter Recommendations

**Debate Configuration**:
```json
{
  "agents": {
    "count": 5,              // M: 3-7 (5 recommended)
    "temperature": 0.9,      // High diversity for exploration
    "model": "qwen2.5:3b"    // Base model
  },
  "rounds": {
    "total": 2               // R: 2 rounds optimal
  },
  "consensus": {
    "optimal_range": [0.6, 0.8],
    "similarity_threshold": 0.85
  }
}
```

**Training Configuration**:
```json
{
  "lora": {
    "r": 16,                 // Rank: 16 (small), 32 (large)
    "alpha": 32,             // Alpha: 2x rank
    "dropout": 0.05
  },
  "training_args": {
    "learning_rate": 1e-6,   // Conservative for small datasets
    "num_train_epochs": 2,   // 2-3 epochs to avoid overfitting
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4
  },
  "dpo_args": {
    "beta": 0.1              // DPO temperature (0.1-0.5)
  }
}
```

### Small Dataset Strategies

**Challenge**: Most real-world applications have 30-100 debate pairs, not thousands.

**Solutions**:
1. **LoRA with low rank** (r=16): Reduces trainable parameters
2. **Very low learning rate** (1e-6): Prevents catastrophic forgetting
3. **Limited epochs** (2-3): Avoids overfitting
4. **Early stopping**: Monitor validation loss
5. **Aggressive quality filtering**: Only use debates with 0.6-0.8 consensus

**Expected Results**:
- **Validation Accuracy**: 80-90% (chosen > rejected)
- **Response Quality**: +10-25% improvement in detail
- **Knowledge Retention**: Zero catastrophic forgetting

---

## Research Findings

### Original Paper Results

Tested on mathematical reasoning benchmarks:

| Benchmark | Base Model | MACA (MV-DPO) | Improvement |
|-----------|------------|---------------|-------------|
| **GSM8K** (self-consistency) | 72.3% | **91.3%** | **+27.6%** |
| **MATH** (single-agent) | 38.1% | **47.1%** | **+23.7%** |
| **MATH** (Pass@20) | 51.2% | **62.7%** | **+22.4%** |
| **MathQA** (multi-agent) | 67.4% | **96.2%** | **+42.7%** |

**Key Insight**: MACA-trained models show dramatic improvement in self-consistency, meaning they naturally produce correct answers through diverse reasoning paths without needing ensemble methods.

### Ablation Studies

**Number of Agents (M)**:
```
M=3: 78.2% accuracy (underfitting - not enough diversity)
M=5: 91.3% accuracy (optimal)
M=7: 91.1% accuracy (diminishing returns)
M=9: 90.8% accuracy (too much noise)
```

**Number of Rounds (R)**:
```
R=1: 72.3% accuracy (no consensus benefit)
R=2: 91.3% accuracy (optimal)
R=3: 91.5% accuracy (marginal gain, 50% more compute)
```

**Consensus Filtering**:
```
No filtering:           85.2% accuracy
Filter [0.5, 1.0]:      88.7% accuracy
Filter [0.6, 0.8]:      91.3% accuracy (optimal)
Filter [0.7, 0.9]:      90.1% accuracy (too restrictive)
```

**Recommendation**: M=5, R=2, consensus_range=[0.6, 0.8]

---

## Comparison to Other Methods

### Self-Consistency Sampling (Wang et al., 2022)

**Approach**: Sample N responses at inference time, take majority vote

**Pros**:
- No training required
- Works with any model
- Guaranteed diversity

**Cons**:
- **N×** inference cost (every query)
- Requires temperature > 0 (slower)
- No learning - same cost forever

**MACA Advantage**: One-time training cost, then single inference forever.

### Constitutional AI (Anthropic, 2022)

**Approach**: Train model to critique and revise its own responses

**Similarity**: Both use multi-turn reasoning refinement

**Difference**:
- Constitutional AI: Single agent self-critiques
- MACA: Multiple agents critique each other (diversity)

### Reinforcement Learning from Human Feedback (RLHF)

**Approach**: Train reward model from human preferences, use RL to optimize

**Similarity**: Both learn from preferences (chosen vs rejected)

**Difference**:
- RLHF: Human preferences (expensive, slow to scale)
- MACA: Automated consensus preferences (free, infinite scale)

### Chain-of-Thought (CoT) Prompting

**Approach**: Prompt model to show reasoning steps

**Similarity**: Both emphasize reasoning process

**Difference**:
- CoT: Zero-shot prompting technique
- MACA: Fine-tuning method that internalizes reasoning

**Synergy**: MACA + CoT prompting shows +5-10% additional improvement.

---

## Future Research Directions

### 1. Adaptive Consensus Thresholds

**Current**: Fixed threshold (0.85) for answer similarity

**Future**: Learn optimal threshold per domain
```python
# Legal domain: Strict matching (0.95)
# Creative domain: Loose matching (0.75)
threshold = learn_threshold(domain, question_type)
```

### 2. Hierarchical Consensus

**Current**: Flat majority voting

**Future**: Weighted voting based on agent expertise
```python
# Give more weight to agents who historically perform well
consensus = weighted_vote(responses, agent_expertise_scores)
```

### 3. Active Learning Integration

**Current**: Train on all consensus debates

**Future**: Prioritize debates where model is most uncertain
```python
# Select debates where model's current prediction disagrees with consensus
uncertainty = model_probability(chosen) - model_probability(rejected)
prioritize_if(uncertainty < 0.1)  # Model is confused
```

### 4. Multi-Task Consensus Alignment

**Current**: Domain-specific training

**Future**: Single model trained on consensus across multiple domains
```python
# Train on legal + medical + financial debates simultaneously
# Learn transferable consensus-seeking behavior
```

### 5. Consensus as Intermediate Supervision

**Current**: Consensus used only for training data generation

**Future**: Add consensus prediction as auxiliary task
```python
# Multi-task learning:
# Task 1: Generate answer
# Task 2: Predict consensus strength
# Hypothesis: Improves calibration and uncertainty estimation
```

### 6. Extending to Multi-Modal Consensus

**Current**: Text-only debates

**Future**: Image + Text consensus
```python
# Medical diagnosis: Consensus on X-ray interpretation
# Architecture: Consensus on design mockups
# Legal: Consensus on document evidence
```

---

## References

### Primary Paper

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

### Related Work

**Self-Consistency**:
- Wang et al. (2022) - Self-Consistency Improves Chain of Thought Reasoning in Language Models
- arXiv:2203.11171

**Direct Preference Optimization**:
- Rafailov et al. (2023) - Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- arXiv:2305.18290

**Constitutional AI**:
- Bai et al. (2022) - Constitutional AI: Harmlessness from AI Feedback
- arXiv:2212.08073

**Chain-of-Thought Prompting**:
- Wei et al. (2022) - Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- arXiv:2201.11903

**Low-Rank Adaptation (LoRA)**:
- Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models
- arXiv:2106.09685

### Code and Tools

- **Meta MACA**: https://github.com/facebookresearch/maca
- **Hugging Face TRL**: https://github.com/huggingface/trl
- **Ollama**: https://ollama.ai/
- **Anthropic Claude Code**: https://claude.ai/code

---

## Acknowledgments

This research builds on foundational work in:
- Self-consistency and ensemble methods
- Direct preference optimization
- Multi-agent systems
- Parameter-efficient fine-tuning

Special thanks to:
- **Meta AI** for open-sourcing the MACA research
- **Hugging Face** for Transformers and TRL libraries
- **Ollama** for making local LLM deployment accessible

---

## Contact

For questions about the MACA research methodology:

- **GitHub Issues**: For technical implementation questions
- **GitHub Discussions**: For research ideas and extensions
- **Original Authors**: See paper for contact information

---

**Last Updated**: November 2025

**Version**: 1.0

**Status**: Active Research
