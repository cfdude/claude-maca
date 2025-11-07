# MACA: Multi-Agent Consensus Alignment - Core Framework

## Abstract Summary

Language Models (LMs) are inconsistent reasoners, often generating contradictory responses to identical prompts. MACA addresses this by formalizing self-consistency as an intrinsic property of well-aligned reasoning models.

### Key Innovation
Instead of inference-time methods, MACA uses reinforcement learning to post-train models to favor reasoning trajectories aligned with their internal consensus through multi-agent debate.

### Performance Improvements
- **Self-consistency**: +27.6% on GSM8K
- **Single-agent reasoning**: +23.7% on MATH
- **Sampling-based inference**: +22.4% Pass@20 on MATH
- **Multi-agent decision-making**: +42.7% on MathQA
- **Generalization**: +16.3% on GPQA, +11.6% on CommonsenseQA

---

## 1. Problem Statement

### Current Limitations
**Probabilistic decoding** gives access to diverse reasoning trajectories but struggles to consistently select high-quality paths.

**Existing solutions** (inference-time):
- Self-consistency prompting: Sample multiple paths, majority vote
- Multi-agent debate: Consensus across models
- **Problem**: Don't improve model's internal reasoning stability

### The Challenge
Teaching models to:
- Sample **diversely** (multiple valid reasoning paths)
- Maintain **consistent quality** and conclusions
- Avoid low-quality reasoning traces that compound during aggregation

---

## 2. Self-Consistency Formalization

### Mathematical Definition

Given prompt $x$, an LM with parameters $\theta$ defines distribution:

$$\pi_\theta(y | x) = \prod_{t=1}^{|y|} \pi_\theta(y_t | x, y_{<t})$$

Under temperature sampling $\tau > 0$, this induces answer distribution:

$$P_{\theta,\tau}(a | x) = \sum_{y:A(y)=a} \pi_{\theta,\tau}(y | x)$$

Where $A(y)$ extracts the answer from reasoning trajectory $y$.

### Majority Answer

$$a_{\theta,\tau}^{\star}(x) = \arg\max_a P_{\theta,\tau}(a | x)$$

### Majority Probability (Internal Consensus)

$$S_{\theta,\tau}^{+}(x) = P_{\theta,\tau}(a_{\theta,\tau}^{\star}(x) | x)$$

This represents the total probability mass on the most likely answer.

### Goal
A self-consistent model maintains high $S_{\theta,\tau}^{+}(x)$ even at high temperatures, allowing diverse reasoning while reliably converging on consistent answers.

---

## 3. Measuring Self-Consistency

### Single-Agent Sampling Consistency

Estimate $S_{\theta,\tau}^{+}(x)$ by sampling $t$ independent trajectories:

$$s_{\theta,\tau}^{t}(x) = \frac{1}{t} \sum_{i=1}^{t} \mathbb{1}[a_i(x) = \hat{a}(x)]$$

Where $\hat{a}(x) = \text{Majority}\{a_1(x), \ldots, a_t(x)\}$

As $t \to \infty$, $s_{\theta,\tau}^{t}(x) \to S_{\theta,\tau}^{+}(x)$

### Multi-Agent Debate Agreement

When $M$ agents produce answers through deliberation:

$$d_{\theta,\tau}^{M}(x) = \frac{1}{M} \sum_{m=1}^{M} \mathbb{1}[a_m(x) = \hat{a}(x)]$$

Higher agreement indicates stronger consensus.

---

## 4. MACA Framework Architecture

### Multi-Agent Debate Setup

**Participants**: $M$ copies of the same model
**Rounds**: $R$ iterations of deliberation

#### Process:
1. **Initial Round**: Each agent generates independent response
2. **Subsequent Rounds** ($r = 2, \ldots, R$):
   - Agents see each other's reasoning
   - Update their answers
   - Answers that persist indicate stronger reasoning

#### Debate Output
For each prompt $x$:
- Final responses: $\mathcal{Y}(x) = \{y_1, \ldots, y_M\}$
- Extracted answers: $a_m = A(y_m)$
- Majority consensus: $\hat{a}(x) = \text{Majority}\{a_1, \ldots, a_M\}$

### Trajectory Partitioning

**Consensus-supporting trajectories**:
$$\mathcal{G}^{+}(x) = \{y \in \mathcal{Y}(x) : A(y) = \hat{a}(x)\}$$

**Dissenting trajectories**:
$$\mathcal{G}^{-}(x) = \{y \in \mathcal{Y}(x) : A(y) \neq \hat{a}(x)\}$$

### Training Dataset
$$\mathcal{D}_{\text{post}} = \{(x, \hat{a}(x), \mathcal{G}^{+}(x), \mathcal{G}^{-}(x))\}_{x \in \mathcal{D}}$$

**Key insight**: Debate consensus from deliberative exchange provides richer signals than statistical sampling.

---

## 5. Training Objectives

All methods treat $\mathcal{G}^{+}$ as preferred and $\mathcal{G}^{-}$ as not preferred.

### 5.1 Majority-Vote SFT (MV-SFT)

Train model to mimic consensus-supporting trajectories:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y^{+} \in \mathcal{G}^{+}(x)} [\log \pi_\theta(y^{+}|x)]$$

### 5.2 Majority-Vote GRPO (MV-GRPO)

Online sampling with consensus-based rewards:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_\theta} \left[ \tilde{A}_x(y) \sum_t \log \pi_\theta(y_t|x, y_{<t}) \right] + \lambda \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

Where:
- $r_x(y) = \mathbb{1}[A(y) = \hat{a}(x)]$ (reward based on consensus match)
- $\tilde{A}_x(y) = r_x(y) - \bar{r}_x$ (group-normalized advantage)

### 5.3 Majority-Vote DPO (MV-DPO)

Direct preference optimization with debate-derived pairs:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{(y^{+}, y^{-}) \in \mathcal{G}^{+}(x) \times \mathcal{G}^{-}(x)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y^{+}|x)}{\pi_{\text{ref}}(y^{+}|x)} - \log \frac{\pi_\theta(y^{-}|x)}{\pi_{\text{ref}}(y^{-}|x)} \right) \right) \right]$$

Contrasts entire reasoning chains, not just final answers.

### 5.4 Majority-Vote KTO (MV-KTO)

Unpaired formulation with class-balancing weights $\lambda^{+}$ and $\lambda^{-}$:

$$\mathcal{L}_{\text{KTO}}(\theta) = -\lambda^{+} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y^{+} \in \mathcal{G}^{+}(x)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^{+}|x)}{\pi_{\text{ref}}(y^{+}|x)} \right) \right]$$

$$- \lambda^{-} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y^{-} \in \mathcal{G}^{-}(x)} \left[ \log \sigma \left( -\beta \log \frac{\pi_\theta(y^{-}|x)}{\pi_{\text{ref}}(y^{-}|x)} \right) \right]$$

Handles imbalanced outcomes where majority trajectories dominate.

---

## 6. Why Debate > Simple Majority Vote

### Peer Grounding
Debate traces show:
- Agents reference specific peer arguments
- Identification of errors in reasoning
- Justification for convergence or divergence

These patterns are **absent in initial-round traces**, providing richer supervision.

### Consensus Quality Evolution
Base models produce mostly:
- Random agreement (1/3 for M=3)
- Weak majorities (2/3)
- Only 13.4% unanimity initially

Post-training with MACA:
- Unparseable responses: 13.8% → 0.6%
- No-agreement cases: 45.6% → 19.8%
- Unanimous agreement: 13.4% → **43.4%** (3.2x improvement)

---

## 7. Comparison with Related Work

### vs. Self-Consistency Prompting
- **Self-consistency** (Wang et al., 2022): Inference-time, no model improvement
- **MACA**: Post-training, internalizes consistency

### vs. LLM-as-a-Judge
- **Judge approaches**: Suffer from preference leakage, bias under ambiguity
- **MACA**: Self-generated consensus signals, no external judge needed

### vs. TTRL & ScDPO
- **TTRL/ScDPO**: Single-round majority vote
- **MACA**: Multi-round debate with peer grounding, richer signals

### vs. Supervised Fine-tuning on Debate
- **Subramaniam et al., 2025**: SFT on debate traces
- **MACA**: RL-based alternatives (DPO/KTO) achieve superior performance

---

## 8. Key Experimental Insights

### Preference Learning > Scalar Rewards
MV-DPO and MV-KTO outperform MV-GRPO in most cases by:
- Optimizing log-probability gaps between full reasoning trajectories
- Better credit assignment than sparse final-answer rewards
- More effective than SFT's imitation learning

### Peer Context Matters
Training with debate-refined peer context (final-round traces) substantially outperforms:
- Training without context
- Training with initial-round (independent) context

Models learn **relative grounding**: evaluating peer arguments and incorporating feedback.

### Debate > Ground Truth (Comparable)
Debate-derived majority vote (DMV) performs comparably to or better than oracle ground-truth labels:
- DMV provides stable learning signals
- GT labels sometimes degrade performance (especially with KTO)
- DMV scales naturally with sample size
- No external supervision required

---

## 9. Generalization Properties

### Cross-Domain Transfer
Training on math datasets (MATH, GSM8K, MathQA) improves:
- **Math reasoning**: SVAMP
- **Science reasoning**: GPQA (+16.3%)
- **Commonsense reasoning**: CSQA (+11.6%)

**Conclusion**: Self-consistency is a foundational capability for general reasoning, not domain-specific.

### Cross-Setting Transfer
Post-trained models improve across:
- Single-agent zero-shot
- Multi-sample inference (Pass@k, MV@k)
- Multi-agent debate

The learned consensus-forming ability transfers broadly.

---

## 10. Implicit Format Reward

### Observation
Post-training dramatically reduces:
- Truncated/incomplete responses
- Format errors (e.g., "answer is A" vs "\\boxed{A}")
- Token inefficiency

### Mechanism
Consensus requires:
- Complete reasoning chains (for comparison)
- Parseable answers (for aggregation)
- Efficient token usage (within budget)

**Result**: Preference learning implicitly rewards proper formatting and efficiency.

### Evidence
- Base models: 74.8% format errors (CSQA), 82.8% (MATH)
- Post-trained: <1% format errors
- Response length: 22-36% shorter while maintaining/improving accuracy

Yet **69-100% of improvements** stem from better reasoning, not just formatting.

---

## 11. Requirements & Limitations

### Baseline Capability Threshold
MACA requires sufficient initial problem-solving ability:
- Models must generate some correct responses for consensus
- Example: Qwen2B on AMC with 256 tokens → 0% accuracy → no consensus
- Increasing token limit to 512 → 5% → enables weak consensus signal

### Potential Concerns
1. **Bias amplification**: May reinforce existing model biases
2. **No intermediate supervision**: Doesn't validate reasoning steps, only final consensus
3. **Computational cost**: Requires multi-agent inference during training

### Future Directions
- Alternative consensus methods (confidence-weighted voting)
- Heterogeneous agents (different models/architectures)
- Better leveraging of minority traces
- Iterative training (diminishing but continued returns)

---

## 12. Applications to Claude Code Architecture

### Natural Fit with Plugin System

**Skills**: Encapsulate debate behaviors and consensus mechanisms
**Hooks**: Trigger multi-agent consensus at key decision points
**MCP Servers**: Coordinate agent communication and debate orchestration
**Agents**: Multiple Claude instances collaborating through debate

### Ideal Use Cases

1. **Complex reasoning tasks**: Architecture decisions, design reviews
2. **Code review**: Multiple validation passes, consensus on changes
3. **Test generation**: Diverse approaches, consensus on coverage
4. **Documentation review**: Multi-perspective validation
5. **Refactoring proposals**: Evaluated by multiple agents
6. **Error diagnosis**: Collaborative debugging with consensus

### Implementation Strategy

1. **Agent Pool Management**: Maintain clones with adapter hot-swapping
2. **Debate Orchestration**: MCP server coordinates rounds, manages context
3. **Consensus Mechanism**: Aggregate responses, partition by agreement
4. **Preference Learning**: Optional: continue training on debate outcomes
5. **Integration Hooks**: Trigger debate for configured events/commands

---

## Summary

MACA demonstrates that **self-consistency can be internalized through multi-agent consensus alignment**. By using debate-derived preference signals, models learn to:

1. Sample more consistent reasoning trajectories
2. Better utilize peer feedback in collaborative settings
3. Produce more efficient and accurate reasoning
4. Generalize improvements across diverse reasoning domains

The framework requires no external supervision, scales with sample size, and maps naturally to Claude Code's multi-agent plugin architecture.

---

**References**: See full paper for complete bibliography
**Next**: Algorithm details, experimental results, implementation guide
