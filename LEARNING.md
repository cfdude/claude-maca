# Learning Guide: Training Your First LLM with MACA

## Introduction

This guide is designed for someone who works extensively with AI but has never trained a model before. We'll go from **absolute basics to production deployment** using the MACA research applied to your mortgage/home loan business domain.

**Real-World Goal**: Train an LLM that gives consistent, accurate advice about home loans, refinancing, and mortgage products for Highway.ai customers.

---

## Part 1: Understanding the Fundamentals

### What Does "Training a Model" Actually Mean?

Think of training like this:

**Before Training**: A model is like a new employee who knows grammar and can form sentences, but doesn't know your business.

**After Training**: The model has learned your domain (mortgages), your products, your business rules, and gives consistent advice aligned with your expertise.

#### The Process:
```
Base Model (e.g., Llama 3.2)
    ↓
+ Your Domain Data (mortgage Q&A, loan scenarios)
    ↓
+ Training Process (teaching through examples)
    ↓
= Your Custom Model (mortgage-expert LLM)
```

### What We're NOT Doing (Too Expensive/Complex)

❌ **Pre-training**: Creating a model from scratch (costs millions, needs massive data)
❌ **Full fine-tuning**: Updating all model parameters (needs expensive GPUs)

### What We ARE Doing (Practical & Affordable)

✅ **PEFT (Parameter-Efficient Fine-Tuning)**: Update only a small portion of the model
✅ **LoRA (Low-Rank Adaptation)**: Add small "adapter" layers that are cheap to train
✅ **Local Training**: Use Ollama on your Mac, no cloud GPUs needed
✅ **DPO (Direct Preference Optimization)**: Learn from comparisons, not just answers

---

## Part 2: MACA-Specific Training (What Makes This Special)

### Traditional Fine-Tuning Problem

**Scenario**: You want to teach the model about PMI removal.

**Traditional approach**:
```
Question: "When can I remove PMI?"
Answer: "When you reach 20% equity and request removal from your lender."
```

**Problem**: Model gives different answers at different times:
- "20% equity"
- "22% equity to be safe"
- "Depends on your lender"
- "After 2 years of payments"

**Result**: Inconsistent advice that confuses customers!

### MACA Training Solution

**MACA approach**: Run debates between model copies, train on consensus.

**Debate Example**:
```
Agent 1: "20% equity, but check your loan terms"
Agent 2: "20% equity and you must request it - not automatic"
Agent 3: "Depends on loan type - FHA has different rules"

Majority Consensus: "20% equity + must request + check loan type"
Minority View: "It's automatic at 22%"

Training: ✅ Reinforce consensus   ❌ Discourage wrong answer
```

**Result**: Model learns to give consistent, nuanced answers!

### Why This Matters for Highway.ai

Your loan officers need LLMs that:
1. **Don't contradict themselves** (same question = same core answer)
2. **Handle edge cases** (FHA vs. conventional loans)
3. **Give confident, correct advice** (not random guessing)

MACA directly addresses all three by training on internal consensus.

---

## Part 3: The Training Stack (Tools You'll Use)

### Layer 1: Base Model (Ollama)

**What it is**: Pre-trained LLM running locally on your Mac

**Models we'll use**:
- `qwen2.5:3b` - Lightweight, good for learning (recommended)
- `llama3.2:3b` - Alternative option
- `mistral:7b` - More powerful but needs more RAM

**Why Ollama**:
- Free and local (no API costs)
- Easy model management
- Exports to standard formats for training

**Installation**:
```bash
brew install ollama
ollama pull qwen2.5:3b
ollama run qwen2.5:3b "What is PMI?"
```

### Layer 2: Training Framework (Python + HuggingFace)

**What it is**: Libraries for fine-tuning models

**Key libraries**:
```python
transformers  # Model loading/saving
peft          # LoRA adapters
trl           # DPO training
datasets      # Data handling
```

**Why these**:
- Industry standard
- Well-documented
- Active community
- Free and open-source

### Layer 3: Orchestration (Claude Code + MACA MCP)

**What it is**: Your custom system that coordinates everything

**MACA MCP Server** does:
1. Coordinates multiple Ollama instances (agents)
2. Runs debates on mortgage questions
3. Calculates consensus
4. Exports training data

**Claude Code** does:
1. Orchestrates the pipeline
2. Provides skills for each step
3. Tracks progress
4. Validates results

---

## Part 4: Understanding the Data

### Input: Questions Your Loan Officers Ask

Real examples from your domain:

```
1. "Should this customer refinance from 6.5% to 5.5% if they plan to move in 3 years?"

2. "Customer has $80k equity on a $300k home. Can they do a cash-out refi?"

3. "What's the best loan product for a first-time buyer with 5% down?"

4. "How do I explain PMI to a customer who wants to avoid it?"

5. "When does it make sense to do a 15-year vs 30-year mortgage?"
```

### Output: Structured Answers

Each question needs:
- **Reasoning**: Step-by-step logic
- **Answer**: Clear recommendation
- **Caveats**: Important conditions

Example:
```
Question: "Should this customer refinance from 6.5% to 5.5% if they plan to move in 3 years?"

Reasoning:
1. Rate drop: 1% (100 basis points) - significant savings
2. Time horizon: 3 years - need to calculate break-even
3. Typical closing costs: 2-3% of loan amount
4. Monthly savings needed to recoup costs in 36 months

Answer: Need more information:
- Current loan balance
- Estimated closing costs
- Any prepayment penalties

If closing costs are $6000 and monthly savings are $200:
- Break-even: 30 months
- 3-year horizon: 36 months
- Net benefit: $1200 over 3 years
→ Marginal benefit, may not be worth it

Recommendation: Only if closing costs < $5000 OR staying longer than 3 years

Caveats:
- Assumes no prepayment penalty
- Market conditions may change
- Consider non-financial factors (simplified process, different loan terms)
```

### Training Data Format (What the Model Sees)

**Debate-Derived Preference Pair**:
```json
{
  "prompt": "Should this customer refinance from 6.5% to 5.5% if they plan to move in 3 years?",

  "chosen": "Need to calculate break-even. With typical 2-3% closing costs...[complete reasoning above]...Marginal benefit, only if costs are low or timeline is longer.",

  "rejected": "Yes, definitely refinance! A 1% rate drop always makes sense and will save money immediately."
}
```

**Why this works**:
- `chosen` = what agents agreed on (consensus)
- `rejected` = what minority said (incorrect)
- Model learns: "This thinking pattern → good, that pattern → bad"

---

## Part 5: The Complete Training Pipeline

### Step 0: Preparation (Week 1)

**Goal**: Gather domain knowledge

**Tasks**:
1. Collect 100-200 real mortgage questions
2. Write expert answers for 20-30 (validation set)
3. Organize by category:
   - Refinancing decisions
   - PMI questions
   - Loan product selection
   - First-time buyer guidance
   - Equity/cash-out scenarios

**How to collect**:
- Review support tickets
- Ask loan officers for common questions
- Look at FAQ pages
- Industry forums

**Data file**:
```
mortgage_questions.json
[
  {
    "id": 1,
    "category": "refinancing",
    "question": "...",
    "expert_answer": "..." // Optional, for validation
  },
  ...
]
```

### Step 1: Baseline Evaluation (Week 2)

**Goal**: Measure how bad the base model is

**Script**:
```python
# baseline_eval.py
import ollama

questions = load_mortgage_questions()
results = []

for q in questions:
    # Get 5 responses to measure consistency
    responses = [
        ollama.generate('qwen2.5:3b', q['question'])
        for _ in range(5)
    ]

    # Measure self-consistency
    answers = [extract_answer(r) for r in responses]
    majority = most_common(answers)
    consistency = sum(1 for a in answers if a == majority) / 5

    results.append({
        'question': q['question'],
        'consistency': consistency,
        'responses': responses
    })

# Calculate overall metrics
avg_consistency = mean([r['consistency'] for r in results])
print(f"Baseline consistency: {avg_consistency:.2%}")
# Expected: 30-40% (pretty bad!)
```

**What you'll see**: The base model gives contradictory advice!

Example:
```
Question: "When can I remove PMI?"

Response 1: "At 20% equity"
Response 2: "At 22% equity"
Response 3: "When you've paid for 2 years"
Response 4: "Contact your lender to ask"
Response 5: "At 20% equity"

Majority: "20% equity" (40% consistency)
```

**This proves**: We need MACA training!

### Step 2: Debate Generation (Weeks 3-4)

**Goal**: Create training data through multi-agent debates

**How it works**:

```typescript
// Via MACA MCP Server

for (const question of mortgageQuestions) {
  // Start debate with 3 Ollama instances
  const debate = await startDebate({
    question: question.text,
    numAgents: 3,
    numRounds: 2,
    model: 'qwen2.5:3b'
  });

  // Round 1: Independent responses
  const round1 = await Promise.all([
    agent1.respond(question),
    agent2.respond(question),
    agent3.respond(question)
  ]);

  // Round 2: See peers, update answers
  const round2 = await Promise.all([
    agent1.respondWith(question, [round1.agent2, round1.agent3]),
    agent2.respondWith(question, [round1.agent1, round1.agent3]),
    agent3.respondWith(question, [round1.agent1, round1.agent2])
  ]);

  // Calculate consensus
  const consensus = calculateMajority(round2);

  // Create training pair
  trainingData.push({
    prompt: question.text,
    chosen: round2.find(r => r.answer === consensus.answer),
    rejected: round2.find(r => r.answer !== consensus.answer)
  });
}
```

**Example debate output**:

```
Question: "Cash-out refi with $80k equity on $300k home?"

Agent 1 (Round 1):
"Yes, you have good equity. Cash-out refis are possible with 20%+ equity."

Agent 2 (Round 1):
"Depends on the loan-to-value ratio you want. 80% LTV is standard."

Agent 3 (Round 1):
"You have 26.7% equity. Can potentially access $50-60k while maintaining 80% LTV."

[Round 2 - Agents see each other's responses]

Agent 1 (Round 2):
"I agree with Agent 3's calculation. With $80k equity on $300k, that's 26.7% equity. For 80% LTV, you can access roughly $60k."

Agent 2 (Round 2):
"Confirmed. $300k home - $220k = $80k equity. Can refinance up to $240k (80% of $300k), accessing $20k cash while staying at 80% LTV."

Agent 3 (Round 2):
"Wait, let me recalculate. If current balance is $220k and we can go to $240k at 80% LTV, that's only $20k cash-out, not $50-60k as I said. I update my answer."

Consensus (2/3 agents): "$20k cash-out while maintaining 80% LTV"
Minority (Agent 1): Still saying $60k (incorrect calculation)

Training pair:
✅ Chosen: Agent 2 & 3's consensus reasoning
❌ Rejected: Agent 1's incorrect math
```

**Result**: 100 high-quality preference pairs!

### Step 3: Model Fine-Tuning (Weeks 5-6)

**Goal**: Actually train the model

**Step 3a: Export Ollama Model**

```bash
# Ollama → HuggingFace format
ollama export qwen2.5:3b ./models/qwen2.5-3b-base
```

**Step 3b: Configure Training**

```python
# train_mortgage_maca.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# Load base model
model = AutoModelForCausalLM.from_pretrained("./models/qwen2.5-3b-base")
tokenizer = AutoTokenizer.from_pretrained("./models/qwen2.5-3b-base")

# Configure LoRA (efficient training)
lora_config = LoraConfig(
    r=16,                    # Rank of adapter matrices
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Load your debate-derived training data
from datasets import load_dataset
dataset = load_dataset("json", data_files="mortgage_training_data.json")

# Configure DPO training
training_config = DPOConfig(
    output_dir="./mortgage-maca-model",
    num_train_epochs=3,              # How many times to see the data
    per_device_train_batch_size=1,   # How many examples at once
    learning_rate=5e-5,               # How fast to learn
    beta=0.1,                         # DPO preference strength (from MACA paper)
    logging_steps=10,
    save_steps=50,
    warmup_steps=50
)

# Train!
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # DPO can use implicit reference
    args=training_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model("./mortgage-maca-qwen2.5-3b")
print("✓ Training complete!")
```

**What happens during training** (simplified):

1. **Load example**: Question + Chosen + Rejected
2. **Forward pass**: Model generates probabilities for both responses
3. **Calculate loss**: "How much more should I prefer chosen vs rejected?"
4. **Backward pass**: Update weights to increase that preference
5. **Repeat**: For all 100 examples, 3 times (epochs)

**Training time**: ~2-4 hours on M-series Mac

**What gets saved**:
```
mortgage-maca-qwen2.5-3b/
├── adapter_config.json       # LoRA settings
├── adapter_model.safetensors # Trained weights (small!)
└── tokenizer files
```

**Step 3c: Merge and Deploy**

```python
# merge_model.py
from peft import PeftModel

# Load base model
base = AutoModelForCausalLM.from_pretrained("./models/qwen2.5-3b-base")

# Load trained adapter
model = PeftModel.from_pretrained(base, "./mortgage-maca-qwen2.5-3b")

# Merge into single model
merged = model.merge_and_unload()

# Save
merged.save_pretrained("./mortgage-maca-final")
```

```bash
# Import back to Ollama
cat > Modelfile << EOF
FROM ./mortgage-maca-final
SYSTEM You are a mortgage and home loan expert assistant.
PARAMETER temperature 0.7
EOF

ollama create mortgage-maca:latest -f Modelfile
```

### Step 4: Evaluation (Week 7)

**Goal**: Prove it worked!

**Comparison script**:
```python
# evaluate_improvement.py

def evaluate_model(model_name, questions):
    results = {
        'accuracy': [],
        'consistency': [],
    }

    for q in questions:
        # Consistency test (5 samples)
        responses = [
            ollama.generate(model_name, q['question'])
            for _ in range(5)
        ]

        answers = [extract_answer(r) for r in responses]
        majority = most_common(answers)
        consistency = sum(1 for a in answers if a == majority) / 5

        results['consistency'].append(consistency)

        # Accuracy (if we have expert answer)
        if 'expert_answer' in q:
            is_correct = compare_to_expert(majority, q['expert_answer'])
            results['accuracy'].append(is_correct)

    return {
        'avg_consistency': mean(results['consistency']),
        'avg_accuracy': mean(results['accuracy']) if results['accuracy'] else None
    }

# Evaluate both models
baseline = evaluate_model('qwen2.5:3b', test_questions)
maca = evaluate_model('mortgage-maca:latest', test_questions)

print("Results:")
print(f"Baseline consistency: {baseline['avg_consistency']:.1%}")
print(f"MACA consistency: {maca['avg_consistency']:.1%}")
print(f"Improvement: +{(maca['avg_consistency'] - baseline['avg_consistency']):.1%}")

# Statistical significance
from scipy import stats
t_stat, p_value = stats.ttest_rel(
    baseline['consistency'],
    maca['consistency']
)
print(f"p-value: {p_value:.4f}")
print(f"Significant: {p_value < 0.05}")
```

**Expected results**:
```
Baseline consistency: 35.2%
MACA consistency: 62.8%
Improvement: +27.6%
p-value: 0.0023
Significant: True
```

**Visualization**:
```python
import matplotlib.pyplot as plt

metrics = ['Consistency', 'Accuracy']
baseline_scores = [35.2, 48.5]
maca_scores = [62.8, 61.2]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, baseline_scores, width, label='Base Model', color='#FF6B6B')
ax.bar(x + width/2, maca_scores, width, label='MACA-Trained', color='#51CF66')

ax.set_ylabel('Percentage')
ax.set_title('MACA Training Impact on Mortgage LLM (Qwen2.5-3B)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for i, (b, m) in enumerate(zip(baseline_scores, maca_scores)):
    improvement = m - b
    ax.annotate(f'+{improvement:.1f}pp',
                xy=(i, max(b, m) + 2),
                ha='center',
                fontweight='bold',
                color='green')

plt.tight_layout()
plt.savefig('mortgage_maca_results.png', dpi=300)
```

---

## Part 6: Deployment to Highway.ai

### Strategy 1: Ollama Server (Simple)

**What**: Run trained model on a server, API access

**Setup**:
```bash
# On server
ollama serve
ollama pull mortgage-maca:latest

# Python API wrapper
from flask import Flask, request
import ollama

app = Flask(__name__)

@app.route('/mortgage-advice', methods=['POST'])
def get_advice():
    question = request.json['question']
    response = ollama.generate('mortgage-maca:latest', question)
    return {'advice': response}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

**Pro**: Simple, cheap
**Con**: Limited scale

### Strategy 2: HuggingFace Inference (Scalable)

**What**: Host on HuggingFace, serverless scaling

**Steps**:
1. Upload model to HuggingFace Hub
2. Use Inference Endpoints
3. Pay per request

**Code**:
```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="highway-ai/mortgage-maca-qwen2.5",
    token="your-hf-token"
)

response = client.text_generation(
    "Should I remove PMI now?",
    max_new_tokens=500
)
```

**Pro**: Scales automatically
**Con**: Costs per API call

### Strategy 3: Integration with Your SaaS

**For Highway.ai web app**:

```typescript
// frontend/src/services/mortgageAI.ts

export async function getMortgageAdvice(question: string) {
  const response = await fetch('https://api.highway.ai/mortgage-advice', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });

  return response.json();
}

// Usage in your loan officer dashboard
const advice = await getMortgageAdvice(
  "Customer has $80k equity on $300k home. Cash-out refi options?"
);

displayAdvice(advice);
```

---

## Part 7: Continuous Improvement

### Collecting Real-World Data

As loan officers use the system:

```python
# Track interactions
def log_interaction(question, ai_response, officer_feedback):
    db.interactions.insert({
        'question': question,
        'ai_response': ai_response,
        'officer_edited_response': officer_feedback,
        'was_helpful': officer_feedback['rating'],
        'timestamp': datetime.now()
    })

# Weekly: Review low-rated responses
low_rated = db.interactions.find({'was_helpful': {'$lt': 3}})

# Add to training data
for interaction in low_rated:
    training_data.add_preference_pair(
        prompt=interaction['question'],
        chosen=interaction['officer_edited_response'],  # What they changed it to
        rejected=interaction['ai_response']  # What AI said
    )

# Re-train monthly with new data
retrain_model(training_data)
```

### Version Control for Models

```bash
models/
├── mortgage-maca-v1.0-2024-11  # Initial training
├── mortgage-maca-v1.1-2024-12  # After first month of feedback
└── mortgage-maca-v2.0-2025-01  # Major update with new products
```

---

## Part 8: Advanced Topics (Future Learning)

Once you've mastered the basics, explore:

### 1. Multi-Task Training

Train one model for multiple tasks:
- Loan qualification
- Refinancing advice
- First-time buyer guidance
- Investment property loans

### 2. Retrieval-Augmented Generation (RAG)

Combine your trained model with:
- Current interest rates (API)
- Loan product database
- Customer history
- Regulatory documents

### 3. Safety & Compliance

Add guardrails:
- Don't give advice outside of scope
- Always disclaimer: "Consult with licensed professional"
- Detect and refuse illegal scenarios
- Log all advice for compliance

### 4. Model Distillation

Once your model works well:
- Train an even smaller model to mimic it
- Deploy tiny, fast version for real-time use
- Keep full model for complex queries

---

## Key Concepts Reference

### LoRA (Low-Rank Adaptation)
**What**: Instead of changing all 3 billion parameters, add small "adapter" layers
**Why**: Trains in hours instead of days, needs 16GB RAM instead of 80GB
**Analogy**: Adding a translation booklet instead of rewriting the entire dictionary

### DPO (Direct Preference Optimization)
**What**: Learn from comparisons: "This answer > That answer"
**Why**: More stable than reinforcement learning, works better with limited data
**Analogy**: Showing examples of good vs bad work instead of explaining rules

### Consensus Training (MACA's Innovation)
**What**: Train on what multiple agents agree on
**Why**: Self-supervising - no need for expert labels on all data
**Analogy**: Democracy of AI - trust the majority opinion

### Self-Consistency
**What**: Asking the same question multiple times gets the same answer
**Why**: Inconsistency = unreliable = can't trust for business decisions
**Analogy**: A calculator that gives different results for 2+2 is broken

---

## Troubleshooting Common Issues

### "Training is too slow"
- Reduce batch size to 1
- Use smaller model (3B instead of 7B)
- Train for fewer epochs (2 instead of 3)
- Use fewer training examples (50 instead of 100)

### "Model isn't improving"
- Check if training data is high quality
- Increase learning rate (1e-4 instead of 5e-5)
- Train for more epochs
- Verify debate consensus was meaningful

### "Out of memory"
- Use 4-bit quantization
- Reduce max sequence length
- Use gradient checkpointing
- Train on fewer examples at once

### "Inconsistent results after training"
- Re-run evaluation with more samples (20 instead of 5)
- Check if using correct temperature (1.0 during eval)
- Verify model was merged correctly
- May need more training data

---

## Summary: Your Learning Path

**Phase 1 (Weeks 1-2)**: Understanding
- Learn concepts (this document!)
- Set up tools (Ollama, Python, MCP server)
- Run baseline evaluation
- See the problem firsthand

**Phase 2 (Weeks 3-4)**: Data Collection
- Build MACA MCP server
- Generate debates on mortgage questions
- Create training dataset
- Understand preference learning

**Phase 3 (Weeks 5-6)**: Training
- Configure LoRA + DPO
- Run training pipeline
- Monitor progress
- Debug issues

**Phase 4 (Week 7)**: Validation
- Evaluate trained model
- Compare to baseline
- Statistical analysis
- Document results

**Phase 5 (Weeks 8+)**: Deployment
- Integration with Highway.ai
- Real-world testing
- Collect feedback
- Continuous improvement

---

## Questions to Ask Me As We Go

**During Setup**:
- "Why do we need LoRA instead of full fine-tuning?"
- "What does the beta parameter in DPO actually control?"
- "How do I know if my training data is good quality?"

**During Training**:
- "What should the loss curve look like?"
- "How do I know when to stop training?"
- "What if the validation loss is increasing?"

**During Evaluation**:
- "How do I interpret the p-value?"
- "What's a good consistency score to aim for?"
- "How many test examples do I need for valid results?"

**For Deployment**:
- "How do I handle rate limiting?"
- "What's the best way to update the model in production?"
- "How do I ensure data privacy with customer questions?"

I'll provide detailed explanations whenever you ask! This is a learning journey - no question is too basic.

---

**Next**: Let's start with Week 1 - gathering your mortgage questions and setting up the environment!
