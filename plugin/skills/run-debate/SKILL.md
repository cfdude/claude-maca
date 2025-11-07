# Run Debate Skill

## Purpose

Execute a complete MACA (Multi-Agent Consensus Alignment) debate workflow, from question selection through DPO pair export. This skill handles the entire orchestration automatically.

## When to Use

Use this skill when you want to:
- Run a single debate end-to-end
- Process a question from the training dataset
- Generate DPO training pairs quickly
- Test the debate system with a new question

## Prerequisites

### 1. Ollama Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### 2. Model Downloaded

```bash
# Download the model (if not already done)
ollama pull qwen2.5:3b

# Verify
ollama list | grep qwen
```

### 3. Agents Registered

Agents must be registered before running debates. Check if agents exist:

```bash
# List registered agents
mcp__maca-debate__list_agents()
```

If no agents, register them:

```bash
mcp__maca-debate__connect_llm(
  id: "agent_1",
  name: "Agent Alpha",
  model: "qwen2.5:3b"
)

mcp__maca-debate__connect_llm(
  id: "agent_2",
  name: "Agent Beta",
  model: "qwen2.5:3b"
)

mcp__maca-debate__connect_llm(
  id: "agent_3",
  name: "Agent Gamma",
  model: "qwen2.5:3b"
)
```

## Usage

### Basic Debate

```bash
/run-debate "Should a client with a 4.5% rate refinance at 6.5%?"
```

This will:
1. Create a debate session
2. Run Round 1 (independent responses)
3. Run Round 2 (with peer feedback)
4. Calculate consensus
5. Export DPO pairs
6. Display results

### With Metadata

```bash
/run-debate "Is 15-year or 30-year better for this client?" category=loan_comparison difficulty=intermediate
```

Metadata fields:
- `category`: refinance_strategy, debt_management, loan_comparison, etc.
- `difficulty`: beginner, intermediate, advanced, expert
- `cma_module`: A, B, C, D, or E
- `debate_worthy`: true/false
- `client_facing`: true/false

### Batch Processing

Run multiple debates from a dataset:

```bash
/run-debate --batch data/training_dataset_complete.json --filter difficulty=advanced
```

Options:
- `--batch <file>`: JSON file with questions
- `--filter <key>=<value>`: Filter questions by metadata
- `--limit <n>`: Process only first N questions
- `--min-consensus <0-1>`: Only export if consensus >= threshold

## Workflow Steps

### Step 1: Start Debate

The skill creates a debate session:

```javascript
const debateId = `debate_${timestamp}`;

mcp__maca-debate__start_debate({
  id: debateId,
  question: question,
  agentIds: ["agent_1", "agent_2", "agent_3"],
  maxRounds: 2,
  metadata: metadata
});
```

### Step 2: Round 1 - Independent Responses

For each agent, call Ollama and submit:

```javascript
for (const agent of agents) {
  // Call Ollama
  const response = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    body: JSON.stringify({
      model: agent.model,
      prompt: question,
      stream: false,
      options: {
        temperature: 0.8,
        top_p: 0.9,
        num_predict: 1000
      }
    })
  });

  const data = await response.json();

  // Extract answer (simple heuristic)
  const answer = extractAnswer(data.response);

  // Submit to MCP
  mcp__maca-debate__submit_response({
    debateId: debateId,
    agentId: agent.id,
    reasoning: data.response,
    answer: answer
  });
}
```

### Step 3: Advance to Round 2

```javascript
mcp__maca-debate__advance_round({
  debateId: debateId
});
```

### Step 4: Round 2 - Peer Feedback

Get Round 1 responses:

```javascript
const history = mcp__maca-debate__get_debate_history({
  debateId: debateId
});

const round1 = history.responses[0];
```

Construct prompts with peer feedback:

```javascript
const prompt = `
Original question: ${question}

Round 1 responses from your peers:
${round1.map(r => `- ${r.agentId}: "${r.answer}" - ${r.reasoning}`).join('\n')}

Now, having seen your peers' reasoning, provide your refined answer:
`;

// Call Ollama for each agent with peer-aware prompt
// Submit responses
```

### Step 5: Advance (Complete)

```javascript
mcp__maca-debate__advance_round({
  debateId: debateId
});
```

### Step 6: Calculate Consensus

```javascript
const consensus = mcp__maca-debate__calculate_consensus({
  debateId: debateId
});

console.log(`Majority: ${consensus.majorityAnswer}`);
console.log(`Consensus: ${(consensus.consensusStrength * 100).toFixed(1)}%`);
```

### Step 7: Export DPO Pairs

```javascript
const pairs = mcp__maca-debate__export_training_data({
  debateId: debateId,
  format: 'json'
});

console.log(`Exported ${pairs.length} DPO training pairs`);
```

## Output Format

The skill returns a comprehensive summary:

```
📊 Debate Complete: debate_20251107_143022

Question: Should a client with a 4.5% rate refinance at 6.5%?
Category: refinance_strategy
Difficulty: advanced

═══════════════════════════════════════
ROUND 1 RESULTS
═══════════════════════════════════════
Responses:
  - Agent Alpha: "wait" (confidence: 0.85)
  - Agent Beta: "wait" (confidence: 0.78)
  - Agent Gamma: "refinance" (confidence: 0.62)

Vote Distribution:
  - wait: 2 votes (66.7%)
  - refinance: 1 vote (33.3%)

Consensus Strength: 66.7%
═══════════════════════════════════════

═══════════════════════════════════════
ROUND 2 RESULTS
═══════════════════════════════════════
Responses:
  - Agent Alpha: "wait" (confidence: 0.90)
  - Agent Beta: "wait" (confidence: 0.88)
  - Agent Gamma: "wait" (confidence: 0.82)

Vote Distribution:
  - wait: 3 votes (100%)

Consensus Strength: 100%
═══════════════════════════════════════

✅ CONVERGENCE: YES (66.7% → 100%)
✅ QUALITY: HIGH (final consensus 100%)
⚠️  TRAINING SIGNAL: LOW (unanimous - no rejected responses)

DPO Pairs Exported: 0
Recommendation: Use for validation set, not training

═══════════════════════════════════════
TIMING
═══════════════════════════════════════
Round 1: 8.2s
Round 2: 9.1s
Total: 17.3s
```

## Quality Thresholds

The skill applies automatic quality filtering:

| Consensus Strength | Action | Reason |
|-------------------|--------|--------|
| ≥ 0.7 | ✅ Export | High-quality training signal |
| 0.5 - 0.7 | ⚠️ Review | Moderate quality, manual review |
| < 0.5 | ❌ Discard | Low agreement, ambiguous question |
| 1.0 | 📝 Validation | Unanimous, use for eval not training |

## Error Handling

### Agent Not Found

```
Error: Agent agent_1 not found
→ Solution: Register agents with /connect_llm first
```

### Ollama Not Responding

```
Error: Failed to connect to Ollama at http://localhost:11434
→ Solution: Start Ollama with `ollama serve`
```

### Model Not Found

```
Error: Model qwen2.5:3b not found
→ Solution: Download model with `ollama pull qwen2.5:3b`
```

### Low Consensus

```
Warning: Consensus strength 0.42 below threshold 0.5
→ This question may be ambiguous. Consider rephrasing or discarding.
```

## Advanced Options

### Custom Temperature

Control response diversity:

```bash
/run-debate "Question" --temperature 0.9
```

Higher temperature = more diverse responses = better debate signal

### Custom Rounds

```bash
/run-debate "Question" --rounds 3
```

More rounds = more convergence but diminishing returns

### Export Format

```bash
/run-debate "Question" --export-format jsonl
```

Options: `json`, `jsonl`, `huggingface`

### Skip Export on Unanimous

```bash
/run-debate "Question" --skip-unanimous
```

Don't export DPO pairs when consensus is 100% (no training signal)

## Integration with Other Skills

This skill works with:

### analyze-consensus

After running multiple debates:

```bash
/analyze-consensus --all-debates
```

Shows aggregate statistics, convergence trends, quality distribution

### export-training-data

Batch export from multiple debates:

```bash
/export-training-data --min-consensus 0.7 --format jsonl
```

## Best Practices

1. **Start Small**: Run 5-10 debates manually before batch processing
2. **Monitor Quality**: Check consensus strength and convergence
3. **Filter Wisely**: Only use high-consensus pairs (>0.7) for training
4. **Save Unanimous**: Perfect consensus questions are great for validation
5. **Iterate**: If low consensus, rephrase question or mark as expert-only

## Performance Tips

### Parallel Execution

Run multiple debates concurrently:

```bash
/run-debate --batch questions.json --parallel 3
```

Runs 3 debates at once (be mindful of Ollama capacity)

### Cache Responses

For testing, cache Ollama responses:

```bash
/run-debate "Question" --cache-responses
```

Subsequent runs with same question use cached responses

## Example Session

Complete example with 3 debates:

```bash
# 1. Register agents (once)
/connect-llm agent_1 "Agent Alpha" qwen2.5:3b
/connect-llm agent_2 "Agent Beta" qwen2.5:3b
/connect-llm agent_3 "Agent Gamma" qwen2.5:3b

# 2. Run debates
/run-debate "Should a client with 4.5% refinance at 6.5%?" category=refinance_strategy difficulty=advanced

/run-debate "Is 15-year or 30-year better for a client staying 10 years?" category=loan_comparison difficulty=intermediate

/run-debate "When should a client lock their rate versus floating?" category=rate_lock_strategy difficulty=expert

# 3. Analyze results
/analyze-consensus --summary

# 4. Export high-quality pairs
/export-training-data --min-consensus 0.7
```

Results:
- Debate 1: 100% consensus, 0 pairs (validation)
- Debate 2: 66.7% → 100% consensus, 0 pairs (validation)
- Debate 3: 60% → 80% consensus, 2 pairs (training)

Total DPO pairs: 2

## Troubleshooting

### Debates Taking Too Long

Each debate should take 15-30 seconds. If slower:
- Check Ollama GPU utilization
- Reduce `num_predict` (max tokens)
- Lower `maxRounds` to 1 or 2

### Poor Convergence

If Round 2 consensus ≤ Round 1:
- Questions may be too ambiguous
- Try higher temperature (more diversity in Round 1)
- Add domain context to prompts
- Consider marking as expert-level (humans disagree too)

### All Unanimous

If all debates result in 100% consensus:
- Questions too easy (use advanced/expert only)
- Temperature too low (increase to 0.8-0.9)
- Agents not seeing peer responses correctly (check prompt construction)

---

**See Also**:
- `/analyze-consensus` - Aggregate debate analysis
- `/export-training-data` - Batch export with filtering
- `debate-orchestrator` agent - Full debate coordination
