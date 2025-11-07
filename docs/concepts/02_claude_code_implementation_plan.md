# MACA Implementation Plan for Claude Code

## Overview

This document outlines a practical implementation strategy for bringing MACA principles into Claude Code through a plugin architecture.

---

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Plugin Structure Setup

```
claude-code-maca-plugin/
├── plugin.json                 # Plugin metadata
├── README.md                   # Documentation
├── skills/                     # Debate & consensus skills
│   ├── initiate-debate.md
│   ├── participate-round.md
│   └── aggregate-consensus.md
├── hooks/                      # Integration hooks
│   ├── pre-commit-consensus.sh
│   └── code-review-debate.sh
├── mcp-server/                 # Debate orchestration server
│   ├── package.json
│   ├── src/
│   │   ├── debate-manager.ts
│   │   ├── agent-pool.ts
│   │   └── consensus-calculator.ts
│   └── tools/
│       ├── start-debate.ts
│       ├── submit-response.ts
│       └── get-consensus.ts
└── agents/                     # Agent configurations
    ├── base-debater.md
    └── specialized-roles.md
```

### 1.2 Core Components

#### MCP Server: Debate Orchestration
**Purpose**: Coordinate multi-agent debates, manage state, calculate consensus

**Tools to expose**:
- `start_debate(question, context, num_agents, num_rounds)`
- `submit_response(debate_id, agent_id, round, response)`
- `get_debate_status(debate_id)`
- `calculate_consensus(debate_id)`
- `get_majority_minority(debate_id)`

#### Skills: Debate Behaviors
**initiate-debate.md**: How to structure initial responses
**participate-round.md**: How to incorporate peer feedback
**aggregate-consensus.md**: How to synthesize final answers

#### Hooks: Integration Points
**pre-commit-consensus**: Run debate before committing changes
**code-review-debate**: Multi-agent code review consensus

---

## Phase 2: MCP Server Implementation (Weeks 3-4)

### 2.1 Debate Manager

```typescript
interface DebateState {
  id: string;
  question: string;
  context: Record<string, any>;
  numAgents: number;
  numRounds: number;
  currentRound: number;
  responses: AgentResponse[][];
  status: 'active' | 'completed' | 'failed';
}

interface AgentResponse {
  agentId: number;
  round: number;
  reasoning: string;
  answer: string;
  confidence?: number;
}

class DebateManager {
  private debates: Map<string, DebateState>;

  startDebate(params: StartDebateParams): DebateState;
  submitResponse(debateId: string, response: AgentResponse): void;
  advanceRound(debateId: string): void;
  calculateConsensus(debateId: string): ConsensusResult;
}
```

### 2.2 Agent Pool Management

```typescript
class AgentPool {
  private baseContext: string;
  private agentConfigs: AgentConfig[];

  // Create agent instances with different perspectives
  createAgents(count: number, task: string): Agent[];

  // Manage agent lifecycle
  initializeAgent(id: number, context: string): Agent;
  resetAgent(id: number): void;

  // Context management for rounds
  preparePeerContext(debateId: string, agentId: number): string;
}
```

### 2.3 Consensus Calculator

```typescript
interface ConsensusResult {
  majorityAnswer: string;
  agreementRate: number;  // d^M_θ,τ(x)
  majorityResponses: AgentResponse[];
  minorityResponses: AgentResponse[];
  unanimity: boolean;
  distribution: Map<string, number>;
}

class ConsensusCalculator {
  calculate(responses: AgentResponse[]): ConsensusResult;

  // Calculate sampling consistency s^t_θ,τ(x)
  calculateSamplingConsistency(responses: AgentResponse[]): number;

  // Partition into G+ and G-
  partitionByConsensus(responses: AgentResponse[], consensus: string): {
    supporting: AgentResponse[];
    dissenting: AgentResponse[];
  };
}
```

---

## Phase 3: Skills Implementation (Weeks 5-6)

### 3.1 Initiate Debate Skill

```markdown
# Skill: Initiate Debate

You are participating in the first round of a multi-agent debate to solve: {{question}}

Context:
{{context}}

Instructions:
1. Analyze the problem independently
2. Develop your reasoning step-by-step
3. Arrive at a clear answer
4. Structure your response as:
   - **Reasoning**: Your step-by-step thought process
   - **Answer**: Your final answer in the format: \\boxed{answer}

Be thorough but concise. Your response will be shared with other agents in subsequent rounds.
```

### 3.2 Participate Round Skill

```markdown
# Skill: Participate in Debate Round {{round}}

Original question: {{question}}

Your previous response:
{{your_previous_response}}

Other agents' responses from round {{previous_round}}:
{{peer_responses}}

Instructions:
1. Review your previous reasoning
2. Consider the peer arguments and reasoning
3. Identify strengths/weaknesses in different approaches
4. Update or maintain your answer based on this deliberation
5. Provide your updated response:
   - **Analysis**: What you learned from peer responses
   - **Reasoning**: Your updated or confirmed reasoning
   - **Answer**: Your final answer in the format: \\boxed{answer}

You may change your answer if peer arguments are more convincing, or maintain it if you believe your reasoning is sound.
```

### 3.3 Aggregate Consensus Skill

```markdown
# Skill: Aggregate Consensus

You are facilitating the final consensus from a multi-agent debate.

Question: {{question}}

Final round responses:
{{all_final_responses}}

Consensus analysis:
- Majority answer: {{majority_answer}}
- Agreement rate: {{agreement_rate}}%
- Unanimous: {{is_unanimous}}

Distribution:
{{answer_distribution}}

Your task:
1. Explain why the majority answer is supported
2. Summarize the key reasoning that led to consensus
3. Note any remaining disagreements or edge cases
4. Provide the final recommended answer

Format:
**Consensus Reasoning**: ...
**Final Answer**: \\boxed{{{majority_answer}}}
**Confidence**: {{agreement_rate}}%
```

---

## Phase 4: Hooks Integration (Week 7)

### 4.1 Pre-Commit Consensus Hook

```bash
#!/bin/bash
# .claude/hooks/pre-commit-consensus.sh

# Trigger multi-agent debate on staged changes

echo "Running multi-agent consensus check on staged changes..."

# Get staged files
STAGED_FILES=$(git diff --cached --name-only)

if [ -z "$STAGED_FILES" ]; then
  exit 0
fi

# Prepare context
CONTEXT=$(git diff --cached)

# Start debate via MCP
DEBATE_RESULT=$(claude-cli mcp call maca start_debate \
  --question "Should these changes be committed?" \
  --context "$CONTEXT" \
  --num_agents 3 \
  --num_rounds 2)

# Parse consensus
CONSENSUS=$(echo "$DEBATE_RESULT" | jq -r '.majorityAnswer')
AGREEMENT=$(echo "$DEBATE_RESULT" | jq -r '.agreementRate')

if [ "$CONSENSUS" = "yes" ] && [ "$AGREEMENT" -gt 66 ]; then
  echo "✓ Consensus reached: Proceed with commit ($AGREEMENT% agreement)"
  exit 0
else
  echo "✗ No consensus reached ($AGREEMENT% agreement)"
  echo "Review the debate results and address concerns before committing."
  echo "$DEBATE_RESULT" | jq -r '.reasoning'
  exit 1
fi
```

### 4.2 Code Review Debate Hook

```bash
#!/bin/bash
# .claude/hooks/code-review-debate.sh

PR_NUMBER=$1
CODE_CHANGES=$(gh pr diff $PR_NUMBER)

DEBATE_RESULT=$(claude-cli mcp call maca start_debate \
  --question "Should PR #$PR_NUMBER be approved?" \
  --context "$CODE_CHANGES" \
  --num_agents 3 \
  --num_rounds 2)

# Post debate summary as PR comment
SUMMARY=$(echo "$DEBATE_RESULT" | jq -r '.summary')
gh pr comment $PR_NUMBER --body "$SUMMARY"
```

---

## Phase 5: Advanced Features (Weeks 8-10)

### 5.1 Iterative Training (Optional)

If collecting debate data for further model improvement:

```typescript
interface TrainingDataCollector {
  // Collect debate outcomes
  collectDebate(debate: DebateState): DebateRecord;

  // Partition into majority/minority for preference learning
  preparePreferenceData(debates: DebateRecord[]): PreferenceDataset;

  // Export for external training
  exportForTraining(format: 'dpo' | 'kto' | 'grpo'): TrainingData;
}
```

### 5.2 Specialized Agents

Create agent roles with different perspectives:

```markdown
# Agent: Security Reviewer
Focus on security implications, potential vulnerabilities, and attack vectors.

# Agent: Performance Optimizer
Focus on efficiency, performance bottlenecks, and optimization opportunities.

# Agent: Maintainability Advocate
Focus on code clarity, documentation, and long-term maintainability.
```

### 5.3 Consensus Metrics Dashboard

Track consensus quality over time:
- Agreement rates per question type
- Unanimous vs. split decisions
- Debate rounds needed for convergence
- Accuracy when consensus is high vs. low

---

## Phase 6: Testing & Validation (Week 11)

### 6.1 Test Scenarios

```typescript
describe('MACA Plugin', () => {
  it('should reach unanimous consensus on clear problems', async () => {
    const result = await startDebate({
      question: 'What is 2 + 2?',
      numAgents: 3,
      numRounds: 2
    });

    expect(result.unanimity).toBe(true);
    expect(result.majorityAnswer).toBe('4');
  });

  it('should handle disagreement constructively', async () => {
    const result = await startDebate({
      question: 'Should we use TypeScript or JavaScript?',
      numAgents: 3,
      numRounds: 3
    });

    expect(result.agreementRate).toBeGreaterThan(0.5);
    expect(result.majorityResponses.length).toBeGreaterThanOrEqual(2);
  });

  it('should improve through debate rounds', async () => {
    const debate = await startDebate({
      question: 'Complex architectural decision',
      numAgents: 3,
      numRounds: 3
    });

    const round1Agreement = calculateAgreement(debate, 1);
    const round3Agreement = calculateAgreement(debate, 3);

    expect(round3Agreement).toBeGreaterThanOrEqual(round1Agreement);
  });
});
```

### 6.2 Benchmark Tasks

- **Reasoning**: Math problems, logic puzzles
- **Code Review**: Sample PRs with known issues
- **Architecture**: Design decisions with clear trade-offs
- **Debugging**: Error diagnosis scenarios

---

## Phase 7: Documentation & Deployment (Week 12)

### 7.1 User Guide

```markdown
# Using the MACA Plugin

## Installation

npm install -g @claude-code/maca-plugin
claude plugin install maca

## Basic Usage

### Start a Debate

claude mcp call maca start_debate \
  --question "Should we refactor this function?" \
  --context "$(cat src/myfile.ts)" \
  --num_agents 3 \
  --num_rounds 2

### Enable Pre-Commit Consensus

claude config set hooks.pre-commit .claude/hooks/pre-commit-consensus.sh

### Code Review with Debate

gh pr create --fill
claude mcp call maca code_review_debate --pr-number 123
```

### 7.2 Best Practices

1. **When to Use Debate**:
   - Complex decisions with multiple valid approaches
   - High-stakes changes requiring validation
   - Ambiguous problems benefiting from diverse perspectives

2. **When NOT to Use Debate**:
   - Simple, deterministic tasks
   - Time-sensitive operations
   - Clear-cut decisions with obvious answers

3. **Optimizing Debates**:
   - Start with 3 agents, increase for complex problems
   - Use 2-3 rounds typically
   - Provide clear, focused questions
   - Include relevant context

---

## Success Metrics

### Quantitative
- **Consensus Rate**: % of debates reaching >66% agreement
- **Accuracy When Consensus**: % correct when agreement >66%
- **Debate Efficiency**: Average rounds to convergence
- **Token Usage**: Tokens per debate vs. value added

### Qualitative
- **Decision Quality**: Better architectural choices
- **Code Quality**: Fewer bugs in reviewed code
- **Team Confidence**: Increased confidence in AI-assisted decisions
- **Learning**: Improved understanding from debate rationales

---

## Future Enhancements

1. **Confidence Weighting**: Weight votes by agent confidence scores
2. **Heterogeneous Agents**: Different models/sizes for specialized roles
3. **Active Learning**: Identify uncertain cases for human review
4. **Debate Visualization**: UI for exploring debate threads
5. **Cross-Project Learning**: Share debate patterns across codebases
6. **Integration with CI/CD**: Automated consensus checks in pipelines

---

## Resource Requirements

### Development
- **Engineer Time**: 12 weeks (1 engineer full-time)
- **Cloud Resources**: MCP server hosting, minimal compute
- **Testing**: Benchmark datasets, test scenarios

### Runtime
- **Per Debate**: 3 agents × 2 rounds = 6 LLM calls
- **Latency**: ~10-30s per debate (parallel agent calls)
- **Cost**: ~$0.01-0.10 per debate depending on model/length

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| High latency kills UX | High | Implement async debates, progress indicators |
| Consensus on wrong answers | Medium | Track accuracy metrics, confidence thresholds |
| Cost concerns | Medium | Configurable agent counts, selective activation |
| Complexity for simple tasks | Low | Clear guidelines on when to use |
| Agent bias amplification | Medium | Monitor diversity, test with edge cases |

---

## Conclusion

This implementation plan provides a structured approach to bringing MACA principles into Claude Code. The plugin architecture naturally supports multi-agent coordination, and the value proposition—improved decision quality through consensus—aligns with real-world development workflows.

**Key Takeaway**: Start with Phase 1-4 for a working prototype, then iterate based on user feedback and measured success metrics.
