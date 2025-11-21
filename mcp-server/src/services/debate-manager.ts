/**
 * MACA Debate Manager
 *
 * This service orchestrates multi-agent debates for consensus-based training data generation.
 * It manages debate sessions, agent registrations, round progression, and consensus calculation.
 *
 * EDUCATIONAL NOTE:
 * ==================
 * The debate manager is the heart of the MACA system. It coordinates:
 * 1. Multiple agent clones (all using the same base model)
 * 2. Multi-round debates where agents refine their reasoning
 * 3. Consensus calculation via majority voting
 * 4. Export to DPO training format
 *
 * This implements the core algorithm from the MACA paper:
 * - M agents engage in R rounds of debate
 * - Each round, agents see previous responses and can update reasoning
 * - Final consensus is majority vote
 * - Majority responses become "chosen", minority become "rejected" for DPO
 */

import {
  Agent,
  AgentResponse,
  DebateSession,
  ConsensusResult,
  DPOTrainingPair,
  DebateStats,
  ExportFormat
} from '../types/debate.js';

export class DebateManager {
  /**
   * Active debate sessions indexed by debate ID
   * This allows multiple concurrent debates if needed
   */
  private sessions: Map<string, DebateSession> = new Map();

  /**
   * Registered agents indexed by agent ID
   * Agents can participate in multiple debates
   */
  private agents: Map<string, Agent> = new Map();

  /**
   * All exported DPO pairs from completed debates
   * Accumulated across all debates for final training dataset
   */
  private trainingPairs: DPOTrainingPair[] = [];

  constructor() {
    // Initialize empty collections
  }

  /**
   * Register a new agent for debates
   *
   * EDUCATIONAL NOTE:
   * ==================
   * In MACA, all agents are typically clones (same model, same weights).
   * The diversity in responses comes from:
   * 1. Sampling randomness (temperature > 0)
   * 2. Different random seeds
   * 3. Seeing different orderings of peer responses in later rounds
   *
   * This is different from ensemble methods that use different models.
   *
   * @param id Unique identifier for this agent
   * @param name Display name for logging
   * @param endpoint Optional Ollama endpoint URL
   * @param model Optional specific model identifier
   * @returns The registered agent
   */
  registerAgent(
    id: string,
    name: string,
    endpoint?: string,
    model?: string
  ): Agent {
    if (this.agents.has(id)) {
      throw new Error(`Agent ${id} is already registered`);
    }

    const agent: Agent = {
      id,
      name,
      endpoint,
      model,
      registeredAt: new Date()
    };

    this.agents.set(id, agent);

    console.log(`✅ Registered agent: ${name} (${id})`);
    if (endpoint) console.log(`   Endpoint: ${endpoint}`);
    if (model) console.log(`   Model: ${model}`);

    return agent;
  }

  /**
   * Get a registered agent by ID
   */
  getAgent(id: string): Agent | undefined {
    return this.agents.get(id);
  }

  /**
   * Get all registered agents
   */
  getAllAgents(): Agent[] {
    return Array.from(this.agents.values());
  }

  /**
   * Start a new debate session
   *
   * EDUCATIONAL NOTE:
   * ==================
   * A debate session proceeds in rounds:
   * - Round 1: Agents respond to the question independently
   * - Round 2+: Agents see all Round 1 responses and can refine their answer
   * - Typically 2-3 rounds is sufficient for consensus
   *
   * More rounds can improve consensus but have diminishing returns and
   * increase compute costs linearly (M agents × R rounds = M*R LLM calls).
   *
   * @param id Unique identifier for this debate
   * @param question The question to debate
   * @param agentIds Which agents should participate
   * @param maxRounds How many rounds to run (default: 2)
   * @param metadata Optional metadata to attach
   * @returns The created debate session
   */
  startDebate(
    id: string,
    question: string,
    agentIds: string[],
    maxRounds: number = 2,
    metadata?: Record<string, any>
  ): DebateSession {
    if (this.sessions.has(id)) {
      throw new Error(`Debate ${id} already exists`);
    }

    if (agentIds.length < 2) {
      throw new Error('At least 2 agents required for debate');
    }

    // Verify all agents exist
    const agents: Agent[] = [];
    for (const agentId of agentIds) {
      const agent = this.agents.get(agentId);
      if (!agent) {
        throw new Error(`Agent ${agentId} not found. Register agents first.`);
      }
      agents.push(agent);
    }

    const session: DebateSession = {
      id,
      question,
      agents,
      responses: [], // Will be populated as agents respond
      currentRound: 1,
      maxRounds,
      consensus: null,
      completed: false,
      createdAt: new Date(),
      completedAt: null,
      metadata
    };

    this.sessions.set(id, session);

    console.log(`\n🎯 Started debate: ${id}`);
    console.log(`   Question: ${question}`);
    console.log(`   Agents: ${agents.map(a => a.name).join(', ')} (${agents.length} total)`);
    console.log(`   Max rounds: ${maxRounds}`);

    return session;
  }

  /**
   * Submit a response from an agent for the current round
   *
   * EDUCATIONAL NOTE:
   * ==================
   * In round 1, responses are independent reasoning.
   * In round 2+, responses should incorporate peer feedback.
   *
   * The calling code (MCP tools) is responsible for:
   * 1. Fetching previous round responses via get_debate_history
   * 2. Constructing a prompt that includes peer responses
   * 3. Calling the LLM with that prompt
   * 4. Submitting the result here
   *
   * This separation allows flexibility in how prompts are constructed
   * and which LLM backend is used.
   *
   * @param debateId Which debate this response is for
   * @param agentId Which agent is responding
   * @param reasoning The full text response
   * @param answer The extracted final answer
   * @param confidence Optional confidence score
   */
  submitResponse(
    debateId: string,
    agentId: string,
    reasoning: string,
    answer: string,
    confidence?: number
  ): void {
    const session = this.sessions.get(debateId);
    if (!session) {
      throw new Error(`Debate ${debateId} not found`);
    }

    if (session.completed) {
      throw new Error(`Debate ${debateId} is already completed`);
    }

    // Verify agent is participating in this debate
    const agentParticipating = session.agents.some(a => a.id === agentId);
    if (!agentParticipating) {
      throw new Error(`Agent ${agentId} is not participating in debate ${debateId}`);
    }

    // Ensure responses array has an entry for current round
    while (session.responses.length < session.currentRound) {
      session.responses.push([]);
    }

    const currentRoundResponses = session.responses[session.currentRound - 1];

    // Check if this agent already responded this round
    const alreadyResponded = currentRoundResponses.some(r => r.agentId === agentId);
    if (alreadyResponded) {
      throw new Error(`Agent ${agentId} has already responded in round ${session.currentRound}`);
    }

    const response: AgentResponse = {
      agentId,
      round: session.currentRound,
      reasoning,
      answer,
      timestamp: new Date(),
      confidence
    };

    currentRoundResponses.push(response);

    const agent = this.agents.get(agentId);
    console.log(`  📝 ${agent?.name || agentId} responded in round ${session.currentRound}`);
    console.log(`     Answer: ${answer}`);
    if (confidence !== undefined) {
      console.log(`     Confidence: ${(confidence * 100).toFixed(1)}%`);
    }
  }

  /**
   * Advance to the next round
   *
   * EDUCATIONAL NOTE:
   * ==================
   * Before advancing, verify all agents have responded in the current round.
   * This ensures fair comparison - we want all agents to see the same peer responses
   * when moving to the next round.
   *
   * @param debateId Which debate to advance
   * @throws Error if not all agents have responded yet
   */
  advanceRound(debateId: string): void {
    const session = this.sessions.get(debateId);
    if (!session) {
      throw new Error(`Debate ${debateId} not found`);
    }

    if (session.completed) {
      throw new Error(`Debate ${debateId} is already completed`);
    }

    // Verify all agents responded in current round
    const currentRoundResponses = session.responses[session.currentRound - 1] || [];
    if (currentRoundResponses.length < session.agents.length) {
      const missing = session.agents.length - currentRoundResponses.length;
      throw new Error(
        `Cannot advance round - waiting for ${missing} more agent response(s). ` +
        `(${currentRoundResponses.length}/${session.agents.length} received)`
      );
    }

    if (session.currentRound >= session.maxRounds) {
      // Reached max rounds, mark as completed
      session.completed = true;
      session.completedAt = new Date();
      console.log(`\n✅ Debate ${debateId} completed after ${session.maxRounds} rounds`);
    } else {
      // Advance to next round
      session.currentRound++;
      console.log(`\n➡️  Advanced to round ${session.currentRound}/${session.maxRounds}`);
    }
  }

  /**
   * Calculate consensus from the current round's responses
   *
   * EDUCATIONAL NOTE:
   * ==================
   * Consensus is determined by simple majority voting on the extracted answers.
   * The majority answer becomes the "chosen" response for DPO training.
   * All minority answers become "rejected" responses, each creating a separate DPO pair.
   *
   * Example with 5 agents:
   * - 3 agents answer "A" (majority)
   * - 2 agents answer "B" (minority)
   *
   * This creates 2 DPO pairs:
   * 1. (prompt, chosen="A reasoning", rejected="B reasoning #1")
   * 2. (prompt, chosen="A reasoning", rejected="B reasoning #2")
   *
   * Consensus strength = 3/5 = 0.6 (60% agreement)
   *
   * Higher consensus strength (>0.7) indicates the question is clear and
   * the majority answer is likely correct. Lower consensus (<0.6) might indicate
   * the question is ambiguous or requires expert judgment.
   *
   * @param debateId Which debate to calculate consensus for
   * @param useRound Which round to calculate from (default: current round)
   * @returns Consensus result with majority/minority breakdown
   */
  calculateConsensus(debateId: string, useRound?: number): ConsensusResult {
    const session = this.sessions.get(debateId);
    if (!session) {
      throw new Error(`Debate ${debateId} not found`);
    }

    const round = useRound || session.currentRound;
    const roundResponses = session.responses[round - 1];

    if (!roundResponses || roundResponses.length === 0) {
      throw new Error(`No responses found for round ${round}`);
    }

    // Count votes for each answer
    const voteDistribution: Record<string, number> = {};
    const answerToResponses: Record<string, AgentResponse[]> = {};

    for (const response of roundResponses) {
      const answer = response.answer;
      voteDistribution[answer] = (voteDistribution[answer] || 0) + 1;

      if (!answerToResponses[answer]) {
        answerToResponses[answer] = [];
      }
      answerToResponses[answer].push(response);
    }

    // Find majority answer (most votes)
    let majorityAnswer = '';
    let maxVotes = 0;

    for (const [answer, votes] of Object.entries(voteDistribution)) {
      if (votes > maxVotes) {
        maxVotes = votes;
        majorityAnswer = answer;
      }
    }

    // Separate majority and minority responses
    const majorityReasoning: string[] = [];
    const minorityAnswers: string[] = [];
    const minorityReasoning: string[] = [];

    for (const [answer, responses] of Object.entries(answerToResponses)) {
      if (answer === majorityAnswer) {
        // Majority responses
        for (const response of responses) {
          majorityReasoning.push(response.reasoning);
        }
      } else {
        // Minority responses
        minorityAnswers.push(answer);
        for (const response of responses) {
          minorityReasoning.push(response.reasoning);
        }
      }
    }

    const totalAgents = roundResponses.length;
    const consensusStrength = maxVotes / totalAgents;

    const consensus: ConsensusResult = {
      majorityAnswer,
      majorityReasoning,
      minorityAnswers,
      minorityReasoning,
      voteDistribution,
      totalAgents,
      consensusStrength
    };

    // Store in session
    session.consensus = consensus;

    console.log(`\n📊 Consensus calculated for round ${round}:`);
    console.log(`   Majority answer: ${majorityAnswer} (${maxVotes}/${totalAgents} votes)`);
    console.log(`   Consensus strength: ${(consensusStrength * 100).toFixed(1)}%`);
    console.log(`   Vote distribution:`, voteDistribution);

    return consensus;
  }

  /**
   * Get the complete debate history
   *
   * @param debateId Which debate to retrieve
   * @returns The full debate session
   */
  getDebateHistory(debateId: string): DebateSession {
    const session = this.sessions.get(debateId);
    if (!session) {
      throw new Error(`Debate ${debateId} not found`);
    }
    return session;
  }

  /**
   * Get responses from a specific round
   *
   * @param debateId Which debate
   * @param round Which round (1-indexed)
   * @returns Array of responses from that round
   */
  getRoundResponses(debateId: string, round: number): AgentResponse[] {
    const session = this.getDebateHistory(debateId);

    if (round < 1 || round > session.responses.length) {
      throw new Error(`Invalid round ${round}. Session has ${session.responses.length} round(s).`);
    }

    return session.responses[round - 1];
  }

  /**
   * Export training data in DPO format
   *
   * EDUCATIONAL NOTE:
   * ==================
   * DPO (Direct Preference Optimization) requires (chosen, rejected) pairs.
   * From a debate with consensus, we create:
   * - One "chosen" response: combined reasoning from majority voters
   * - Multiple "rejected" responses: one for each minority vote
   *
   * If a debate has 5 agents with 3 voting for "A" and 2 voting for "B":
   * - Chosen: Reasoning from the 3 "A" voters (often combined)
   * - Rejected #1: Reasoning from "B" voter #1
   * - Rejected #2: Reasoning from "B" voter #2
   *
   * This creates 2 training pairs from one debate.
   *
   * The chosen response can be:
   * 1. Best individual majority response (highest quality)
   * 2. Combined majority reasoning (more comprehensive)
   * 3. Synthesized from all majority responses (requires LLM call)
   *
   * Currently implementing option #1 (first majority response) for simplicity.
   *
   * @param debateId Which debate to export
   * @param format Export format (json, jsonl, or huggingface dataset)
   * @returns Array of DPO training pairs
   */
  exportTrainingData(
    debateId: string,
    format: ExportFormat = 'json'
  ): DPOTrainingPair[] {
    const session = this.getDebateHistory(debateId);

    if (!session.consensus) {
      throw new Error(`Debate ${debateId} has no consensus. Call calculate_consensus first.`);
    }

    if (!session.completed) {
      console.warn(`⚠️  Warning: Debate ${debateId} not marked as completed yet`);
    }

    const { consensus } = session;
    const pairs: DPOTrainingPair[] = [];

    // For now, use the first majority reasoning as the chosen response
    // In production, might want to synthesize or select the best one
    const chosenResponse = consensus.majorityReasoning[0] || '';

    if (!chosenResponse) {
      throw new Error(`No majority reasoning found for debate ${debateId}`);
    }

    // Create one DPO pair for each minority response
    for (let i = 0; i < consensus.minorityReasoning.length; i++) {
      const rejectedResponse = consensus.minorityReasoning[i];

      const pair: DPOTrainingPair = {
        id: `${debateId}_pair_${i + 1}`,
        prompt: session.question,
        chosen: chosenResponse,
        rejected: rejectedResponse,
        debateId: session.id,
        consensusStrength: consensus.consensusStrength,
        metadata: session.metadata
      };

      pairs.push(pair);
      this.trainingPairs.push(pair);
    }

    console.log(`\n💾 Exported ${pairs.length} DPO training pair(s) from debate ${debateId}`);
    console.log(`   Format: ${format}`);
    console.log(`   Consensus strength: ${(consensus.consensusStrength * 100).toFixed(1)}%`);

    return pairs;
  }

  /**
   * Get all accumulated training pairs from all completed debates
   */
  getAllTrainingPairs(): DPOTrainingPair[] {
    return this.trainingPairs;
  }

  /**
   * Calculate statistics for a debate
   *
   * @param debateId Which debate to analyze
   * @returns Statistics object
   */
  calculateStats(debateId: string): DebateStats {
    const session = this.getDebateHistory(debateId);

    const totalResponses = session.responses.flat().length;
    const uniqueAnswersSet = new Set<string>();
    const responseLengths: number[] = [];

    for (const roundResponses of session.responses) {
      for (const response of roundResponses) {
        uniqueAnswersSet.add(response.answer);
        responseLengths.push(response.reasoning.length);
      }
    }

    const uniqueAnswers = uniqueAnswersSet.size;

    // Calculate consensus strength for each round
    const consensusStrengths: number[] = [];
    for (let round = 1; round <= session.responses.length; round++) {
      const roundConsensus = this.calculateConsensus(debateId, round);
      consensusStrengths.push(roundConsensus.consensusStrength);
    }

    const avgConsensusStrength =
      consensusStrengths.reduce((sum, val) => sum + val, 0) / consensusStrengths.length;

    // Check if consensus improved over rounds (convergence)
    const converged = consensusStrengths.length > 1
      ? consensusStrengths[consensusStrengths.length - 1] > consensusStrengths[0]
      : false;

    const avgResponseLength =
      responseLengths.reduce((sum, len) => sum + len, 0) / responseLengths.length;
    const minResponseLength = Math.min(...responseLengths);
    const maxResponseLength = Math.max(...responseLengths);

    return {
      totalResponses,
      uniqueAnswers,
      avgConsensusStrength,
      converged,
      avgResponseLength,
      minResponseLength,
      maxResponseLength
    };
  }

  /**
   * List all active debate sessions
   */
  listDebates(): DebateSession[] {
    return Array.from(this.sessions.values());
  }

  /**
   * Delete a debate session
   *
   * @param debateId Which debate to delete
   */
  deleteDebate(debateId: string): void {
    if (!this.sessions.has(debateId)) {
      throw new Error(`Debate ${debateId} not found`);
    }

    this.sessions.delete(debateId);
    console.log(`🗑️  Deleted debate: ${debateId}`);
  }

  /**
   * Clear all training pairs
   * Useful for starting fresh or exporting in batches
   */
  clearTrainingPairs(): void {
    const count = this.trainingPairs.length;
    this.trainingPairs = [];
    console.log(`🗑️  Cleared ${count} training pair(s)`);
  }
}
