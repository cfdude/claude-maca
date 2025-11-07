/**
 * MACA Debate Type Definitions
 *
 * This file defines the core data structures for the MACA (Multi-Agent Consensus Alignment)
 * debate system. The debate system is based on the research paper:
 * "Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment"
 *
 * EDUCATIONAL NOTE:
 * ==================
 * The MACA framework uses multiple agent clones (M agents) that engage in R rounds of debate.
 * Each agent shares its reasoning, learns from peer feedback, and updates its position.
 * The majority consensus is used as the "chosen" response, while minority responses become "rejected"
 * responses, creating DPO (Direct Preference Optimization) training pairs.
 *
 * Key Concepts:
 * - Agent: An independent LLM instance (clone) participating in the debate
 * - Round: A single iteration where all agents submit responses
 * - Consensus: The majority answer across all agents
 * - DPO Pair: A (chosen, rejected) response pair used for preference learning
 */

/**
 * Represents a single agent participating in the debate
 */
export interface Agent {
  /** Unique identifier for this agent (e.g., "agent_1", "agent_2") */
  id: string;

  /** Display name for logging and debugging */
  name: string;

  /**
   * Connection info for the LLM backend (e.g., Ollama endpoint)
   * Format: "http://localhost:11434" or model name like "qwen2.5:3b"
   */
  endpoint?: string;

  /**
   * Model identifier if using a specific model for this agent
   * All agents typically use the same model for fair comparison
   */
  model?: string;

  /** Timestamp when this agent was registered */
  registeredAt: Date;
}

/**
 * A single response submitted by an agent during a debate round
 */
export interface AgentResponse {
  /** Which agent submitted this response */
  agentId: string;

  /** Which round this response was submitted in (1-indexed) */
  round: number;

  /**
   * The agent's reasoning/answer to the question
   * This is the full text response from the LLM
   */
  reasoning: string;

  /**
   * The final answer extracted from the reasoning
   * For multiple choice: "A", "B", "C", "D"
   * For numeric: "42", "3.14"
   * For open-ended: summary of position
   */
  answer: string;

  /** When this response was submitted */
  timestamp: Date;

  /**
   * Optional: Confidence score (0-1) if the LLM provides one
   * Higher values indicate stronger confidence in the answer
   */
  confidence?: number;
}

/**
 * Results from calculating consensus among agent responses
 */
export interface ConsensusResult {
  /**
   * The majority answer - this becomes the "chosen" response for DPO
   * Determined by simple majority vote across all agents
   */
  majorityAnswer: string;

  /**
   * Full reasoning from agents who voted for the majority answer
   * These are combined to create the "chosen" response in DPO training
   */
  majorityReasoning: string[];

  /**
   * Answers that were in the minority - become "rejected" responses for DPO
   * Each minority answer creates a separate DPO pair
   */
  minorityAnswers: string[];

  /**
   * Full reasoning from agents who voted for minority answers
   * These become the "rejected" responses in DPO training
   */
  minorityReasoning: string[];

  /**
   * Vote distribution - how many agents voted for each answer
   * Example: { "A": 5, "B": 2, "C": 1 } means 5 agents chose A, 2 chose B, 1 chose C
   */
  voteDistribution: Record<string, number>;

  /**
   * Total number of agents that participated in this round
   * Should match the number of registered agents
   */
  totalAgents: number;

  /**
   * Consensus strength (0-1) - measures agreement among agents
   * 1.0 = unanimous, 0.5 = evenly split, 0.33 = three-way tie, etc.
   * Higher values indicate stronger consensus
   * Formula: (votes for majority answer) / (total agents)
   */
  consensusStrength: number;
}

/**
 * Complete state of an active debate session
 *
 * EDUCATIONAL NOTE:
 * ==================
 * A debate proceeds in rounds. In round 1, agents respond to the original question.
 * In subsequent rounds (2, 3, ..., R), agents can see all previous responses and
 * refine their reasoning based on peer feedback. This iterative process often leads
 * to consensus and improved reasoning quality.
 */
export interface DebateSession {
  /** Unique identifier for this debate */
  id: string;

  /**
   * The question being debated
   * Example: "Should this client refinance at 6.5% if their current rate is 4.5%?"
   */
  question: string;

  /**
   * Agents participating in this debate
   * Typically M=3 to M=7 agents for good consensus without excessive compute
   */
  agents: Agent[];

  /**
   * All responses submitted across all rounds
   * Organized chronologically as [round1_responses, round2_responses, ...]
   */
  responses: AgentResponse[][];

  /**
   * Which round we're currently in (1-indexed)
   * Starts at 1, increments with each advance_round call
   */
  currentRound: number;

  /**
   * Maximum number of rounds for this debate
   * Typically 2-3 rounds is sufficient for convergence
   * More rounds = more compute but potentially better consensus
   */
  maxRounds: number;

  /**
   * Current consensus if calculated, null if not yet calculated
   * Updated each time calculate_consensus is called
   */
  consensus: ConsensusResult | null;

  /**
   * Whether this debate has completed (reached maxRounds or forced completion)
   */
  completed: boolean;

  /** When this debate session was created */
  createdAt: Date;

  /** When this debate was completed (null if still active) */
  completedAt: Date | null;

  /**
   * Optional metadata about this debate
   * Can include: category, difficulty, source_material, experiment_id, etc.
   */
  metadata?: Record<string, any>;
}

/**
 * A DPO (Direct Preference Optimization) training pair
 *
 * EDUCATIONAL NOTE:
 * ==================
 * DPO is a training method that teaches models to prefer certain responses over others.
 * Each pair contains a "chosen" response (the better one, from majority consensus) and
 * a "rejected" response (the worse one, from minority opinions).
 *
 * The model is trained to increase the probability of generating the "chosen" response
 * and decrease the probability of the "rejected" response, given the same prompt.
 *
 * This is more efficient than RLHF (Reinforcement Learning from Human Feedback) and
 * doesn't require a separate reward model.
 */
export interface DPOTrainingPair {
  /** Unique identifier for this training example */
  id: string;

  /** The question/prompt that was posed to the agents */
  prompt: string;

  /**
   * The preferred response (from majority consensus)
   * This is what we want the model to learn to generate
   */
  chosen: string;

  /**
   * The dis-preferred response (from minority opinions)
   * This is what we want the model to learn NOT to generate
   */
  rejected: string;

  /**
   * Which debate session this pair came from
   * Useful for tracking provenance and experiment analysis
   */
  debateId: string;

  /**
   * How strong the consensus was (0-1)
   * Higher values = more confident training signal
   * Low values (<0.6) might indicate the question is ambiguous
   */
  consensusStrength: number;

  /**
   * Optional metadata matching the original debate
   * Can include: category, difficulty, module, debate_worthy, etc.
   */
  metadata?: Record<string, any>;
}

/**
 * Configuration for connecting to an Ollama instance
 */
export interface OllamaConfig {
  /**
   * Base URL for Ollama API
   * Default: "http://localhost:11434"
   */
  endpoint: string;

  /**
   * Model to use for debates
   * Example: "qwen2.5:3b", "llama3.2:3b", "mistral:latest"
   */
  model: string;

  /**
   * Temperature for sampling (0.0 - 1.0)
   * Higher = more creative/diverse responses
   * Lower = more focused/deterministic responses
   * Recommended: 0.7-0.9 for debates to encourage diversity
   */
  temperature?: number;

  /**
   * Maximum tokens to generate per response
   * Longer allows more detailed reasoning but costs more compute
   * Recommended: 500-1000 for mortgage advisory
   */
  maxTokens?: number;
}

/**
 * Export format options for training data
 */
export type ExportFormat = 'json' | 'jsonl' | 'huggingface';

/**
 * Statistics about a debate session
 */
export interface DebateStats {
  /** Total number of responses submitted */
  totalResponses: number;

  /** Number of unique answers across all rounds */
  uniqueAnswers: number;

  /** Average consensus strength across rounds */
  avgConsensusStrength: number;

  /**
   * Convergence: Did agents move toward consensus over rounds?
   * Measures if round 2+ has higher consensus than round 1
   */
  converged: boolean;

  /**
   * Response length statistics
   * Helps identify if agents are being too verbose or too terse
   */
  avgResponseLength: number;
  minResponseLength: number;
  maxResponseLength: number;
}
