#!/usr/bin/env node

/**
 * MACA Debate MCP Server
 *
 * This MCP (Model Context Protocol) server implements the MACA (Multi-Agent Consensus Alignment)
 * framework for generating training data through multi-agent debates.
 *
 * EDUCATIONAL NOTE - What is MCP?
 * ================================
 * MCP (Model Context Protocol) is a standard for exposing tools and resources to AI assistants
 * like Claude. Think of it as an API that Claude can call to perform actions or fetch data.
 *
 * An MCP server exposes:
 * - **Tools**: Functions Claude can call (like start_debate, submit_response)
 * - **Resources**: Data Claude can read (like debate history, training pairs)
 * - **Prompts**: Pre-built templates Claude can use
 *
 * This server exposes tools that orchestrate multi-agent debates for creating
 * high-quality training data for mortgage advisory LLMs.
 *
 * EDUCATIONAL NOTE - How MACA Works:
 * ===================================
 * 1. Register M agents (all using same base model, e.g., qwen2.5:3b via Ollama)
 * 2. Start debate with a question from the mortgage domain
 * 3. Round 1: Each agent independently generates reasoning + answer
 * 4. Round 2+: Agents see peer responses and can refine their reasoning
 * 5. Calculate consensus via majority voting
 * 6. Export to DPO format: majority = "chosen", minority = "rejected"
 * 7. Use DPO pairs to fine-tune the model to prefer consensus reasoning
 *
 * This creates a virtuous cycle:
 * - Base model → Debates → DPO pairs → Fine-tuned model → Better debates → Better model
 *
 * Research shows +27.6% improvement on reasoning tasks after MACA training.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool
} from '@modelcontextprotocol/sdk/types.js';
import { DebateManager } from './services/debate-manager.js';
import { ExportFormat } from './types/debate.js';

/**
 * Initialize the debate manager
 * This is a singleton that persists across tool calls
 */
const debateManager = new DebateManager();

/**
 * Create the MCP server instance
 */
const server = new Server(
  {
    name: 'maca-debate-server',
    version: '1.0.0'
  },
  {
    capabilities: {
      tools: {}
    }
  }
);

/**
 * Define all available tools
 *
 * EDUCATIONAL NOTE:
 * ==================
 * Each tool is a function Claude can call. The input schema defines what
 * parameters the tool accepts. Think of these like REST API endpoints.
 */

const tools: Tool[] = [
  {
    name: 'connect_llm',
    description: `Register an Ollama LLM instance as an agent for debates.

EDUCATIONAL NOTE:
In MACA, all agents are typically clones (same model, same weights). The diversity in responses comes from sampling randomness (temperature > 0) and different random seeds.

Call this once for each agent you want to participate in debates. For a typical setup with 3 agents, call this 3 times with different agent IDs but the same model/endpoint.

Example: 3-agent setup with qwen2.5:3b
- connect_llm(id="agent_1", name="Agent Alpha", model="qwen2.5:3b")
- connect_llm(id="agent_2", name="Agent Beta", model="qwen2.5:3b")
- connect_llm(id="agent_3", name="Agent Gamma", model="qwen2.5:3b")`,
    inputSchema: {
      type: 'object',
      properties: {
        id: {
          type: 'string',
          description: 'Unique identifier for this agent (e.g., "agent_1", "agent_2")'
        },
        name: {
          type: 'string',
          description: 'Display name for this agent (e.g., "Agent Alpha")'
        },
        endpoint: {
          type: 'string',
          description: 'Ollama endpoint URL (default: http://localhost:11434)',
          default: 'http://localhost:11434'
        },
        model: {
          type: 'string',
          description: 'Model identifier (e.g., "qwen2.5:3b", "llama3.2:3b")'
        }
      },
      required: ['id', 'name', 'model']
    }
  },

  {
    name: 'start_debate',
    description: `Initialize a new multi-agent debate session.

EDUCATIONAL NOTE:
A debate session coordinates M agents debating a question over R rounds. Typical settings:
- M = 3 to 5 agents (good balance of diversity vs compute cost)
- R = 2 rounds (round 1 = independent, round 2 = with peer feedback)

The question should be from your training domain (mortgage advisory). Good questions are:
- Strategic decisions with multiple valid approaches
- Scenarios requiring reasoning and trade-off analysis
- Questions where expert consensus adds value

After starting a debate, use submit_response for each agent to provide their reasoning.`,
    inputSchema: {
      type: 'object',
      properties: {
        id: {
          type: 'string',
          description: 'Unique identifier for this debate session'
        },
        question: {
          type: 'string',
          description: 'The question to debate (from mortgage/finance domain)'
        },
        agentIds: {
          type: 'array',
          items: { type: 'string' },
          description: 'Array of agent IDs to participate (must be pre-registered via connect_llm)'
        },
        maxRounds: {
          type: 'number',
          description: 'Maximum number of debate rounds (default: 2)',
          default: 2
        },
        metadata: {
          type: 'object',
          description: 'Optional metadata (e.g., {category: "debt_management", difficulty: "advanced"})'
        }
      },
      required: ['id', 'question', 'agentIds']
    }
  },

  {
    name: 'submit_response',
    description: `Submit a response from an agent for the current round.

EDUCATIONAL NOTE:
In round 1, the agent responds independently to the question.
In round 2+, the agent should see previous responses and refine their reasoning.

The calling code (you, Claude) is responsible for:
1. Getting debate history via get_debate_history
2. Constructing a prompt that includes the question + previous responses
3. Calling Ollama to get the agent's response
4. Extracting the final answer (e.g., "A", "B", "refinance", "15-year")
5. Submitting both reasoning and answer here

Example workflow for round 2:
1. get_debate_history(debate_id="debate_001") → see round 1 responses
2. Construct prompt: "Question: ... | Agent 1 said: ... | Agent 2 said: ... | Your turn:"
3. Call Ollama with that prompt
4. Extract answer from response
5. submit_response(debate_id, agent_id, reasoning, answer)`,
    inputSchema: {
      type: 'object',
      properties: {
        debateId: {
          type: 'string',
          description: 'Which debate this response is for'
        },
        agentId: {
          type: 'string',
          description: 'Which agent is submitting this response'
        },
        reasoning: {
          type: 'string',
          description: 'Full text of the agent\'s reasoning/response from the LLM'
        },
        answer: {
          type: 'string',
          description: 'Extracted final answer (e.g., "refinance", "15-year", "A")'
        },
        confidence: {
          type: 'number',
          description: 'Optional confidence score (0-1) if the LLM provides one'
        }
      },
      required: ['debateId', 'agentId', 'reasoning', 'answer']
    }
  },

  {
    name: 'advance_round',
    description: `Move the debate to the next round.

EDUCATIONAL NOTE:
Call this after all agents have submitted responses for the current round.
The server will verify all agents responded before advancing.

If this is the final round (currentRound >= maxRounds), the debate will be marked as completed.

Workflow:
1. All agents submit responses for round N
2. Call advance_round
3. Server checks: did all agents respond?
   - Yes → advance to round N+1 (or mark completed if maxRounds reached)
   - No → throw error with count of missing responses

After advancing, you can start collecting round N+1 responses using submit_response.`,
    inputSchema: {
      type: 'object',
      properties: {
        debateId: {
          type: 'string',
          description: 'Which debate to advance'
        }
      },
      required: ['debateId']
    }
  },

  {
    name: 'calculate_consensus',
    description: `Calculate consensus from agent responses via majority voting.

EDUCATIONAL NOTE:
Consensus uses simple majority voting on the extracted answers:
- Count votes for each unique answer
- Majority answer (most votes) → "chosen" response for DPO
- Minority answers → "rejected" responses for DPO

Consensus strength = (votes for majority) / (total agents)
- High strength (>0.7): Strong agreement, high-quality signal
- Medium strength (0.5-0.7): Moderate agreement, usable signal
- Low strength (<0.5): Weak agreement, question may be ambiguous

You can calculate consensus for any round, not just the final one.
This helps analyze how consensus evolved across rounds (convergence).

Returns: Vote distribution, majority/minority reasoning, consensus strength`,
    inputSchema: {
      type: 'object',
      properties: {
        debateId: {
          type: 'string',
          description: 'Which debate to calculate consensus for'
        },
        round: {
          type: 'number',
          description: 'Which round to analyze (default: current round)'
        }
      },
      required: ['debateId']
    }
  },

  {
    name: 'get_debate_history',
    description: `Retrieve the full debate history including all rounds and responses.

EDUCATIONAL NOTE:
Use this to see previous responses when constructing prompts for later rounds.

The history includes:
- Original question
- All participating agents
- All responses organized by round
- Current round number
- Completion status
- Calculated consensus (if calculate_consensus was called)

Example use case - Constructing round 2 prompt:
1. history = get_debate_history(debate_id)
2. round1_responses = history.responses[0]
3. Build prompt: "Question: {question}\\n\\nRound 1 responses:\\n{round1_responses}\\n\\nNow refine your answer:"
4. Send to LLM
5. Submit via submit_response`,
    inputSchema: {
      type: 'object',
      properties: {
        debateId: {
          type: 'string',
          description: 'Which debate to retrieve'
        }
      },
      required: ['debateId']
    }
  },

  {
    name: 'export_training_data',
    description: `Export debate results as DPO training pairs.

EDUCATIONAL NOTE:
DPO (Direct Preference Optimization) trains models to prefer "chosen" over "rejected" responses.

From each debate, we create training pairs:
- Prompt: The original question
- Chosen: Reasoning from majority voters
- Rejected: Reasoning from minority voters

If 5 agents voted 3-2, we get:
- 2 training pairs (one per minority vote)
- Both pairs have the same chosen response (majority reasoning)
- Each pair has a different rejected response (minority reasoning)

These pairs are used with the HuggingFace TRL library for DPO fine-tuning.

Format options:
- json: Single JSON array of all pairs
- jsonl: One pair per line (better for large datasets)
- huggingface: HuggingFace Dataset format

Returns: Array of {id, prompt, chosen, rejected, metadata} objects`,
    inputSchema: {
      type: 'object',
      properties: {
        debateId: {
          type: 'string',
          description: 'Which debate to export'
        },
        format: {
          type: 'string',
          enum: ['json', 'jsonl', 'huggingface'],
          description: 'Export format (default: json)',
          default: 'json'
        }
      },
      required: ['debateId']
    }
  },

  {
    name: 'list_debates',
    description: `List all debate sessions (active and completed).

Returns summary info for each debate:
- ID, question, participating agents
- Current round, max rounds
- Completion status
- Metadata

Useful for tracking multiple concurrent debates or reviewing past debates.`,
    inputSchema: {
      type: 'object',
      properties: {},
      required: []
    }
  },

  {
    name: 'get_debate_stats',
    description: `Calculate statistics for a debate.

EDUCATIONAL NOTE:
Stats help evaluate debate quality:
- Total responses, unique answers
- Consensus strength across rounds
- Convergence: Did agents move toward agreement?
- Response lengths (detect verbose or terse agents)

Convergence = true when final round has higher consensus than round 1.
This validates the MACA hypothesis: multi-round debates improve agreement.

Use these stats to filter training data:
- Keep high-consensus debates (>0.7 strength)
- Discard low-consensus debates (<0.5) as ambiguous
- Prioritize converged debates for training`,
    inputSchema: {
      type: 'object',
      properties: {
        debateId: {
          type: 'string',
          description: 'Which debate to analyze'
        }
      },
      required: ['debateId']
    }
  },

  {
    name: 'get_all_training_pairs',
    description: `Get all accumulated DPO training pairs from all completed debates.

After running multiple debates and exporting each one, this returns the complete
training dataset ready for fine-tuning.

Returns: Array of all exported DPO pairs across all debates`,
    inputSchema: {
      type: 'object',
      properties: {},
      required: []
    }
  }
];

/**
 * Register the list_tools handler
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools };
});

/**
 * Register the call_tool handler
 *
 * EDUCATIONAL NOTE:
 * ==================
 * This is where the actual tool logic lives. When Claude calls a tool,
 * this handler routes the request to the appropriate method on debateManager.
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'connect_llm': {
        const { id, name, endpoint, model } = args as {
          id: string;
          name: string;
          endpoint?: string;
          model: string;
        };

        const agent = debateManager.registerAgent(id, name, endpoint, model);

        return {
          content: [
            {
              type: 'text',
              text: `✅ Registered agent: ${agent.name} (${agent.id})\n` +
                    `Model: ${agent.model}\n` +
                    `Endpoint: ${agent.endpoint || 'default'}\n\n` +
                    `Agent is ready to participate in debates. ` +
                    `Register ${2} more agents for a 3-agent debate.`
            }
          ]
        };
      }

      case 'start_debate': {
        const { id, question, agentIds, maxRounds, metadata } = args as {
          id: string;
          question: string;
          agentIds: string[];
          maxRounds?: number;
          metadata?: Record<string, any>;
        };

        const session = debateManager.startDebate(id, question, agentIds, maxRounds, metadata);

        return {
          content: [
            {
              type: 'text',
              text: `🎯 Started debate: ${session.id}\n\n` +
                    `**Question:** ${session.question}\n\n` +
                    `**Agents:** ${session.agents.map(a => a.name).join(', ')} (${session.agents.length} total)\n` +
                    `**Max rounds:** ${session.maxRounds}\n` +
                    `**Current round:** ${session.currentRound}\n\n` +
                    `**Next steps:**\n` +
                    `1. For each agent, construct a prompt with the question\n` +
                    `2. Call Ollama to get the agent's response\n` +
                    `3. Extract the final answer from the response\n` +
                    `4. Use submit_response to record each agent's reasoning + answer\n` +
                    `5. After all agents respond, call advance_round\n` +
                    `6. Repeat for round 2 with peer responses included in prompts`
            }
          ]
        };
      }

      case 'submit_response': {
        const { debateId, agentId, reasoning, answer, confidence } = args as {
          debateId: string;
          agentId: string;
          reasoning: string;
          answer: string;
          confidence?: number;
        };

        debateManager.submitResponse(debateId, agentId, reasoning, answer, confidence);

        const session = debateManager.getDebateHistory(debateId);
        const currentRoundResponses = session.responses[session.currentRound - 1] || [];
        const remaining = session.agents.length - currentRoundResponses.length;

        return {
          content: [
            {
              type: 'text',
              text: `📝 Response recorded for ${agentId}\n\n` +
                    `**Answer:** ${answer}\n` +
                    `${confidence !== undefined ? `**Confidence:** ${(confidence * 100).toFixed(1)}%\n` : ''}` +
                    `**Round:** ${session.currentRound}/${session.maxRounds}\n` +
                    `**Responses this round:** ${currentRoundResponses.length}/${session.agents.length}\n\n` +
                    (remaining > 0
                      ? `⏳ Waiting for ${remaining} more response(s) before advancing.`
                      : `✅ All agents responded! Call advance_round to continue.`)
            }
          ]
        };
      }

      case 'advance_round': {
        const { debateId } = args as { debateId: string };

        debateManager.advanceRound(debateId);

        const session = debateManager.getDebateHistory(debateId);

        if (session.completed) {
          return {
            content: [
              {
                type: 'text',
                text: `✅ Debate completed after ${session.maxRounds} rounds!\n\n` +
                      `**Next steps:**\n` +
                      `1. Call calculate_consensus to determine majority/minority answers\n` +
                      `2. Call export_training_data to generate DPO pairs\n` +
                      `3. Use get_debate_stats to analyze convergence and quality`
              }
            ]
          };
        } else {
          return {
            content: [
              {
                type: 'text',
                text: `➡️ Advanced to round ${session.currentRound}/${session.maxRounds}\n\n` +
                      `**Next steps:**\n` +
                      `1. Call get_debate_history to see previous responses\n` +
                      `2. For each agent, construct a prompt including:\n` +
                      `   - The original question\n` +
                      `   - All round ${session.currentRound - 1} responses from peers\n` +
                      `   - Request to refine reasoning based on peer feedback\n` +
                      `3. Submit responses for round ${session.currentRound}`
              }
            ]
          };
        }
      }

      case 'calculate_consensus': {
        const { debateId, round } = args as {
          debateId: string;
          round?: number;
        };

        const consensus = debateManager.calculateConsensus(debateId, round);

        const voteBreakdown = Object.entries(consensus.voteDistribution)
          .map(([answer, votes]) => `  ${answer}: ${votes} vote(s)`)
          .join('\n');

        return {
          content: [
            {
              type: 'text',
              text: `📊 Consensus calculated\n\n` +
                    `**Majority answer:** ${consensus.majorityAnswer}\n` +
                    `**Consensus strength:** ${(consensus.consensusStrength * 100).toFixed(1)}%\n` +
                    `**Total agents:** ${consensus.totalAgents}\n\n` +
                    `**Vote distribution:**\n${voteBreakdown}\n\n` +
                    `**Majority reasoning:** ${consensus.majorityReasoning.length} response(s)\n` +
                    `**Minority reasoning:** ${consensus.minorityReasoning.length} response(s)\n\n` +
                    `${consensus.minorityReasoning.length > 0
                      ? `This will create ${consensus.minorityReasoning.length} DPO training pair(s).`
                      : `⚠️ Unanimous consensus - no minority responses for DPO pairs.`}`
            }
          ]
        };
      }

      case 'get_debate_history': {
        const { debateId } = args as { debateId: string };

        const session = debateManager.getDebateHistory(debateId);

        // Format responses for display
        const formattedHistory = session.responses.map((roundResponses, idx) => {
          const roundNum = idx + 1;
          const responses = roundResponses.map(r =>
            `    ${r.agentId}: "${r.answer}" (${r.reasoning.slice(0, 100)}...)`
          ).join('\n');

          return `  Round ${roundNum}:\n${responses}`;
        }).join('\n\n');

        return {
          content: [
            {
              type: 'text',
              text: `📜 Debate history for ${session.id}\n\n` +
                    `**Question:** ${session.question}\n\n` +
                    `**Agents:** ${session.agents.map(a => a.name).join(', ')}\n` +
                    `**Current round:** ${session.currentRound}/${session.maxRounds}\n` +
                    `**Status:** ${session.completed ? 'Completed' : 'Active'}\n\n` +
                    `**Responses:**\n${formattedHistory}\n\n` +
                    (session.consensus
                      ? `**Consensus:** ${session.consensus.majorityAnswer} (${(session.consensus.consensusStrength * 100).toFixed(1)}% agreement)`
                      : ``)
            }
          ]
        };
      }

      case 'export_training_data': {
        const { debateId, format } = args as {
          debateId: string;
          format?: ExportFormat;
        };

        const pairs = debateManager.exportTrainingData(debateId, format);

        return {
          content: [
            {
              type: 'text',
              text: `💾 Exported ${pairs.length} DPO training pair(s)\n\n` +
                    `**Format:** ${format || 'json'}\n` +
                    `**Debate:** ${debateId}\n` +
                    `**Consensus strength:** ${(pairs[0]?.consensusStrength * 100 || 0).toFixed(1)}%\n\n` +
                    `**Training pairs:**\n` +
                    pairs.map((p, idx) => (
                      `  ${idx + 1}. ${p.id}\n` +
                      `     Prompt: ${p.prompt.slice(0, 80)}...\n` +
                      `     Chosen: ${p.chosen.slice(0, 80)}...\n` +
                      `     Rejected: ${p.rejected.slice(0, 80)}...`
                    )).join('\n\n') +
                    `\n\n✅ Pairs added to training dataset. Call get_all_training_pairs to retrieve complete dataset.`
            }
          ]
        };
      }

      case 'list_debates': {
        const debates = debateManager.listDebates();

        if (debates.length === 0) {
          return {
            content: [
              {
                type: 'text',
                text: `No debates found. Start a debate with start_debate.`
              }
            ]
          };
        }

        const debateList = debates.map(d => (
          `**${d.id}**\n` +
          `  Question: ${d.question.slice(0, 80)}...\n` +
          `  Agents: ${d.agents.length}\n` +
          `  Round: ${d.currentRound}/${d.maxRounds}\n` +
          `  Status: ${d.completed ? '✅ Completed' : '🔄 Active'}`
        )).join('\n\n');

        return {
          content: [
            {
              type: 'text',
              text: `📋 Debates (${debates.length} total)\n\n${debateList}`
            }
          ]
        };
      }

      case 'get_debate_stats': {
        const { debateId } = args as { debateId: string };

        const stats = debateManager.calculateStats(debateId);

        return {
          content: [
            {
              type: 'text',
              text: `📈 Statistics for ${debateId}\n\n` +
                    `**Total responses:** ${stats.totalResponses}\n` +
                    `**Unique answers:** ${stats.uniqueAnswers}\n` +
                    `**Avg consensus strength:** ${(stats.avgConsensusStrength * 100).toFixed(1)}%\n` +
                    `**Convergence:** ${stats.converged ? '✅ Yes (improved over rounds)' : '❌ No'}\n\n` +
                    `**Response lengths:**\n` +
                    `  Average: ${Math.round(stats.avgResponseLength)} characters\n` +
                    `  Min: ${stats.minResponseLength} characters\n` +
                    `  Max: ${stats.maxResponseLength} characters\n\n` +
                    (stats.converged
                      ? `✅ Good quality debate - agents converged toward consensus.`
                      : `⚠️ Agents did not converge - question may be ambiguous or need more rounds.`)
            }
          ]
        };
      }

      case 'get_all_training_pairs': {
        const allPairs = debateManager.getAllTrainingPairs();

        if (allPairs.length === 0) {
          return {
            content: [
              {
                type: 'text',
                text: `No training pairs exported yet. Complete debates and call export_training_data.`
              }
            ]
          };
        }

        // Group by debate ID
        const byDebate = allPairs.reduce((acc, pair) => {
          if (!acc[pair.debateId]) acc[pair.debateId] = [];
          acc[pair.debateId].push(pair);
          return acc;
        }, {} as Record<string, typeof allPairs>);

        const summary = Object.entries(byDebate).map(([debateId, pairs]) => (
          `  ${debateId}: ${pairs.length} pair(s)`
        )).join('\n');

        return {
          content: [
            {
              type: 'text',
              text: `💾 Training Dataset Summary\n\n` +
                    `**Total pairs:** ${allPairs.length}\n` +
                    `**From debates:** ${Object.keys(byDebate).length}\n\n` +
                    `**Breakdown:**\n${summary}\n\n` +
                    `✅ Dataset ready for DPO fine-tuning with HuggingFace TRL.`
            }
          ]
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);

    return {
      content: [
        {
          type: 'text',
          text: `❌ Error: ${errorMessage}`
        }
      ],
      isError: true
    };
  }
});

/**
 * Start the MCP server
 */
async function main() {
  console.error('🚀 MACA Debate MCP Server starting...');
  console.error('📚 Educational comments enabled - see source code for learning notes');
  console.error('');
  console.error('Available tools:');
  console.error('  1. connect_llm - Register Ollama agents');
  console.error('  2. start_debate - Initialize debate session');
  console.error('  3. submit_response - Submit agent reasoning');
  console.error('  4. advance_round - Move to next round');
  console.error('  5. calculate_consensus - Determine majority');
  console.error('  6. get_debate_history - View all responses');
  console.error('  7. export_training_data - Generate DPO pairs');
  console.error('  8. list_debates - List all sessions');
  console.error('  9. get_debate_stats - Analyze debate quality');
  console.error('  10. get_all_training_pairs - Get full dataset');
  console.error('');

  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error('✅ Server ready and listening for MCP requests\n');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
