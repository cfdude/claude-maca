# MACA Research Extraction & Claude Code Plugin Development

## Project Overview

This project extracts and organizes research from "Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment" (MACA) to develop a Claude Code plugin that implements multi-agent consensus principles.

## Research Paper Details

- **Title**: Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment
- **Authors**: Ankur Samanta, Akshayaa Magesh, Youliang Yu, et al. (Meta AI, Meta Superintelligence Labs, Columbia University, Cornell Tech)
- **Date**: September 19, 2025
- **arXiv**: 2509.15172v2
- **Code**: https://github.com/facebookresearch/maca

## Core Concept

MACA is a reinforcement learning framework where multiple LM clones collaborate through iterative debate to achieve consensus. The framework:

- Improves self-consistency (+27.6% on GSM8K)
- Enhances single-agent reasoning (+23.7% on MATH)
- Boosts sampling-based inference (+22.4% Pass@20 on MATH)
- Strengthens multi-agent decision-making (+42.7% on MathQA)

## Project Structure

```
maca/
├── README.md                          # This file
├── 2509.15172v2.pdf                  # Original research paper
├── docs/                              # Organized markdown documentation
│   ├── concepts/                      # Core concepts and theory
│   ├── algorithms/                    # Algorithm descriptions
│   ├── experiments/                   # Experimental results
│   └── appendices/                    # Additional materials
├── images/                            # Extracted images from PDF
├── extracted_data/                    # Structured data extractions
├── scripts/                           # Extraction and processing scripts
└── plugin/                            # Claude Code plugin (future)
    ├── skills/                        # Agent skills
    ├── hooks/                         # Integration hooks
    ├── mcp-server/                    # MCP server implementation
    └── agents/                        # Agent definitions
```

## Extraction Strategy

### Phase 1: Content Extraction
- [x] Initialize project structure
- [ ] Extract all images with figure numbers
- [ ] Extract text content organized by sections
- [ ] Preserve mathematical equations in LaTeX format
- [ ] Extract tables and algorithmic descriptions
- [ ] Create image reference system with explanations

### Phase 2: Organization
- [ ] Create markdown files for each major concept
- [ ] Document MACA framework components
- [ ] Extract training methodology
- [ ] Document experimental results
- [ ] Create Claude Code application notes

### Phase 3: Plugin Design
- [ ] Map MACA principles to Claude Code architecture
- [ ] Design multi-agent debate system
- [ ] Create consensus mechanism
- [ ] Implement preference learning components
- [ ] Build MCP server for agent coordination

## Key Research Insights

### 1. Self-Consistency as Intrinsic Property
MACA formalizes self-consistency as a core trait of well-aligned reasoning models.

### 2. Multi-Agent Debate Framework
- **M agents** (clones) engage in **R rounds** of debate
- Agents share reasoning and update through peer feedback
- Majority consensus used for training signal

### 3. Training Objectives
Four post-training methods:
- **MV-SFT**: Majority-Vote Supervised Fine-Tuning
- **MV-GRPO**: Majority-Vote Group Relative Policy Optimization
- **MV-DPO**: Majority-Vote Direct Preference Optimization
- **MV-KTO**: Majority-Vote Kahneman-Tversky Optimization

### 4. Applicability to Claude Code
- Plugin architecture supports multiple agent instances
- Skills can encapsulate debate behaviors
- MCP servers can coordinate agent communication
- Hooks can trigger consensus mechanisms

## Use Cases for Claude Code

Projects that would benefit from MACA-based plugin:
1. **Complex reasoning tasks** requiring multiple perspectives
2. **Code review systems** with multi-agent validation
3. **Architecture decision-making** through consensus
4. **Test generation** with diverse approaches
5. **Documentation review** with multiple validation passes
6. **Refactoring proposals** evaluated by multiple agents

## Next Steps

1. Complete content extraction
2. Organize into digestible markdown
3. Create vector database entries (optional)
4. Design plugin architecture
5. Implement core components
6. Test with sample Claude Code project

## Notes

- Mathematical equations will be preserved in LaTeX format
- Images will be extracted with descriptive footnotes
- Citations will be preserved with paper references
- Code examples from the research will be adapted for Claude Code

---

**Last Updated**: 2025-11-06
