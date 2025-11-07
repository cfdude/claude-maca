# Next Steps: MACA Plugin Development

## Immediate Actions (This Week)

### 1. Run Image Extraction
```bash
cd /Users/robsherman/Downloads/maca

# Install Python dependencies
pip install -r requirements.txt

# Note: If on macOS, install poppler for PDF conversion
brew install poppler

# Run extraction script
python scripts/extract_pdf.py
```

This will:
- Extract all embedded images from the PDF
- Convert key figure pages to images
- Create `images_metadata.json` with descriptions
- Extract full text to `extracted_data/full_text.txt`

### 2. Review Extracted Content
- Check `images/` directory for extracted figures
- Review `images/figures/` for converted pages
- Open `docs/concepts/01_core_framework.md` to understand MACA
- Read `docs/concepts/02_claude_code_implementation_plan.md` for architecture

### 3. Enhance Image Metadata
Open `images/images_metadata.json` and add descriptions:

```json
{
  "filename": "page_4_img_1.png",
  "page": 4,
  "description": "MACA Framework diagram showing multi-agent debate flow",
  "figure_number": "Figure 1",
  "caption": "Multi-Agent Consensus Alignment framework: Multiple clones of a base LM engage in debate..."
}
```

## Short-Term Goals (Next 2 Weeks)

### Phase 1: Complete Documentation Extraction

1. **Create Detailed Markdown Files** for:
   - Algorithm 1 (Multi-Agent Consensus Alignment)
   - Experimental results (Tables 1-22)
   - Training curves (Figures 7-8)
   - Key equations and formalizations

2. **Image Integration**:
   - Add figure references to markdown files
   - Create footnotes explaining each figure
   - Link figures to relevant sections

3. **Vector Database Setup** (Optional):
   ```bash
   # If you want to use your knowledge store MCP
   # Create project in vector DB
   claude mcp call knowledge-store create_project \
     --name "maca-research" \
     --description "Multi-Agent Consensus Alignment research"

   # Index extracted markdown
   claude mcp call knowledge-store index_documents \
     --project "maca-research" \
     --documents docs/**/*.md
   ```

### Phase 2: Prototype MCP Server

1. **Initialize Node.js Project**:
   ```bash
   mkdir -p mcp-server
   cd mcp-server
   npm init -y
   npm install @modelcontextprotocol/sdk
   ```

2. **Create Basic Tools**:
   - `start_debate` - Initialize a multi-agent debate
   - `submit_response` - Agent submits reasoning
   - `calculate_consensus` - Determine majority answer

3. **Test Locally**:
   ```bash
   # Start MCP server
   npm run dev

   # Test with Claude CLI
   claude mcp call maca start_debate \
     --question "Test question" \
     --num_agents 3
   ```

## Medium-Term Goals (Weeks 3-6)

### Implement Core Debate Logic

1. **Debate State Management**:
   - Track multiple concurrent debates
   - Store round-by-round responses
   - Calculate agreement metrics

2. **Agent Pool**:
   - Manage multiple Claude instances
   - Prepare peer context for each round
   - Optimize parallel execution

3. **Consensus Algorithms**:
   - Implement majority vote calculation
   - Calculate sampling consistency metrics
   - Partition responses into G+ and G-

### Create Skills

1. **Debate Skills** (`skills/` directory):
   - `initiate-debate.md` - First round template
   - `participate-round.md` - Subsequent rounds
   - `aggregate-consensus.md` - Final synthesis

2. **Integration Skills**:
   - `code-review-debate.md` - Code review consensus
   - `architecture-debate.md` - Design decisions
   - `debug-consensus.md` - Error diagnosis

### Testing

1. **Unit Tests**:
   - Consensus calculation
   - Response parsing
   - Agreement metrics

2. **Integration Tests**:
   - Full debate flow
   - Multi-round evolution
   - Error handling

3. **Benchmark Tasks**:
   - Math problems (GSM8K-style)
   - Code review scenarios
   - Architecture decisions

## Long-Term Goals (Weeks 7-12)

### Production-Ready Plugin

1. **Package as Claude Code Plugin**:
   ```json
   {
     "name": "@your-org/claude-maca-plugin",
     "version": "0.1.0",
     "description": "Multi-Agent Consensus Alignment for Claude Code",
     "skills": ["skills/*.md"],
     "hooks": ["hooks/*.sh"],
     "mcp-servers": ["mcp-server/"],
     "agents": ["agents/*.md"]
   }
   ```

2. **Documentation**:
   - User guide
   - API reference
   - Best practices
   - Example scenarios

3. **Performance Optimization**:
   - Parallel agent execution
   - Caching strategies
   - Cost optimization

### Advanced Features

1. **Iterative Training** (Optional):
   - Collect debate outcomes
   - Prepare preference datasets
   - Export for model fine-tuning

2. **Specialized Agents**:
   - Security reviewer
   - Performance optimizer
   - Maintainability advocate

3. **Analytics Dashboard**:
   - Consensus metrics over time
   - Accuracy vs. agreement correlation
   - Debate efficiency statistics

### Deployment

1. **Publishing**:
   - Submit to Claude Code marketplace
   - Create demo video
   - Write blog post

2. **Community**:
   - GitHub repository
   - Documentation site
   - Example projects

## Decision Points

### Should You Use Vector Database?

**Pros**:
- Semantic search over research content
- Easy retrieval of relevant sections
- Can query "How does MACA handle X?"

**Cons**:
- Additional complexity
- Well-organized markdown might be sufficient
- Claude Code can already read markdown files

**Recommendation**: Start with markdown files. Add vector DB later if you find yourself frequently searching for specific concepts.

### Should You Implement Iterative Training?

**Pros**:
- Could improve model performance over time
- Collect valuable debate data

**Cons**:
- Requires ML infrastructure
- Complex to implement correctly
- May not be necessary for initial value

**Recommendation**: Focus on inference-time consensus first. Iterative training is an advanced feature for later.

### Scope of Initial Release

**Minimal Viable Plugin** (4-6 weeks):
- MCP server with basic debate orchestration
- 2-3 core skills for different use cases
- Simple pre-commit hook
- Documentation

**Full-Featured Plugin** (10-12 weeks):
- All planned features
- Multiple specialized agents
- Comprehensive testing
- Analytics and monitoring

**Recommendation**: Start with MVP, get user feedback, iterate.

## Project Types That Would Benefit Most

Based on MACA research, ideal projects have:

1. **Complex Decision-Making**:
   - Architecture choices with trade-offs
   - Design pattern selection
   - Technology stack decisions

2. **High-Stakes Changes**:
   - Security-critical code
   - Performance-sensitive systems
   - Production deployments

3. **Ambiguous Problems**:
   - Multiple valid solutions
   - Subjective quality criteria
   - Novel problem domains

4. **Code Quality Focus**:
   - Thorough code review processes
   - High testing standards
   - Strong documentation requirements

**Less Suitable** for:
- Simple CRUD applications
- Straightforward implementations
- Time-sensitive hotfixes
- Solo hobby projects (overhead not worth it)

## Success Criteria

### Week 2 (Extraction Complete)
- [x] Project structure initialized
- [ ] All images extracted and documented
- [ ] Core concepts documented in markdown
- [ ] Implementation plan created

### Week 6 (MVP Working)
- [ ] MCP server running locally
- [ ] Can execute 3-agent, 2-round debate
- [ ] Consensus calculated correctly
- [ ] At least one skill working

### Week 12 (Production Ready)
- [ ] Published as Claude Code plugin
- [ ] Full documentation
- [ ] Test coverage >80%
- [ ] At least 3 example use cases
- [ ] User guide and best practices

## Resources Needed

### Development
- **Time**: 1 engineer, 12 weeks for full implementation
- **Skills**: TypeScript, Claude Code plugin architecture, MCP
- **Tools**: Node.js, Claude CLI, testing frameworks

### Infrastructure
- **Hosting**: MCP server (minimal - can run locally)
- **CI/CD**: GitHub Actions for testing
- **Monitoring**: Optional - debate metrics tracking

### Costs
- **Development**: Your time
- **Runtime**: ~$0.01-0.10 per debate (6 LLM calls)
- **Infrastructure**: Minimal (local or cheap cloud hosting)

## Questions to Answer

Before proceeding, consider:

1. **Primary Use Case**: What's the #1 problem you want to solve with this?
2. **Timeline**: How quickly do you need this working?
3. **Scope**: MVP or full-featured first release?
4. **Resources**: How much time can you dedicate?
5. **Users**: Just you, your team, or public release?

## Getting Help

- **MACA Paper**: Reference the extracted docs in `docs/concepts/`
- **Claude Code Docs**: https://docs.claude.com/en/docs/claude-code
- **MCP SDK**: https://github.com/modelcontextprotocol/sdk
- **Plugin Examples**: Check Claude Code marketplace

---

## Recommended Path Forward

1. **This Week**: Run extraction script, review generated docs
2. **Week 2**: Plan your specific use case, design debate flow
3. **Weeks 3-6**: Build MCP server MVP, test with real scenarios
4. **Weeks 7-12**: Polish, document, publish

**Start small, iterate quickly, get feedback early!**
