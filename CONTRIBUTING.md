# Contributing to MACA

Thank you for your interest in contributing to the Multi-Agent Consensus Alignment (MACA) framework!

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Documentation Guidelines](#documentation-guidelines)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, Ollama version)
- **Relevant logs or error messages**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- Include examples of how it would be used

### Code Contributions

We welcome contributions in these areas:

1. **Domain-Specific Parsers**: Add support for new domains (legal, medical, technical, etc.)
2. **Training Recipes**: Optimize hyperparameters for different dataset sizes
3. **Evaluation Metrics**: Improve quality assessment and consensus analysis
4. **Documentation**: Expand guides, tutorials, and examples
5. **Bug Fixes**: Fix issues and improve stability
6. **Performance**: Optimize debate orchestration and training speed

---

## Development Setup

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/)
- Node.js 18+ (for MCP server)
- Git

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/yourusername/maca.git
cd maca

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Build MCP server
cd mcp-server
npm install
npm run build
cd ..
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=scripts tests/

# Run specific test file
pytest tests/test_parser.py
```

---

## Code Standards

### Python Style Guide

We follow [PEP 8](https://peps.python.org/pep-0008/) with these specific requirements:

**Formatting**:
```bash
# Format code with black (line length: 100)
black --line-length 100 scripts/

# Check linting
flake8 scripts/ --max-line-length 100

# Type checking
mypy scripts/
```

**Code Structure**:
```python
# Good: Clear function names with type hints
def calculate_consensus(
    responses: List[Dict[str, Any]],
    threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Calculate consensus from agent responses using fuzzy matching.

    Args:
        responses: List of agent response dictionaries
        threshold: Similarity threshold for answer grouping (0.0-1.0)

    Returns:
        Dictionary containing majority_answer and consensus_strength
    """
    pass

# Bad: Unclear names, no types, no docstring
def calc(data, t=0.85):
    pass
```

**Error Handling**:
```python
# Good: Specific exceptions with context
try:
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    logger.error(f"Failed to connect to Ollama: {e}")
    raise ConnectionError(f"Ollama service unavailable: {e}")

# Bad: Bare except
try:
    do_something()
except:
    pass
```

### TypeScript/Node.js Style Guide (MCP Server)

**Formatting**:
```bash
# Format with prettier
npm run format

# Lint
npm run lint
```

**Code Structure**:
```typescript
// Good: Clear types and error handling
async function calculateConsensus(
  responses: AgentResponse[],
  threshold: number = 0.85
): Promise<ConsensusResult> {
  if (responses.length === 0) {
    throw new Error("No responses provided for consensus calculation");
  }
  // Implementation...
}

// Bad: Any types, no validation
async function calc(data: any) {
  // Implementation...
}
```

### Configuration Files

- Use JSON for configuration files
- Include schema validation where possible
- Provide example configurations in `examples/configs/`

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Required tests**: All public functions must have unit tests
- **Integration tests**: Required for debate orchestration and training pipelines

### Writing Tests

```python
# tests/test_parser.py
import pytest
from scripts.parser import AnswerParser

def test_financial_normalization():
    """Test that financial amounts are normalized correctly."""
    parser = AnswerParser(domain="financial")

    assert parser.normalize("$1,000") == parser.normalize("1000")
    assert parser.normalize("$1K") == parser.normalize("one thousand")
    assert parser.normalize("$1,000.00") == parser.normalize("1000")

def test_fuzzy_matching_threshold():
    """Test that similarity threshold works correctly."""
    parser = AnswerParser(similarity_threshold=0.85)

    # Should match (high similarity)
    assert parser.are_similar("Yes, I agree", "yes")

    # Should not match (low similarity)
    assert not parser.are_similar("Yes", "No")
```

### Running Tests Before Commit

```bash
# Run all checks
black --check scripts/
flake8 scripts/ --max-line-length 100
mypy scripts/
pytest tests/ --cov=scripts
```

---

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Write clear, focused commits
   - Follow code standards
   - Add tests for new functionality
   - Update documentation

4. **Run all tests**:
   ```bash
   pytest tests/
   black --check scripts/
   flake8 scripts/
   ```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring (no behavior changes)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(parser): add legal domain citation normalization

Implements citation normalization for legal domain that handles:
- U.S.C. variations (42 U.S.C. § 1983, 42 USC 1983)
- Case name variations
- Bluebook citation formats

Closes #123

---

fix(consensus): handle empty response list

Prevents crash when no agents provide responses.
Added validation and meaningful error message.

---

docs(README): add training recipes for small datasets

Provides guidance on:
- Learning rate recommendations
- LoRA configuration
- Early stopping strategies
```

### Submitting the PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Description explaining what and why
   - Link to related issues
   - Screenshots/examples if applicable

3. **PR Checklist**:
   - [ ] Code follows style guidelines
   - [ ] Tests pass locally
   - [ ] New tests added for new functionality
   - [ ] Documentation updated
   - [ ] Commit messages follow format
   - [ ] No merge conflicts with main

### Review Process

- Maintainers will review your PR within 3-5 business days
- Address feedback in new commits (don't force-push during review)
- Once approved, a maintainer will merge your PR

---

## Documentation Guidelines

### Code Documentation

**Python Docstrings** (Google style):
```python
def run_debate(
    question: str,
    num_agents: int = 5,
    max_rounds: int = 2,
    temperature: float = 0.9
) -> Dict[str, Any]:
    """
    Run a multi-agent debate on the given question.

    Orchestrates M agents through R rounds of debate, where agents see
    peer responses after Round 1 and can revise their reasoning.

    Args:
        question: The question to debate
        num_agents: Number of agents (M) to participate
        max_rounds: Number of debate rounds (R)
        temperature: LLM sampling temperature (higher = more diverse)

    Returns:
        Dictionary containing:
            - question: The original question
            - rounds: List of round results
            - consensus: Final consensus calculation
            - metadata: Debate configuration

    Raises:
        ConnectionError: If Ollama service is unavailable
        ValueError: If num_agents < 2 or max_rounds < 1

    Example:
        >>> result = run_debate(
        ...     question="Should we prioritize feature A or B?",
        ...     num_agents=5,
        ...     max_rounds=2
        ... )
        >>> print(result['consensus']['majority_answer'])
        'Feature A'
    """
    pass
```

### Markdown Documentation

- Use clear headings and structure
- Include code examples with syntax highlighting
- Add "Why this matters" context for complex topics
- Link to related documentation

### Example Code

- All examples must be runnable
- Include complete configurations
- Add comments explaining key parameters
- Show expected output

---

## Domain-Specific Contributions

### Adding a New Domain Parser

If you're adding support for a new domain (e.g., legal, medical, technical):

1. **Create parser class** in `scripts/parser.py`:
   ```python
   class LegalParser(BaseDomainParser):
       """Parser for legal domain (citations, case names, etc.)."""

       def normalize(self, answer: str) -> str:
           # Implement domain-specific normalization
           pass
   ```

2. **Add tests** in `tests/test_parser.py`:
   ```python
   def test_legal_citation_normalization():
       parser = AnswerParser(domain="legal")
       assert parser.normalize("42 U.S.C. § 1983") == \
              parser.normalize("42 USC 1983")
   ```

3. **Update documentation**:
   - Add domain to README.md parser section
   - Create example in `examples/domains/legal/`
   - Document normalization rules

4. **Provide example dataset**:
   - Create `examples/datasets/legal_questions.json`
   - Include 5-10 representative questions
   - Show expected consensus patterns

---

## Questions?

- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Email**: your.email@example.com

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to MACA!**
