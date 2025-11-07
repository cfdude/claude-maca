#!/bin/bash
#
# MACA Debate Plugin Installation Script
#
# Installs agents, skills, hooks, and MCP server to target project
#

set -euo pipefail

PLUGIN_NAME="maca-debate"
PLUGIN_VERSION="1.0.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
info() {
  echo -e "${BLUE}ℹ${NC} $1"
}

success() {
  echo -e "${GREEN}✓${NC} $1"
}

warning() {
  echo -e "${YELLOW}⚠${NC} $1"
}

error() {
  echo -e "${RED}✗${NC} $1"
}

# Determine target directory
if [[ -n "${CLAUDE_PROJECT_DIR:-}" ]]; then
  TARGET_DIR="$CLAUDE_PROJECT_DIR"
else
  TARGET_DIR=$(pwd)
fi

PLUGIN_SOURCE=$(dirname "$(dirname "$(realpath "$0")")")

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  MACA Debate Plugin Installer v${PLUGIN_VERSION}"
echo "═══════════════════════════════════════════════════════"
echo ""
info "Plugin source: $PLUGIN_SOURCE"
info "Target directory: $TARGET_DIR"
echo ""

# Verify target directory
if [[ ! -d "$TARGET_DIR/.claude" ]]; then
  error "No .claude directory found in target. Is this a Claude Code project?"
  exit 1
fi

# Backup existing files
info "Creating backup..."
BACKUP_DIR="$TARGET_DIR/.claude/backup-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [[ -d "$TARGET_DIR/.claude/agents" ]]; then
  cp -r "$TARGET_DIR/.claude/agents" "$BACKUP_DIR/"
fi
if [[ -d "$TARGET_DIR/.claude/skills" ]]; then
  cp -r "$TARGET_DIR/.claude/skills" "$BACKUP_DIR/"
fi
if [[ -d "$TARGET_DIR/.claude/hooks" ]]; then
  cp -r "$TARGET_DIR/.claude/hooks" "$BACKUP_DIR/"
fi

success "Backup created at $BACKUP_DIR"

# Install agents
info "Installing agents..."
mkdir -p "$TARGET_DIR/.claude/agents"
cp "$PLUGIN_SOURCE/agents/"*.md "$TARGET_DIR/.claude/agents/"
success "Installed 3 agents: debate-orchestrator, dpo-trainer, dataset-curator"

# Install skills
info "Installing skills..."
mkdir -p "$TARGET_DIR/.claude/skills"
cp -r "$PLUGIN_SOURCE/skills/"* "$TARGET_DIR/.claude/skills/"
success "Installed 3 skills: run-debate, export-training-data, analyze-consensus"

# Install hooks
info "Installing hooks..."
mkdir -p "$TARGET_DIR/.claude/hooks"
cp "$PLUGIN_SOURCE/hooks/hooks.json" "$TARGET_DIR/.claude/hooks/"
cp "$PLUGIN_SOURCE/hooks/"*.sh "$TARGET_DIR/.claude/hooks/"
chmod +x "$TARGET_DIR/.claude/hooks/"*.sh
success "Installed hooks with post-debate-export automation"

# Check if MCP server is already configured
MCP_CONFIGURED=false
if [[ -f "$TARGET_DIR/.mcp.json" ]]; then
  if grep -q "maca-debate" "$TARGET_DIR/.mcp.json"; then
    MCP_CONFIGURED=true
  fi
fi

if [[ "$MCP_CONFIGURED" == "true" ]]; then
  warning "MACA Debate MCP server already configured in .mcp.json"
else
  info "Installing MCP server configuration..."

  # Copy MCP server files
  mkdir -p "$TARGET_DIR/mcp-server"
  cp -r "$PLUGIN_SOURCE/../mcp-server/"* "$TARGET_DIR/mcp-server/"

  # Bundle MCP configuration
  if [[ -f "$TARGET_DIR/.mcp.json" ]]; then
    # Merge with existing config
    jq -s '.[0] * .[1]' "$TARGET_DIR/.mcp.json" "$PLUGIN_SOURCE/.mcp.json" > "$TARGET_DIR/.mcp.json.tmp"
    mv "$TARGET_DIR/.mcp.json.tmp" "$TARGET_DIR/.mcp.json"
  else
    # Create new config
    cp "$PLUGIN_SOURCE/.mcp.json" "$TARGET_DIR/.mcp.json"
  fi

  success "MCP server configuration added to .mcp.json"
fi

# Install MCP server dependencies
if [[ -d "$TARGET_DIR/mcp-server" ]]; then
  info "Installing MCP server dependencies..."
  cd "$TARGET_DIR/mcp-server"

  if command -v npm &> /dev/null; then
    npm install
    npm run build
    success "MCP server built successfully"
  else
    warning "npm not found - skipping MCP server build"
    warning "Run 'cd mcp-server && npm install && npm run build' manually"
  fi

  cd "$TARGET_DIR"
fi

# Create data directories
info "Setting up data directories..."
mkdir -p "$TARGET_DIR/data"
mkdir -p "$TARGET_DIR/data/debates"
mkdir -p "$TARGET_DIR/results"
success "Data directories created"

# Check for Ollama
info "Checking for Ollama..."
if command -v ollama &> /dev/null; then
  OLLAMA_VERSION=$(ollama --version 2>&1 | head -n1)
  success "Ollama found: $OLLAMA_VERSION"

  # Check if model is downloaded
  if ollama list | grep -q "qwen2.5:3b"; then
    success "qwen2.5:3b model already downloaded"
  else
    warning "qwen2.5:3b model not found"
    echo "    Download with: ollama pull qwen2.5:3b"
  fi
else
  warning "Ollama not found"
  echo "    Install from: https://ollama.ai"
fi

# Validation
info "Validating installation..."
VALIDATION_PASSED=true

# Check agents
if [[ ! -f "$TARGET_DIR/.claude/agents/debate-orchestrator.md" ]]; then
  error "Agent installation failed"
  VALIDATION_PASSED=false
fi

# Check skills
if [[ ! -d "$TARGET_DIR/.claude/skills/run-debate" ]]; then
  error "Skills installation failed"
  VALIDATION_PASSED=false
fi

# Check hooks
if [[ ! -f "$TARGET_DIR/.claude/hooks/hooks.json" ]]; then
  error "Hooks installation failed"
  VALIDATION_PASSED=false
fi

# Check MCP server
if [[ ! -f "$TARGET_DIR/mcp-server/dist/index.js" ]] && [[ ! -f "$TARGET_DIR/mcp-server/src/index.ts" ]]; then
  warning "MCP server not built (run 'npm run build' in mcp-server/)"
fi

if [[ "$VALIDATION_PASSED" == "true" ]]; then
  success "Installation validation passed"
else
  error "Installation validation failed - check errors above"
  exit 1
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Installation Complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
success "MACA Debate plugin installed successfully"
echo ""
info "Components installed:"
echo "  • 3 agents (debate-orchestrator, dpo-trainer, dataset-curator)"
echo "  • 3 skills (run-debate, export-training-data, analyze-consensus)"
echo "  • 1 hook (post-debate-export)"
echo "  • MCP server (maca-debate)"
echo ""
info "Next steps:"
echo "  1. Ensure Ollama is running: ollama serve"
echo "  2. Download model (if needed): ollama pull qwen2.5:3b"
echo "  3. Register agents: Use debate-orchestrator agent"
echo "  4. Run a test debate: /run-debate \"Your question\""
echo ""
info "Documentation:"
echo "  • README: $TARGET_DIR/plugin/README.md"
echo "  • Skills: $TARGET_DIR/.claude/skills/*/SKILL.md"
echo "  • Agents: $TARGET_DIR/.claude/agents/*.md"
echo ""
success "Happy debating! 🎯"
echo ""
