#!/bin/bash
#
# Post-Debate Export Hook
#
# Triggers after export_training_data is called
# Automatically:
# 1. Validates exported pairs
# 2. Updates training dataset
# 3. Logs export metrics
# 4. Suggests next actions

set -euo pipefail

# Get hook input
HOOK_INPUT=$(cat)

# Extract tool use info
TOOL_NAME=$(echo "$HOOK_INPUT" | jq -r '.toolUse.name // "unknown"')
TOOL_ARGS=$(echo "$HOOK_INPUT" | jq -r '.toolUse.arguments // {}')
TOOL_RESULT=$(echo "$HOOK_INPUT" | jq -r '.toolResult.content[0].text // ""')

# Only process export_training_data calls
if [[ "$TOOL_NAME" != "mcp__maca-debate__export_training_data" ]]; then
  echo "{}"
  exit 0
fi

# Extract debate ID
DEBATE_ID=$(echo "$TOOL_ARGS" | jq -r '.debateId // ""')

# Count exported pairs from result
PAIR_COUNT=$(echo "$TOOL_RESULT" | grep -o "Exported [0-9]* DPO" | grep -o "[0-9]*" || echo "0")

# Log to metrics file
METRICS_FILE="${CLAUDE_PLUGIN_ROOT}/data/export_metrics.jsonl"
mkdir -p "$(dirname "$METRICS_FILE")"

echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"debate_id\":\"$DEBATE_ID\",\"pairs_exported\":$PAIR_COUNT}" >> "$METRICS_FILE"

# Calculate cumulative stats
TOTAL_EXPORTS=$(wc -l < "$METRICS_FILE" | tr -d ' ')
TOTAL_PAIRS=$(jq -s 'map(.pairs_exported) | add' "$METRICS_FILE")

# Generate suggestions
SUGGESTIONS=""

if [[ $PAIR_COUNT -eq 0 ]]; then
  SUGGESTIONS="⚠️ No training pairs exported (likely unanimous consensus). Consider using this question for validation set."
elif [[ $PAIR_COUNT -lt 2 ]]; then
  SUGGESTIONS="✅ Low pair count suggests strong consensus. Good for validation."
else
  SUGGESTIONS="✅ Multiple pairs exported - good training signal!"
fi

# Check if we have enough pairs for training
if [[ $TOTAL_PAIRS -ge 300 ]]; then
  SUGGESTIONS="$SUGGESTIONS\n\n🎯 You now have $TOTAL_PAIRS training pairs - ready for DPO fine-tuning!"
elif [[ $TOTAL_PAIRS -ge 100 ]]; then
  SUGGESTIONS="$SUGGESTIONS\n\n📊 Progress: $TOTAL_PAIRS/$((300)) pairs (need $((300 - TOTAL_PAIRS)) more for optimal training)"
fi

# Output JSON response
cat <<EOF
{
  "hookSpecificOutput": {
    "export_metrics": {
      "debate_id": "$DEBATE_ID",
      "pairs_exported": $PAIR_COUNT,
      "total_exports": $TOTAL_EXPORTS,
      "total_pairs": $TOTAL_PAIRS
    },
    "suggestions": "$SUGGESTIONS"
  }
}
EOF
