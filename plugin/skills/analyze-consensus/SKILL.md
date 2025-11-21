# Analyze Consensus Skill

## Purpose

Analyze debate quality, convergence patterns, and training dataset statistics to optimize MACA workflows.

## Usage

```bash
# Summary of all debates
/analyze-consensus --summary

# Specific debate analysis
/analyze-consensus --debate debate_001

# Convergence trends
/analyze-consensus --convergence-report

# Quality distribution
/analyze-consensus --quality-distribution
```

## Summary Report

```
📊 MACA Debate Analysis Summary

Total Debates: 120
Time Period: 2025-11-01 to 2025-11-07

═══════════════════════════════════════
CONSENSUS METRICS
═══════════════════════════════════════
Average Consensus (Round 1): 58.3%
Average Consensus (Round 2): 75.1%
Improvement: +16.8%

Distribution:
  - High (≥0.7): 78 debates (65%)
  - Medium (0.5-0.7): 32 debates (27%)
  - Low (<0.5): 10 debates (8%)

═══════════════════════════════════════
CONVERGENCE ANALYSIS
═══════════════════════════════════════
Converged: 102/120 (85%)
Diverged: 12/120 (10%)
Stable: 6/120 (5%)

Average Improvement: +16.8%
Best Convergence: debate_045 (+50%)
Worst: debate_089 (-15%)

═══════════════════════════════════════
TRAINING SIGNAL
═══════════════════════════════════════
Total DPO Pairs: 287
  - High Quality (≥0.7): 198 (69%)
  - Medium Quality: 76 (26%)
  - Unanimous (no signal): 13 (5%)

Unique Answers per Debate: 2.1 avg
Response Length: 842 chars avg

═══════════════════════════════════════
RECOMMENDATIONS
═══════════════════════════════════════
✅ Convergence rate excellent (85%)
✅ High-quality pairs abundant (198)
⚠️  10 debates below 0.5 - review questions
⚠️  13 unanimous debates - too easy?

Next Steps:
1. Export 198 high-quality pairs
2. Review 10 low-consensus questions
3. Rephrase or discard ambiguous questions
4. Add more expert-level questions
```

## Visualization

Generates charts (when in web interface):
- Consensus distribution histogram
- Convergence trend line
- Quality heatmap by category
- Response length distribution

## Quality Metrics

Tracks:
- Consensus strength (0-1)
- Convergence rate (%)
- Unique answers per debate
- Response lengths
- Time per debate
- DPO pairs generated

## See Also
- `/run-debate` - Execute debates
- `/export-training-data` - Export filtered pairs
