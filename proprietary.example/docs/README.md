# Documentation Directory

Use this directory to document your domain-specific training process, results, and learnings.

## Recommended Files

### training_report.md

Document each training run:

```markdown
# Training Run v1 - [Date]

## Dataset
- Size: 50 training pairs, 12 validation pairs
- Categories: contract_review (20), compliance (15), litigation (15)
- Quality: Manual review, all expert-level responses

## Configuration
- Base model: Qwen2.5-3B
- LoRA rank: 16
- Learning rate: 1e-6
- Epochs: 2
- Beta: 0.1

## Results
- Validation accuracy: 87%
- Reward margin: 2.8
- Training time: 12 minutes

## Observations
- Model learned to prefer detailed analysis over brief responses
- Strong performance on contract review questions
- Weaker on compliance edge cases (need more data)

## Next Steps
- Add 30 more compliance examples
- Retrain with r=32, lr=5e-6
```

### evaluation_report.md

Document model evaluation:

```markdown
# Evaluation Report v1

## Quantitative Metrics
- Validation accuracy: 87%
- Test set accuracy: 84%
- Reward margin: 2.8
- No catastrophic forgetting (99% on general knowledge test)

## Qualitative Analysis

### Example 1: Contract Review
**Question**: [...]
**Base model**: Generic, brief response
**Fine-tuned**: Detailed, cites specific clauses
**Rating**: 4/5 (expert-level)

### Example 2: Compliance
**Question**: [...]
**Base model**: Missed key regulation
**Fine-tuned**: Identified relevant regulation, explained implications
**Rating**: 5/5 (exceeds expectations)

## Expert Feedback
- Reviewed by 3 domain experts
- 85% of responses rated "expert-level"
- Main weakness: edge cases in category X

## Recommendations
- Deploy to pilot group
- Gather user feedback
- Plan v2 with expanded dataset
```

### deployment_notes.md

Document deployment process:

```markdown
# Deployment Notes

## Production Setup
- Environment: AWS SageMaker
- Instance: ml.g5.2xlarge
- Model: legal-advisor-v1
- Endpoint: https://...

## Performance
- Latency: 500ms average
- Throughput: 10 requests/sec
- Cost: $2.50/hour

## Monitoring
- CloudWatch metrics
- Error rate < 0.1%
- User satisfaction: 4.2/5

## Rollback Plan
- Keep base model endpoint active
- Can switch traffic instantly
- Gradual rollout: 10% → 50% → 100%
```

### lessons_learned.md

Document insights for future iterations:

```markdown
# Lessons Learned

## What Worked
- Starting conservative (small LR, few epochs) prevented overfitting
- 5 agents in debates provided good diversity
- Filtering unanimous debates improved data quality
- Expert review of training pairs caught quality issues

## What Didn't Work
- Temperature 0.5 in debates: too little diversity
- Training directly on initial questions: needed debate refinement
- Skipping validation split: couldn't detect overfitting early

## Domain-Specific Insights
- Legal reasoning requires higher LoRA rank (32 vs 16)
- Compliance questions need very specific examples
- Contract review benefits from longer context (2048 tokens)

## Process Improvements
- Batch generate debates overnight
- Have experts review debate outputs before training
- Test on held-out cases before deployment
- Maintain separate test set never used in training
```

## Privacy & Security

**Important**: This directory may contain:
- Domain expertise (your competitive advantage)
- Training insights (valuable IP)
- Performance metrics (business sensitive)

Remember:
- This entire `proprietary/` directory is gitignored
- Don't commit to public repos
- Encrypt before sharing with team
- Consider access controls

## Backup

Regularly backup this directory:

```bash
# Create dated backup
tar -czf backup-$(date +%Y%m%d).tar.gz proprietary/docs/

# Or use cloud storage
aws s3 sync proprietary/docs/ s3://your-bucket/maca-docs/
```

## See Also

- `../data/` - Training data and results
- `../models/` - Trained models
- `../results/` - Training artifacts
- `../../docs/` - Public MACA framework docs
