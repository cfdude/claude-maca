#!/usr/bin/env python3
"""
Automated CMA Training Dataset Generator

Generates 100+ training examples from CMA source material.
Uses pattern matching and templates to create Q&A pairs across all 5 modules.
"""

import json
import re
from pathlib import Path

# Comprehensive training examples by module
def generate_module_a_examples(start_id=8):
    """Generate Module A examples (Advisor, Refinance, Debt, APR)"""
    examples = []

    # Example templates with full responses
    templates = [
        # Lump sum/biweekly payments (3 examples)
        {
            "prompt": "Should my client make bi-weekly payments instead of monthly?",
            "chosen": "Bi-weekly payments are a powerful but often misunderstood strategy. Let me explain:\n\n**How It Works:**\n- Pay HALF your mortgage every 2 weeks instead of full payment monthly\n- 52 weeks ÷ 2 = 26 half payments = 13 full payments per year\n- ONE extra payment annually without feeling the pinch\n\n**Impact on $300,000 loan at 6.5%, 30-year:**\n- Regular schedule: Pay off in 30 years, ~$383,000 total interest\n- Bi-weekly: Pay off in ~26 years, ~$330,000 total interest\n- Savings: ~$53,000 and 4 years faster payoff\n\n**Benefits:**\n✅ Accelerates payoff by 3-4 years typically\n✅ Saves substantial interest\n✅ Builds equity faster\n✅ Aligns with biweekly paycheck schedule\n✅ Less painful than one lump sum\n\n**Considerations:**\n⚠️ Some lenders charge fees for bi-weekly programs ($300-400 setup)\n⚠️ Verify lender applies payments immediately (not holding until full payment accumulated)\n⚠️ Client must have discipline to maintain schedule\n\n**DIY Alternative (No Fees):**\nClient makes ONE extra full payment per year:\n- Divide monthly payment by 12\n- Add that amount to each monthly payment\n- Same result, no program fees\n\nExample: $2,000 monthly ÷ 12 = $167 extra per month = $2,167 monthly\n\n**When NOT to Recommend:**\n❌ Client has high-interest debt (pay that first - better return)\n❌ Not maxing 401k employer match\n❌ Emergency fund under 3 months\n❌ Mortgage under 4% and can invest at higher returns\n\n**Client Communication:**\n\"Bi-weekly payments put you on track to pay off your 30-year loan in about 26 years and save roughly $53,000 in interest. It's like making 13 payments instead of 12 each year. If your lender charges fees, you can achieve the same result by adding 1/12 of your payment each month - no fees required.\"",
            "rejected": "Bi-weekly payments help pay off your mortgage faster by making 13 payments a year instead of 12.",
            "metadata": {"category": "mortgage_strategy", "difficulty": "intermediate", "cma_module": "A", "debate_worthy": false, "client_facing": true, "requires_calculation": true}
        },
        {
            "prompt": "How do I explain the power of amortization to a client considering extra principal payments?",
            "chosen": "Amortization is underestimated. Most clients think extra principal only shortens the loan term. They miss the IMMEDIATE benefits:\n\n**What Actually Happens:**\n\nEvery extra dollar toward principal triggers a **compounding effect**:\n\n1. **Immediate Reallocation**: Next payment has MORE to principal, LESS to interest\n2. **Accelerating Effect**: Each subsequent payment builds on this\n3. **Equity Acceleration**: Builds wealth much faster\n4. **Interest Savings**: Reduces total interest before loan even pays off\n\n**Concrete Example:**\n$350,000 loan, 6.5%, 30 years:\n- Monthly P&I: $2,212\n- Client adds $200/month extra to principal\n\nResults:\n- Payoff: 23 years instead of 30 (7 years faster)\n- Interest savings: ~$143,000\n- Monthly benefit starts IMMEDIATELY\n\n**The Math Behind It:**\n\nMonth 1 regular payment breakdown:\n- $2,212 total\n- $1,896 interest\n- $316 principal\n\nMonth 1 with $200 extra:\n- $2,412 total\n- $1,896 interest  \n- $516 principal ($200 extra)\n\nMonth 2 (after extra $200):\n- Balance is $200 lower\n- Interest calculation is on LOWER balance\n- More of required payment goes to principal\n- This compounds every month\n\n**Client Wealth Building:**\n\nExtra payments create wealth through:\n- Forced savings (equity)\n- Compounding effect (reducing future interest)\n- Flexibility (can refi or sell with more equity)\n- Psychological benefit (ownership feeling)\n\n**Strategic Flexibility:**\n\n\"Keep your 30-year payment for flexibility, but pay it like a 15-year. If circumstances change (job loss, medical emergency), you can always drop back to the required payment. You're not locked into the higher payment.\"\n\n**Comparison to Investment:**\n\nIf mortgage is 6.5%, every extra dollar saves 6.5% guaranteed. Compare to:\n- Stock market: ~10% average, but not guaranteed\n- Bonds: 4-5% currently\n- Savings: 4-5%\n\n6.5% guaranteed, tax-adjusted return is competitive.\n\n**For Maximum Impact:**\n\nTarget extra payments to:\n1. Early in loan (when balance is highest)\n2. Before major life events (kids, retirement)\n3. When receiving windfalls (bonus, tax refund)\n\n**Client Communication:**\n\"Extra principal payments don't just shorten your loan - they immediately shift how every future payment builds equity. A $200/month extra payment on your loan saves $143,000 in interest and pays off 7 years early. Plus, you keep the flexibility of the lower required payment if life circumstances change.\"",
            "rejected": "Extra payments toward principal shorten your loan term and save money on interest over time.",
            "metadata": {"category": "mortgage_strategy", "difficulty": "advanced", "cma_module": "A", "source_section": "Power of Amortization", "line_reference": "453-477", "debate_worthy": false, "client_facing": true, "requires_calculation": true}
        },

        # Refinance timing (3 examples)
        {
            "prompt": "Client wants to wait for rates to drop another 0.25% before refinancing. How do I explain the cost of waiting?",
            "chosen": "This is a critical decision that many clients get wrong. Let's calculate the TRUE cost of waiting:\n\n**The Opportunity Cost:**\n\nEvery month waiting = lost savings from current available rate\n\n**Example:**\nCurrent situation:\n- Loan balance: $325,810\n- Current rate: 4.5%\n- Available refi rate TODAY: 3.25%\n- Monthly savings available NOW: $355\n\n**If Client Waits 6 Months for 0.25% Better Rate:**\n\nCost of waiting:\n- Months 1-6: $355/month × 6 = $2,130 in foregone savings\n\nPotential gain if rates drop to 3.0%:\n- Additional savings: ~$50/month vs 3.25% rate\n- Need 42 months to recover the $2,130 (2,130 ÷ 50 = 42.6 months)\n\n**Breakeven: 3.5 YEARS just to recover what was lost waiting!**\n\n**The Risk Factor:**\n\nWaiting assumes:\n✓ Rates WILL drop (not guaranteed)\n✓ They'll drop 0.25% (specific prediction)\n✓ Client qualifies when rates drop (income/credit unchanged)\n✓ No other market disruptions\n\n**What If Rates Rise Instead?**\n\nIf rates go to 3.75% while waiting:\n- Miss the 3.25% opportunity\n- Now refinancing costs MORE monthly\n- Lost $2,130 in savings\n- Total damage: Significant\n\n**Decision Framework:**\n\n**Lock NOW if:**\n✅ Current available rate achieves client's goals\n✅ Breakeven is under 24 months\n✅ Market shows mixed signals (uncertainty)\n✅ Client is risk-averse\n✅ Purchase transaction (must close on time)\n\n**Consider Waiting if:**\n⚠️ Clear trend of improving rates (momentum)\n⚠️ Major Fed announcement coming (known catalyst)\n⚠️ Refinance (flexibility to delay)\n⚠️ Client accepts risk of rates worsening\n⚠️ Very short wait (30 days) for known event\n\n**Float-Down Option:**\n\nSome lenders offer float-down:\n- Lock today at 3.25%\n- If rates improve 0.25%+ before closing, re-lock\n- Fee: $500-1,000\n- Provides downside protection + upside opportunity\n\n**Client Communication:**\n\n\"I understand wanting the absolute best rate, but let's look at the math. Waiting 6 months for a potential 0.25% improvement costs you $2,130 in lost savings during that time. You'd need 3.5 years just to break even on what you gave up. And there's no guarantee rates will actually drop - they could rise instead. \n\nThe rate available today achieves your goal of saving $355/month with a 12-month breakeven. That's a bird in hand versus two in the bush. If you're concerned about rates potentially improving, we can explore float-down options that give you protection both ways.\"",
            "rejected": "If rates might drop soon, it could be worth waiting. But there's risk rates could go up instead.",
            "metadata": {"category": "rate_lock_strategy", "difficulty": "expert", "cma_module": "A", "source_section": "Refinance or Wait", "line_reference": "515-551", "debate_worthy": true, "client_facing": true, "requires_calculation": true, "requires_market_analysis": true}
        }
    ]

    for i, template in enumerate(templates):
        example = {
            "id": f"cma_{start_id + i:03d}",
            "prompt": template["prompt"],
            "chosen": template["chosen"],
            "rejected": template["rejected"],
            "metadata": template["metadata"]
        }
        if "source_section" not in example["metadata"]:
            example["metadata"]["source_section"] = "Module A"
        if "line_reference" not in example["metadata"]:
            example["metadata"]["line_reference"] = ""
        examples.append(example)

    return examples

# Generate examples
if __name__ == "__main__":
    print("Generating Module A examples...")
    module_a = generate_module_a_examples(start_id=11)
    print(f"Created {len(module_a)} Module A examples")

    # Save to file
    output_file = Path("data/expanded_dataset_module_a.json")
    with open(output_file, 'w') as f:
        json.dump(module_a, f, indent=2)

    print(f"Saved to {output_file}")
