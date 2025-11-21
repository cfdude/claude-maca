#!/usr/bin/env python3
"""
Automated Training Example Generator for CMA Dataset

Generates comprehensive Q&A pairs from CMA source material across all 5 modules.
Target: 93-143 new examples to reach 100-150 total.
"""

import json
import re
from pathlib import Path

# Training example templates organized by module and concept
TRAINING_TEMPLATES = {
    "A": [
        # Amortization concepts
        {
            "prompt": "A client received a ${amount} bonus and wants to know if they should use it to make a lump sum payment on their mortgage. What should I tell them?",
            "concept": "lump_sum_payments",
            "difficulty": "intermediate",
            "category": "mortgage_strategy"
        },
        {
            "prompt": "How do bi-weekly payments help clients pay off mortgages faster, and is it worth setting up?",
            "concept": "biweekly_payments",
            "difficulty": "beginner",
            "category": "mortgage_basics"
        },
        # Refinance breakeven
        {
            "prompt": "How do I correctly calculate the breakeven period for a refinance? My calculations seem different from what clients expect.",
            "concept": "refinance_breakeven",
            "difficulty": "advanced",
            "category": "refinance_strategy"
        },
        {
            "prompt": "A client wants to wait for rates to drop another 0.25% before refinancing. How do I explain the cost of waiting?",
            "concept": "cost_of_waiting",
            "difficulty": "intermediate",
            "category": "rate_lock_strategy"
        },
        {
            "prompt": "Do clients actually skip a payment when they refinance? How should I explain this?",
            "concept": "skip_payment_myth",
            "difficulty": "beginner",
            "category": "client_advisory"
        },
        # Debt consolidation
        {
            "prompt": "When should I advise a client AGAINST debt consolidation through cash-out refinance?",
            "concept": "debt_consolidation_risks",
            "difficulty": "advanced",
            "category": "debt_management"
        },
        {
            "prompt": "How do I explain to a client that a higher mortgage rate might still save them money through debt consolidation?",
            "concept": "total_monthly_obligation",
            "difficulty": "intermediate",
            "category": "debt_management"
        },
        # APR concepts
        {
            "prompt": "A client is comparing two loan offers - one has lower rate but higher APR, the other has higher rate but lower APR. Which should they choose?",
            "concept": "apr_comparison",
            "difficulty": "advanced",
            "category": "loan_comparison"
        },
        {
            "prompt": "Why is APR often misleading when comparing loans, and what's a better method?",
            "concept": "apr_limitations",
            "difficulty": "expert",
            "category": "mortgage_basics"
        },
        {
            "prompt": "How does the loan term affect APR, and why does refinancing early make APR calculations inaccurate?",
            "concept": "apr_timeframe",
            "difficulty": "intermediate",
            "category": "refinance_strategy"
        }
    ],
    "B": [
        # Bonds and yields
        {
            "prompt": "Why do bond prices and yields move in opposite directions? Can you explain this with a simple example?",
            "concept": "bond_price_yield_inverse",
            "difficulty": "intermediate",
            "category": "interest_rates_markets"
        },
        {
            "prompt": "How are mortgage rates connected to Mortgage-Backed Securities (MBS) trading?",
            "concept": "mbs_mortgage_rates",
            "difficulty": "advanced",
            "category": "interest_rates_markets"
        },
        {
            "prompt": "Why do we watch the 10-year Treasury yield for mortgage rate trends, but mortgage rates are based on MBS?",
            "concept": "treasury_vs_mbs",
            "difficulty": "expert",
            "category": "interest_rates_markets"
        },
        {
            "prompt": "What is Yield to Maturity (YTM) and why does it matter for mortgage investors?",
            "concept": "yield_to_maturity",
            "difficulty": "advanced",
            "category": "interest_rates_markets"
        },
        # Stocks vs Bonds
        {
            "prompt": "When the stock market crashes, what typically happens to mortgage rates and why?",
            "concept": "flight_to_safety",
            "difficulty": "intermediate",
            "category": "interest_rates_markets"
        },
        {
            "prompt": "Why do stocks and bonds often move in opposite directions?",
            "concept": "stocks_bonds_inverse",
            "difficulty": "beginner",
            "category": "interest_rates_markets"
        },
        # Market timing
        {
            "prompt": "What time of day do lenders typically price rate sheets, and why does this timing matter?",
            "concept": "lender_pricing_timing",
            "difficulty": "intermediate",
            "category": "rate_lock_strategy"
        }
    ],
    "C": [
        # Employment reports
        {
            "prompt": "How do employment reports (like the BLS Jobs Report) affect mortgage rates?",
            "concept": "jobs_report_impact",
            "difficulty": "intermediate",
            "category": "economic_analysis"
        },
        {
            "prompt": "What's the difference between the U3 and U6 unemployment rates, and which gives a more accurate picture?",
            "concept": "unemployment_measures",
            "difficulty": "advanced",
            "category": "economic_analysis"
        },
        # Inflation
        {
            "prompt": "Why does inflation negatively affect bond prices and mortgage rates?",
            "concept": "inflation_bonds",
            "difficulty": "intermediate",
            "category": "economic_analysis"
        },
        {
            "prompt": "What's the difference between CPI and PCE inflation measures, and why does the Fed prefer PCE?",
            "concept": "cpi_vs_pce",
            "difficulty": "advanced",
            "category": "economic_analysis"
        },
        # Housing reports
        {
            "prompt": "A housing report shows median home prices up 10% year-over-year. Does this mean home values appreciated 10%?",
            "concept": "median_price_misconception",
            "difficulty": "intermediate",
            "category": "economic_analysis"
        },
        {
            "prompt": "What's the difference between Existing Home Sales and New Home Sales reports?",
            "concept": "housing_reports",
            "difficulty": "beginner",
            "category": "economic_analysis"
        }
    ],
    "D": [
        # Federal Reserve
        {
            "prompt": "When the Fed cuts the Fed Funds Rate, why don't mortgage rates automatically drop by the same amount?",
            "concept": "fed_funds_vs_mortgage_rates",
            "difficulty": "advanced",
            "category": "fed_policy"
        },
        {
            "prompt": "What's the difference between a 'dovish' and 'hawkish' Fed member?",
            "concept": "dovish_hawkish",
            "difficulty": "beginner",
            "category": "fed_policy"
        },
        {
            "prompt": "How should I interpret FOMC statements and forward guidance for rate lock timing?",
            "concept": "fomc_interpretation",
            "difficulty": "expert",
            "category": "fed_policy"
        },
        {
            "prompt": "What is the Wealth Effect and how does the Fed use it to stimulate the economy?",
            "concept": "wealth_effect",
            "difficulty": "intermediate",
            "category": "fed_policy"
        },
        # Recession indicators
        {
            "prompt": "What is an inverted yield curve and why is it a recession indicator?",
            "concept": "inverted_yield_curve",
            "difficulty": "advanced",
            "category": "economic_analysis"
        }
    ],
    "E": [
        # Technical analysis - Western
        {
            "prompt": "What are support and resistance levels in MBS trading, and how do they help with rate lock decisions?",
            "concept": "support_resistance",
            "difficulty": "advanced",
            "category": "technical_analysis"
        },
        {
            "prompt": "How do I use moving averages to identify trends in mortgage rate movements?",
            "concept": "moving_averages",
            "difficulty": "intermediate",
            "category": "technical_analysis"
        },
        {
            "prompt": "What is a 'gap' or 'window' in trading charts and what does it signal?",
            "concept": "gaps_windows",
            "difficulty": "intermediate",
            "category": "technical_analysis"
        },
        # Technical analysis - Eastern (Candlesticks)
        {
            "prompt": "What is a Bullish Engulfing candlestick pattern and what does it indicate for mortgage rates?",
            "concept": "bullish_engulfing",
            "difficulty": "advanced",
            "category": "technical_analysis"
        },
        {
            "prompt": "How do I use candlestick patterns to time rate locks more effectively?",
            "concept": "candlestick_rate_locks",
            "difficulty": "expert",
            "category": "rate_lock_strategy"
        },
        {
            "prompt": "What does a Doji candlestick pattern tell us about market sentiment?",
            "concept": "doji_pattern",
            "difficulty": "intermediate",
            "category": "technical_analysis"
        },
        # Fibonacci and stochastics
        {
            "prompt": "How do Fibonacci retracement levels help identify potential support/resistance in MBS prices?",
            "concept": "fibonacci_retracement",
            "difficulty": "expert",
            "category": "technical_analysis"
        },
        {
            "prompt": "What are Stochastic indicators and how do they show overbought/oversold conditions?",
            "concept": "stochastic_indicators",
            "difficulty": "advanced",
            "category": "technical_analysis"
        }
    ]
}

# Response frameworks for generating "chosen" answers
RESPONSE_FRAMEWORKS = {
    "beginner": {
        "structure": [
            "Simple explanation of the concept",
            "Why it matters to the client",
            "Clear example with numbers",
            "What to tell the client"
        ],
        "length": "300-500 words"
    },
    "intermediate": {
        "structure": [
            "Detailed explanation of concept",
            "How it works with examples",
            "When to apply vs when not to",
            "Client communication tips",
            "Pro tip or advanced consideration"
        ],
        "length": "500-700 words"
    },
    "advanced": {
        "structure": [
            "Comprehensive explanation",
            "Multiple scenarios with calculations",
            "Trade-offs and considerations",
            "Decision framework",
            "Client communication strategies",
            "Advanced insights"
        ],
        "length": "700-1000 words"
    },
    "expert": {
        "structure": [
            "Deep technical explanation",
            "Multiple perspectives and edge cases",
            "Strategic decision-making framework",
            "Market analysis integration",
            "Advanced client scenarios",
            "Competitive differentiation insights"
        ],
        "length": "800-1200 words"
    }
}

def generate_metadata(module, concept_data):
    """Generate metadata for training example"""
    return {
        "category": concept_data["category"],
        "difficulty": concept_data["difficulty"],
        "cma_module": module,
        "source_section": concept_data.get("source_section", "Multiple sections"),
        "line_reference": concept_data.get("line_reference", ""),
        "debate_worthy": concept_data.get("debate_worthy", False),
        "client_facing": concept_data.get("client_facing", True),
        "requires_calculation": concept_data.get("requires_calculation", False),
        "requires_market_analysis": concept_data.get("requires_market_analysis", False),
        "visual_concept": concept_data.get("visual_concept", False),
        "image_references": concept_data.get("image_references", [])
    }

def main():
    """Generate training examples"""
    print("=" * 60)
    print("CMA Training Example Generator")
    print("=" * 60)

    total_templates = sum(len(templates) for templates in TRAINING_TEMPLATES.values())
    print(f"\nTotal concept templates defined: {total_templates}")
    print(f"\nBreakdown by module:")
    for module, templates in TRAINING_TEMPLATES.items():
        print(f"  Module {module}: {len(templates)} concepts")

    print(f"\nTo reach 100-150 total examples:")
    print(f"  Current seed examples: 7")
    print(f"  Need to generate: 93-143 examples")
    print(f"  Templates available: {total_templates}")
    print(f"\nEach template can generate 1-3 variations")
    print(f"Estimated total: {total_templates * 2} examples")

if __name__ == "__main__":
    main()
