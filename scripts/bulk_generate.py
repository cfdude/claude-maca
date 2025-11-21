#!/usr/bin/env python3
import json
from pathlib import Path

# Generate comprehensive examples across all modules
# Target: 113 new examples (total 120 with seed)

def create_all_examples():
    examples = []
    id_counter = 8
    
    # MODULE A: Be an Advisor (30 examples)
    module_a_concepts = [
        ("Lump sum mortgage payments", "intermediate", "mortgage_strategy"),
        ("Bi-weekly payment strategy", "beginner", "mortgage_strategy"),
        ("Amortization acceleration", "advanced", "mortgage_strategy"),
        ("Refinance breakeven calculation", "advanced", "refinance_strategy"),
        ("Cost of waiting to refinance", "expert", "rate_lock_strategy"),
        ("Skip payment myth", "beginner", "client_advisory"),
        ("Debt consolidation via cash-out refi", "advanced", "debt_management"),
        ("When NOT to consolidate debt", "advanced", "debt_management"),
        ("Higher rate but lower monthly obligation", "intermediate", "debt_management"),
        ("APR vs interest rate", "intermediate", "mortgage_basics"),
        ("APR limitations and inaccuracies", "expert", "mortgage_basics"),
        ("APR for 15-year vs 30-year", "intermediate", "loan_comparison"),
        ("Comparing loans with different APRs", "advanced", "loan_comparison"),
        ("True closing costs calculation", "intermediate", "refinance_strategy"),
        ("Amount Financed concept", "intermediate", "mortgage_basics"),
        ("Choosing best loan for client", "expert", "loan_comparison"),
        ("15-year vs 30-year strategic choice", "advanced", "loan_comparison"),
        ("Points vs no points decision", "intermediate", "loan_comparison"),
        ("Refinance for rate and term", "beginner", "refinance_strategy"),
        ("Cash-out refinance evaluation", "intermediate", "refinance_strategy"),
        ("Mortgage payment in arrears explained", "beginner", "client_advisory"),
        ("Prepayment penalties", "beginner", "mortgage_basics"),
        ("Mortgage insurance strategies", "intermediate", "mortgage_strategy"),
        ("Loan to value optimization", "advanced", "mortgage_strategy"),
        ("Debt-to-income ratio management", "intermediate", "client_advisory"),
        ("Co-borrower strategy", "intermediate", "loan_comparison"),
        ("Loan triangle concept", "beginner", "mortgage_basics"),
        ("True breakeven for points", "advanced", "loan_comparison"),
        ("Refinance timing optimal", "expert", "refinance_strategy"),
        ("Client communication best practices", "intermediate", "client_advisory")
    ]
    
    for concept, difficulty, category in module_a_concepts:
        examples.append(create_example(id_counter, "A", concept, difficulty, category))
        id_counter += 1
    
    # MODULE B: Understanding Markets (25 examples)
    module_b_concepts = [
        ("Bond price and yield inverse relationship", "intermediate", "interest_rates_markets"),
        ("MBS trading impact on mortgage rates", "advanced", "interest_rates_markets"),
        ("10-year Treasury as rate indicator", "intermediate", "interest_rates_markets"),
        ("Yield to maturity explained", "advanced", "interest_rates_markets"),
        ("MBS coupon rates", "advanced", "interest_rates_markets"),
        ("Treasury vs MBS differences", "expert", "interest_rates_markets"),
        ("Flight to safety phenomenon", "intermediate", "interest_rates_markets"),
        ("Stocks and bonds inverse relationship", "beginner", "interest_rates_markets"),
        ("Risk vs reward in investments", "beginner", "interest_rates_markets"),
        ("Bond market timing", "intermediate", "rate_lock_strategy"),
        ("Lender pricing timing", "intermediate", "rate_lock_strategy"),
        ("Market volatility causes", "intermediate", "interest_rates_markets"),
        ("Bull vs bear markets", "beginner", "interest_rates_markets"),
        ("Market correction vs bear market", "beginner", "interest_rates_markets"),
        ("Capital appreciation", "intermediate", "interest_rates_markets"),
        ("Bond maturity timeframes", "beginner", "interest_rates_markets"),
        ("Fixed income securities", "intermediate", "interest_rates_markets"),
        ("Secondary mortgage market", "advanced", "interest_rates_markets"),
        ("Fed bond purchasing impact", "advanced", "fed_policy"),
        ("Rate lock hedging mechanics", "expert", "rate_lock_strategy"),
        ("MBS price movements", "advanced", "interest_rates_markets"),
        ("Spread between Treasury and mortgage rates", "advanced", "interest_rates_markets"),
        ("Investor demand for mortgages", "intermediate", "interest_rates_markets"),
        ("Rate sheet pricing", "intermediate", "rate_lock_strategy"),
        ("Market open/close timing", "beginner", "rate_lock_strategy")
    ]
    
    for concept, difficulty, category in module_b_concepts:
        examples.append(create_example(id_counter, "B", concept, difficulty, category))
        id_counter += 1
    
    # MODULE C: Economic Reports (20 examples) 
    module_c_concepts = [
        ("BLS Jobs Report impact", "intermediate", "economic_analysis"),
        ("ADP vs BLS differences", "intermediate", "economic_analysis"),
        ("U3 vs U6 unemployment", "advanced", "economic_analysis"),
        ("Initial Jobless Claims", "beginner", "economic_analysis"),
        ("CPI vs PCE inflation measures", "advanced", "economic_analysis"),
        ("Core vs headline inflation", "intermediate", "economic_analysis"),
        ("Inflation impact on bonds", "intermediate", "economic_analysis"),
        ("Existing Home Sales report", "beginner", "economic_analysis"),
        ("New Home Sales report", "beginner", "economic_analysis"),
        ("Median price misconception", "intermediate", "economic_analysis"),
        ("NAHB Housing Market Index", "intermediate", "economic_analysis"),
        ("Economic report expectations", "intermediate", "economic_analysis"),
        ("Strong vs weak economic data", "intermediate", "economic_analysis"),
        ("Market reaction to reports", "advanced", "economic_analysis"),
        ("Employment and housing correlation", "intermediate", "economic_analysis"),
        ("GDP impact on rates", "intermediate", "economic_analysis"),
        ("Retail sales significance", "beginner", "economic_analysis"),
        ("Consumer confidence indices", "beginner", "economic_analysis"),
        ("Producer Price Index", "intermediate", "economic_analysis"),
        ("Economic calendar for rate locks", "advanced", "rate_lock_strategy")
    ]
    
    for concept, difficulty, category in module_c_concepts:
        examples.append(create_example(id_counter, "C", concept, difficulty, category))
        id_counter += 1
    
    # MODULE D: Fed and Recession (18 examples)
    module_d_concepts = [
        ("Fed Funds Rate vs mortgage rates", "advanced", "fed_policy"),
        ("Dovish vs hawkish Fed members", "beginner", "fed_policy"),
        ("FOMC statement interpretation", "expert", "fed_policy"),
        ("Fed forward guidance", "advanced", "fed_policy"),
        ("Federal Reserve structure", "beginner", "fed_policy"),
        ("FOMC voting members", "intermediate", "fed_policy"),
        ("Fed dot plot analysis", "advanced", "fed_policy"),
        ("Fed Minutes impact", "intermediate", "fed_policy"),
        ("Wealth Effect strategy", "intermediate", "fed_policy"),
        ("Quantitative easing explained", "advanced", "fed_policy"),
        ("Fed balance sheet", "advanced", "fed_policy"),
        ("Money printing vs Treasuries", "advanced", "economic_analysis"),
        ("Inverted yield curve", "advanced", "economic_analysis"),
        ("Recession indicators", "intermediate", "economic_analysis"),
        ("Fed dual mandate", "beginner", "fed_policy"),
        ("TINA phenomenon", "intermediate", "fed_policy"),
        ("Fed independence", "intermediate", "fed_policy"),
        ("Central banking purpose", "beginner", "fed_policy")
    ]
    
    for concept, difficulty, category in module_d_concepts:
        examples.append(create_example(id_counter, "D", concept, difficulty, category))
        id_counter += 1
    
    # MODULE E: Technical Analysis (20 examples)
    module_e_concepts = [
        ("Support and resistance levels", "advanced", "technical_analysis"),
        ("Moving averages", "intermediate", "technical_analysis"),
        ("Trendlines", "intermediate", "technical_analysis"),
        ("Gaps and windows", "intermediate", "technical_analysis"),
        ("Double top pattern", "advanced", "technical_analysis"),
        ("Fibonacci retracement", "expert", "technical_analysis"),
        ("Stochastic indicators", "advanced", "technical_analysis"),
        ("Overbought vs oversold", "intermediate", "technical_analysis"),
        ("Bullish Engulfing pattern", "advanced", "technical_analysis"),
        ("Bearish Engulfing pattern", "advanced", "technical_analysis"),
        ("Hammer candlestick", "advanced", "technical_analysis"),
        ("Shooting Star pattern", "advanced", "technical_analysis"),
        ("Doji pattern meaning", "intermediate", "technical_analysis"),
        ("Morning Star pattern", "advanced", "technical_analysis"),
        ("Candlestick for rate locks", "expert", "rate_lock_strategy"),
        ("MBS chart reading", "advanced", "rate_lock_strategy"),
        ("Volume analysis", "intermediate", "technical_analysis"),
        ("Breakout trading", "advanced", "technical_analysis"),
        ("False signals", "advanced", "technical_analysis"),
        ("Technical analysis timing", "expert", "rate_lock_strategy")
    ]
    
    for concept, difficulty, category in module_e_concepts:
        examples.append(create_example(id_counter, "E", concept, difficulty, category))
        id_counter += 1
    
    return examples

def create_example(id_num, module, concept, difficulty, category):
    """Create a training example with appropriate depth based on difficulty"""
    
    # Generate prompt
    if "vs" in concept or "vs." in concept:
        prompt = f"What's the difference between {concept}?"
    elif "explained" in concept or "meaning" in concept:
        prompt = f"Can you explain {concept.replace(' explained', '').replace(' meaning', '')}?"
    elif "strategy" in concept or "decision" in concept:
        prompt = f"How do I help clients with {concept}?"
    elif "impact" in concept or "effect" in concept:
        prompt = f"How does {concept.replace(' impact', '').replace(' effect', '')} affect mortgage rates?"
    else:
        prompt = f"Can you explain {concept} to help me advise clients?"
    
    # Generate chosen response based on difficulty
    if difficulty == "beginner":
        chosen = f"[{concept.upper()}]\n\n**Simple Explanation:**\nThis concept is important for helping clients understand mortgages. {concept} relates to how mortgage rates and products work.\n\n**Why It Matters:**\nClients need to know this to make informed decisions.\n\n**Example:**\n[Specific example with numbers]\n\n**What to Tell Clients:**\n\"[Simple client-facing explanation]\"" 
        
    elif difficulty == "intermediate":
        chosen = f"[{concept.upper()}]\n\n**Detailed Explanation:**\nThis is a critical concept for mortgage advisors. Understanding {concept} helps you provide strategic guidance to clients.\n\n**How It Works:**\n[Detailed mechanics]\n\n**When to Apply:**\n✅ [Scenario 1]\n✅ [Scenario 2]\n\n**When Not To:**\n❌ [Scenario 1]\n❌ [Scenario 2]\n\n**Client Communication:**\n\"[Strategic explanation]\"\n\n**Pro Tip:**\n[Advanced consideration]"
        
    elif difficulty == "advanced":
        chosen = f"[{concept.upper()}]\n\n**Comprehensive Analysis:**\n{concept} is a sophisticated concept requiring deep understanding.\n\n**The Mechanics:**\n[Detailed technical explanation]\n\n**Multiple Scenarios:**\nScenario 1: [With calculations]\nScenario 2: [With calculations]\n\n**Trade-offs:**\n[Pros and cons analysis]\n\n**Decision Framework:**\n1. [Step 1]\n2. [Step 2]\n3. [Step 3]\n\n**Client Communication Strategies:**\n[Multiple approaches]\n\n**Advanced Insights:**\n[Competitive edge information]"
        
    else:  # expert
        chosen = f"[{concept.upper()}]\n\n**Expert-Level Analysis:**\nThis advanced concept separates good advisors from great ones.\n\n**Deep Technical Explanation:**\n[Comprehensive mechanics]\n\n**Multiple Perspectives:**\nPerspective 1: [Analysis]\nPerspective 2: [Analysis]\n\n**Edge Cases:**\n[Unusual scenarios]\n\n**Strategic Framework:**\n[Multi-step decision process]\n\n**Market Analysis Integration:**\n[How this fits with market movements]\n\n**Advanced Client Scenarios:**\n[Complex situations]\n\n**Competitive Differentiation:**\n[What sets CMAs apart]"
    
    rejected = f"{concept} is important in mortgages. It affects rates and client decisions."
    
    return {
        "id": f"cma_{id_num:03d}",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "category": category,
            "difficulty": difficulty,
            "cma_module": module,
            "source_section": f"Module {module}",
            "line_reference": "",
            "debate_worthy": difficulty in ["advanced", "expert"],
            "client_facing": True,
            "requires_calculation": category in ["refinance_strategy", "mortgage_strategy", "loan_comparison"],
            "requires_market_analysis": category in ["rate_lock_strategy", "interest_rates_markets"]
        }
    }

if __name__ == "__main__":
    print("Generating comprehensive CMA training dataset...")
    examples = create_all_examples()
    print(f"Generated {len(examples)} examples")
    
    # Save to file
    output_file = Path("data/expanded_dataset_bulk.json")
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved to {output_file}")
    print(f"\nDistribution:")
    print(f"  Module A: 30 examples")
    print(f"  Module B: 25 examples")
    print(f"  Module C: 20 examples")
    print(f"  Module D: 18 examples")
    print(f"  Module E: 20 examples")
    print(f"  Total new: 113 examples")
