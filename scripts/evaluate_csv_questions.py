#!/usr/bin/env python3
"""
Comprehensive evaluation of loan_origination_questions.csv
Analyzes question quality, duplicates, and alignment with V2 criteria.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_csv_questions(csv_path: str) -> List[Dict[str, str]]:
    """Load questions from CSV file."""
    questions = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    return questions


def analyze_duplicates(questions: List[Dict[str, str]]) -> Dict[str, Any]:
    """Find and analyze duplicate questions."""
    question_texts = [q["Question_Text"] for q in questions]
    text_counts = Counter(question_texts)

    duplicates = {text: count for text, count in text_counts.items() if count > 1}

    # Group question IDs by duplicate text
    duplicate_groups = defaultdict(list)
    for q in questions:
        text = q["Question_Text"]
        if text in duplicates:
            duplicate_groups[text].append(q["Question_ID"])

    return {
        "total_questions": len(questions),
        "unique_questions": len(text_counts),
        "duplicate_count": len(duplicates),
        "duplicate_groups": dict(duplicate_groups),
        "duplicates_list": duplicates,
    }


def analyze_pattern_distribution(questions: List[Dict[str, str]]) -> Dict[str, int]:
    """Analyze distribution of pattern types."""
    pattern_types = [q["Pattern_Type"] for q in questions]
    return dict(Counter(pattern_types))


def analyze_uncertainty_sources(questions: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze primary and secondary uncertainty sources."""
    primary_sources = [q["Primary_Uncertainty_Source"] for q in questions]
    secondary_sources = [q["Secondary_Uncertainty_Source"] for q in questions]

    return {
        "primary_distribution": dict(Counter(primary_sources)),
        "secondary_distribution": dict(Counter(secondary_sources)),
        "combinations": dict(
            Counter(
                f"{q['Primary_Uncertainty_Source']} + {q['Secondary_Uncertainty_Source']}"
                for q in questions
            )
        ),
    }


def analyze_quality_scores(questions: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze quality score distribution."""
    scores = [int(q["Quality_Score_out_of_8"]) for q in questions]

    return {
        "distribution": dict(Counter(scores)),
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "high_quality_count": sum(1 for s in scores if s >= 6),
        "low_quality_count": sum(1 for s in scores if s < 4),
    }


def analyze_difficulty(questions: List[Dict[str, str]]) -> Dict[str, int]:
    """Analyze expected difficulty distribution."""
    difficulties = [q["Expected_Difficulty"] for q in questions]
    return dict(Counter(difficulties))


def validate_v2_criteria(questions: List[Dict[str, str]]) -> Dict[str, Any]:
    """Check alignment with V2 criteria requirements."""

    # Check minimum uncertainty requirement (2+ sources)
    min_uncertainty_met = sum(1 for q in questions if "✓ Yes" in q.get("Min_Uncertainty_Check", ""))

    # Check debate-worthiness
    high_debate_worthy = sum(1 for q in questions if "High" in q.get("Debate_Worthiness", ""))

    # Check for deterministic patterns (red flags from V2)
    red_flags = {
        "exact_numbers": 0,
        "specific_timeframes": 0,
        "clear_market_direction": 0,
        "complete_information": 0,
    }

    for q in questions:
        text = q["Question_Text"].lower()

        # Check for exact dollar amounts (e.g., "$8,000", "$200")
        if "$" in text and any(char.isdigit() for char in text):
            # More nuanced: only flag if very specific (e.g., "exactly $8,000")
            if "exactly" in text or "pay $" in text:
                red_flags["exact_numbers"] += 1

        # Check for specific timeframes (e.g., "exactly 7 years", "in 3 months")
        if "exactly" in text and "year" in text:
            red_flags["specific_timeframes"] += 1

        # Check for clear market direction statements
        if any(
            phrase in text
            for phrase in ["will drop", "will rise", "expected to", "rates are dropping"]
        ):
            red_flags["clear_market_direction"] += 1

        # Check for overly complete information (harder to detect, focus on "all" or "complete")
        if any(phrase in text for phrase in ["all information", "complete data", "fully known"]):
            red_flags["complete_information"] += 1

    return {
        "min_uncertainty_met": min_uncertainty_met,
        "min_uncertainty_percentage": (min_uncertainty_met / len(questions)) * 100,
        "high_debate_worthy": high_debate_worthy,
        "debate_worthy_percentage": (high_debate_worthy / len(questions)) * 100,
        "red_flags": red_flags,
        "red_flag_total": sum(red_flags.values()),
        "red_flag_percentage": (sum(red_flags.values()) / len(questions)) * 100,
    }


def find_strongest_candidates(
    questions: List[Dict[str, str]], top_n: int = 50
) -> List[Dict[str, str]]:
    """Identify strongest questions for validation batch."""

    # Score each question
    scored_questions = []
    for q in questions:
        score = 0

        # Quality score (0-8) - weight heavily
        score += int(q["Quality_Score_out_of_8"]) * 3

        # High debate-worthiness
        if "High" in q.get("Debate_Worthiness", ""):
            score += 5

        # Minimum uncertainty met
        if "✓ Yes" in q.get("Min_Uncertainty_Check", ""):
            score += 3

        # Medium difficulty (not too easy, not impossibly hard)
        if q["Expected_Difficulty"] == "Medium":
            score += 2
        elif q["Expected_Difficulty"] == "Medium-Hard":
            score += 1

        # Diverse pattern types (bonus for less common patterns)
        if q["Pattern_Type"] not in ["Uncertain Timeline"]:
            score += 1

        scored_questions.append((q, score))

    # Sort by score descending
    scored_questions.sort(key=lambda x: x[1], reverse=True)

    return [q for q, score in scored_questions[:top_n]]


def generate_report(questions: List[Dict[str, str]]) -> str:
    """Generate comprehensive evaluation report."""

    # Run all analyses
    dup_analysis = analyze_duplicates(questions)
    pattern_dist = analyze_pattern_distribution(questions)
    uncertainty_analysis = analyze_uncertainty_sources(questions)
    quality_analysis = analyze_quality_scores(questions)
    difficulty_dist = analyze_difficulty(questions)
    v2_validation = validate_v2_criteria(questions)

    # Build report
    report = []
    report.append("=" * 80)
    report.append("CSV QUESTION EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Overview
    report.append("## OVERVIEW")
    report.append(f"Total questions in CSV: {dup_analysis['total_questions']}")
    report.append(f"Unique questions: {dup_analysis['unique_questions']}")
    report.append(f"Duplicate questions: {dup_analysis['duplicate_count']}")
    report.append("")

    # Duplicate Analysis
    report.append("## DUPLICATE ANALYSIS")
    if dup_analysis["duplicate_count"] > 0:
        report.append(f"⚠️  Found {dup_analysis['duplicate_count']} duplicate question texts")
        report.append(
            f"   This reduces effective dataset from {dup_analysis['total_questions']} → {dup_analysis['unique_questions']} questions"
        )
        report.append("")
        report.append("Top 10 duplicates:")
        sorted_dups = sorted(
            dup_analysis["duplicates_list"].items(), key=lambda x: x[1], reverse=True
        )
        for text, count in sorted_dups[:10]:
            # Truncate long text
            display_text = text[:100] + "..." if len(text) > 100 else text
            report.append(f"   {count}x: {display_text}")
    else:
        report.append("✅ No duplicates found")
    report.append("")

    # Pattern Distribution
    report.append("## PATTERN TYPE DISTRIBUTION")
    sorted_patterns = sorted(pattern_dist.items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_patterns:
        percentage = (count / dup_analysis["total_questions"]) * 100
        report.append(f"   {pattern}: {count} ({percentage:.1f}%)")
    report.append("")

    # Uncertainty Sources
    report.append("## UNCERTAINTY SOURCE ANALYSIS")
    report.append("")
    report.append("Primary Uncertainty Sources:")
    sorted_primary = sorted(
        uncertainty_analysis["primary_distribution"].items(), key=lambda x: x[1], reverse=True
    )
    for source, count in sorted_primary:
        percentage = (count / dup_analysis["total_questions"]) * 100
        report.append(f"   {source}: {count} ({percentage:.1f}%)")
    report.append("")
    report.append("Secondary Uncertainty Sources:")
    sorted_secondary = sorted(
        uncertainty_analysis["secondary_distribution"].items(), key=lambda x: x[1], reverse=True
    )
    for source, count in sorted_secondary:
        percentage = (count / dup_analysis["total_questions"]) * 100
        report.append(f"   {source}: {count} ({percentage:.1f}%)")
    report.append("")

    # Quality Scores
    report.append("## QUALITY SCORE ANALYSIS")
    report.append(f"Mean quality score: {quality_analysis['mean']:.2f} / 8")
    report.append(f"Range: {quality_analysis['min']} - {quality_analysis['max']}")
    report.append(
        f"High quality (≥6): {quality_analysis['high_quality_count']} ({(quality_analysis['high_quality_count'] / dup_analysis['total_questions']) * 100:.1f}%)"
    )
    report.append(
        f"Low quality (<4): {quality_analysis['low_quality_count']} ({(quality_analysis['low_quality_count'] / dup_analysis['total_questions']) * 100:.1f}%)"
    )
    report.append("")
    report.append("Distribution:")
    for score in sorted(quality_analysis["distribution"].keys()):
        count = quality_analysis["distribution"][score]
        percentage = (count / dup_analysis["total_questions"]) * 100
        bar = "█" * int(percentage / 2)
        report.append(f"   {score}/8: {count:3d} ({percentage:5.1f}%) {bar}")
    report.append("")

    # Difficulty Distribution
    report.append("## EXPECTED DIFFICULTY DISTRIBUTION")
    sorted_diff = sorted(difficulty_dist.items(), key=lambda x: x[1], reverse=True)
    for difficulty, count in sorted_diff:
        percentage = (count / dup_analysis["total_questions"]) * 100
        report.append(f"   {difficulty}: {count} ({percentage:.1f}%)")
    report.append("")

    # V2 Criteria Validation
    report.append("## V2 CRITERIA COMPLIANCE")
    report.append(
        f"✅ Minimum uncertainty met (2+ sources): {v2_validation['min_uncertainty_met']} ({v2_validation['min_uncertainty_percentage']:.1f}%)"
    )
    report.append(
        f"✅ High debate-worthiness: {v2_validation['high_debate_worthy']} ({v2_validation['debate_worthy_percentage']:.1f}%)"
    )
    report.append("")
    report.append("🚨 Red Flags (Deterministic Patterns):")
    report.append(f"   Exact numbers with 'exactly': {v2_validation['red_flags']['exact_numbers']}")
    report.append(
        f"   Specific timeframes ('exactly X years'): {v2_validation['red_flags']['specific_timeframes']}"
    )
    report.append(
        f"   Clear market direction stated: {v2_validation['red_flags']['clear_market_direction']}"
    )
    report.append(
        f"   Complete information claims: {v2_validation['red_flags']['complete_information']}"
    )
    report.append(
        f"   Total red flags: {v2_validation['red_flag_total']} ({v2_validation['red_flag_percentage']:.1f}%)"
    )
    report.append("")

    # Overall Assessment
    report.append("## OVERALL ASSESSMENT")
    report.append("")

    # Calculate overall grade
    grade_score = 0
    issues = []

    # Deduct for duplicates
    dup_penalty = (dup_analysis["duplicate_count"] / dup_analysis["total_questions"]) * 100
    if dup_penalty > 50:
        grade_score -= 30
        issues.append(f"❌ CRITICAL: {dup_penalty:.1f}% duplicates - major dataset quality issue")
    elif dup_penalty > 20:
        grade_score -= 15
        issues.append(
            f"⚠️  WARNING: {dup_penalty:.1f}% duplicates - significant reduction in effective size"
        )
    elif dup_penalty > 5:
        grade_score -= 5
        issues.append(f"⚠️  Minor: {dup_penalty:.1f}% duplicates")
    else:
        grade_score += 10
        issues.append(f"✅ Minimal duplicates ({dup_penalty:.1f}%)")

    # Check quality scores
    if quality_analysis["mean"] >= 6.5:
        grade_score += 20
        issues.append(f"✅ Excellent quality scores (mean: {quality_analysis['mean']:.2f}/8)")
    elif quality_analysis["mean"] >= 5.5:
        grade_score += 10
        issues.append(f"✅ Good quality scores (mean: {quality_analysis['mean']:.2f}/8)")
    else:
        grade_score -= 10
        issues.append(f"⚠️  Quality scores below target (mean: {quality_analysis['mean']:.2f}/8)")

    # Check V2 criteria compliance
    if v2_validation["min_uncertainty_percentage"] >= 95:
        grade_score += 15
        issues.append(
            f"✅ Strong V2 compliance ({v2_validation['min_uncertainty_percentage']:.1f}% meet 2+ uncertainty sources)"
        )
    elif v2_validation["min_uncertainty_percentage"] >= 80:
        grade_score += 10
        issues.append(
            f"✅ Good V2 compliance ({v2_validation['min_uncertainty_percentage']:.1f}% meet 2+ uncertainty sources)"
        )
    else:
        grade_score -= 5
        issues.append(
            f"⚠️  V2 compliance needs work ({v2_validation['min_uncertainty_percentage']:.1f}% meet 2+ uncertainty sources)"
        )

    # Check red flags
    if v2_validation["red_flag_percentage"] < 5:
        grade_score += 15
        issues.append(
            f"✅ Very few deterministic patterns ({v2_validation['red_flag_percentage']:.1f}%)"
        )
    elif v2_validation["red_flag_percentage"] < 15:
        grade_score += 5
        issues.append(
            f"✅ Low deterministic patterns ({v2_validation['red_flag_percentage']:.1f}%)"
        )
    else:
        grade_score -= 10
        issues.append(
            f"⚠️  Too many deterministic patterns ({v2_validation['red_flag_percentage']:.1f}%)"
        )

    # Check pattern diversity
    pattern_diversity = len(pattern_dist)
    if pattern_diversity >= 5:
        grade_score += 10
        issues.append(f"✅ Good pattern diversity ({pattern_diversity} different types)")
    elif pattern_diversity >= 3:
        grade_score += 5
        issues.append(f"✅ Moderate pattern diversity ({pattern_diversity} different types)")
    else:
        grade_score -= 5
        issues.append(f"⚠️  Limited pattern diversity ({pattern_diversity} different types)")

    # Final grade
    final_grade = max(0, min(100, 50 + grade_score))

    if final_grade >= 85:
        grade_letter = "A (Excellent)"
        recommendation = "✅ READY FOR VALIDATION - High confidence in question quality"
    elif final_grade >= 70:
        grade_letter = "B (Good)"
        recommendation = "✅ READY AFTER DEDUPLICATION - Should produce good results"
    elif final_grade >= 60:
        grade_letter = "C (Fair)"
        recommendation = "⚠️  PROCEED WITH CAUTION - May need question refinement"
    else:
        grade_letter = "D (Needs Work)"
        recommendation = "❌ NOT READY - Significant issues need addressing"

    report.append(f"Overall Grade: {final_grade:.0f}/100 ({grade_letter})")
    report.append("")
    report.append("Key Findings:")
    for issue in issues:
        report.append(f"   {issue}")
    report.append("")
    report.append(f"Recommendation: {recommendation}")
    report.append("")

    # Expected Outcomes
    report.append("## PREDICTED VALIDATION BATCH OUTCOMES")
    report.append("")

    # Predict based on quality scores and V2 compliance
    expected_optimal = int(
        dup_analysis["unique_questions"] * (v2_validation["min_uncertainty_percentage"] / 100) * 0.7
    )
    expected_unanimous = int(
        dup_analysis["unique_questions"] * (v2_validation["red_flag_percentage"] / 100) * 1.5
    )
    expected_kept = expected_optimal + int(
        (dup_analysis["unique_questions"] - expected_optimal - expected_unanimous) * 0.3
    )

    report.append("Based on CSV metadata and V2 criteria:")
    report.append(f"   Expected optimal consensus (60-80%): {expected_optimal} questions")
    report.append(f"   Expected unanimous (100%): {expected_unanimous} questions")
    report.append(
        f"   Expected kept for training: {expected_kept} / {dup_analysis['unique_questions']} ({(expected_kept / dup_analysis['unique_questions']) * 100:.1f}%)"
    )
    report.append(f"   Expected DPO pairs: ~{expected_kept * 2}")
    report.append("")

    # Comparison to previous batch
    report.append("📊 Comparison to Previous Validation Batch:")
    report.append("   Previous: 78% unanimous, 22% kept for training")
    if v2_validation["red_flag_percentage"] < 10:
        report.append(
            f"   Expected: ~{expected_unanimous / dup_analysis['unique_questions'] * 100:.1f}% unanimous, ~{expected_kept / dup_analysis['unique_questions'] * 100:.1f}% kept"
        )
        report.append("   ✅ SIGNIFICANT IMPROVEMENT expected")
    elif v2_validation["red_flag_percentage"] < 20:
        report.append(
            f"   Expected: ~{expected_unanimous / dup_analysis['unique_questions'] * 100:.1f}% unanimous, ~{expected_kept / dup_analysis['unique_questions'] * 100:.1f}% kept"
        )
        report.append("   ✅ MODERATE IMPROVEMENT expected")
    else:
        report.append(
            f"   Expected: Similar to previous (~{expected_unanimous / dup_analysis['unique_questions'] * 100:.1f}% unanimous)"
        )
        report.append("   ⚠️  May still have too many deterministic questions")
    report.append("")

    # Recommendations
    report.append("=" * 80)
    report.append("## RECOMMENDED NEXT STEPS")
    report.append("=" * 80)
    report.append("")

    if dup_analysis["duplicate_count"] > 0:
        report.append("1. **DEDUPLICATE DATASET**")
        report.append(f"   Remove {dup_analysis['duplicate_count']} duplicate questions")
        report.append(
            f"   This will reduce dataset from {dup_analysis['total_questions']} → {dup_analysis['unique_questions']} questions"
        )
        report.append("")

    if final_grade >= 70:
        report.append("2. **RUN 50-QUESTION VALIDATION BATCH**")
        report.append("   Extract top 50 unique questions by quality score")
        report.append("   Use same config as previous validation (M=5, 2 rounds, temp=0.9)")
        report.append("   Expected completion: ~45 minutes")
        report.append("")
        report.append("3. **IF VALIDATION PASSES (>50% kept for training)**")
        report.append(f"   Proceed with full {dup_analysis['unique_questions']}-question batch")
        report.append(
            f"   Expected duration: ~{(dup_analysis['unique_questions'] * 0.75):.0f} minutes ({(dup_analysis['unique_questions'] * 0.75 / 60):.1f} hours)"
        )
        report.append(f"   Expected DPO pairs: {expected_kept * 2}+")
    else:
        report.append("2. **REFINE QUESTIONS BEFORE VALIDATION**")
        report.append("   Focus on questions with red flags")
        report.append("   Apply V2 criteria quick fix formula:")
        report.append("      - Remove exact numbers/timeframes")
        report.append("      - Add conflicting information")
        report.append("      - Introduce unknown variables")
        report.append("      - Force value judgments")
        report.append("")
        report.append("3. **REVALIDATE WITH PILOT BATCH**")
        report.append("   Test 10-20 refined questions first")
        report.append("   Verify improvements before full batch")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main evaluation function."""
    csv_path = (
        Path(__file__).parent.parent / "proprietary" / "data" / "loan_origination_questions.csv"
    )

    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return

    print(f"Loading questions from: {csv_path}")
    questions = load_csv_questions(str(csv_path))

    print(f"Loaded {len(questions)} questions")
    print("\nGenerating comprehensive evaluation report...\n")

    report = generate_report(questions)
    print(report)

    # Save report
    report_path = csv_path.parent / "csv_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {report_path}")

    # Also find and save top 50 candidates for validation
    top_50 = find_strongest_candidates(questions, top_n=50)

    # Convert to JSON format for batch processing
    validation_questions = []
    for _i, q in enumerate(top_50, 1):
        validation_questions.append(
            {
                "id": q["Question_ID"],
                "prompt": q["Question_Text"],
                "metadata": {
                    "pattern_type": q["Pattern_Type"],
                    "primary_uncertainty": q["Primary_Uncertainty_Source"],
                    "secondary_uncertainty": q["Secondary_Uncertainty_Source"],
                    "quality_score": q["Quality_Score_out_of_8"],
                    "expected_difficulty": q["Expected_Difficulty"],
                },
            }
        )

    validation_path = csv_path.parent / "loan_origination_questions_validation_v2.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_questions, f, indent=2)

    print(f"✅ Top 50 validation candidates saved to: {validation_path}")


if __name__ == "__main__":
    main()
