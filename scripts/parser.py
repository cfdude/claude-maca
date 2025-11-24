"""
Answer Parser Module

Provides domain-agnostic and domain-specific answer normalization and comparison
for improved consensus detection in MACA debates.

This module implements flexible answer parsing that handles:
- Generic normalization (yes/no, prefixes, whitespace)
- Domain-specific normalization (financial, legal, medical)
- Fuzzy string matching for equivalence checking
- Robust handling of None/empty answers

Usage:
    from parser import AnswerParser

    parser = AnswerParser()
    normalized = parser.normalize("Yes, I agree", domain="generic")
    is_equivalent = parser.grade_answer("$1,000.00", "1000", domain="financial")
"""

import re
from difflib import SequenceMatcher


class AnswerParser:
    """
    Domain-agnostic answer parser with pluggable domain-specific normalizers.

    Provides methods to normalize answers and check equivalence with fuzzy matching,
    improving consensus detection accuracy across different answer formats.
    """

    # Default similarity threshold for fuzzy matching
    DEFAULT_SIMILARITY_THRESHOLD = 0.85

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        """
        Initialize the answer parser.

        Args:
            similarity_threshold: Minimum similarity score (0.0-1.0) for answers
                                to be considered equivalent. Default: 0.85
        """
        self.similarity_threshold = similarity_threshold

    def normalize(self, answer: str, domain: str = "generic") -> str:
        """
        Normalize an answer based on domain-specific rules.

        Args:
            answer: The raw answer string to normalize
            domain: The domain context ("generic", "financial", "legal", "medical")

        Returns:
            Normalized answer string, or empty string if answer is None/empty

        Examples:
            >>> parser = AnswerParser()
            >>> parser.normalize("Yes, I agree")
            'yes i agree'
            >>> parser.normalize("$1,000.00", domain="financial")
            '1000.00'
            >>> parser.normalize("42 U.S.C. § 1983", domain="legal")
            '42 usc 1983'
        """
        # Handle None/empty gracefully
        if answer is None or answer == "":
            return ""

        # Apply generic normalization first
        normalized = self._normalize_generic(answer)

        # Apply domain-specific normalization
        if domain == "financial":
            normalized = self._normalize_financial(normalized)
        elif domain == "legal":
            normalized = self._normalize_legal(normalized)
        elif domain == "medical":
            normalized = self._normalize_medical(normalized)

        return normalized

    def grade_answer(self, given: str, ground_truth: str, domain: str = "generic") -> bool:
        """
        Check if two answers are equivalent using fuzzy matching.

        Normalizes both answers and compares using string similarity. Answers are
        considered equivalent if similarity >= similarity_threshold.

        Args:
            given: The answer to check
            ground_truth: The reference answer
            domain: The domain context for normalization

        Returns:
            True if answers are equivalent (similarity >= threshold), False otherwise

        Examples:
            >>> parser = AnswerParser()
            >>> parser.grade_answer("Yes", "yes")
            True
            >>> parser.grade_answer("$1,000", "$1000.00", domain="financial")
            True
            >>> parser.grade_answer("completely different", "answer here")
            False
        """
        # Handle None/empty gracefully
        if given is None or ground_truth is None:
            return given == ground_truth

        # Normalize both answers
        normalized_given = self.normalize(given, domain=domain)
        normalized_truth = self.normalize(ground_truth, domain=domain)

        # Empty answers only match other empty answers
        if not normalized_given or not normalized_truth:
            return normalized_given == normalized_truth

        # Exact match after normalization
        if normalized_given == normalized_truth:
            return True

        # Fuzzy match with similarity threshold
        similarity = self._similarity(normalized_given, normalized_truth)
        return similarity >= self.similarity_threshold

    def _normalize_generic(self, answer: str) -> str:
        """
        Apply generic normalization rules applicable to all domains.

        Rules applied:
        - Convert to lowercase
        - Strip leading/trailing whitespace
        - Remove common prefixes ("Answer:", "Response:", etc.)
        - Normalize whitespace (collapse multiple spaces)
        - Strip punctuation from edges

        Args:
            answer: The answer string to normalize

        Returns:
            Generically normalized answer string
        """
        # Convert to lowercase
        normalized = answer.lower().strip()

        # Remove common prefixes
        prefixes = [
            "answer:",
            "response:",
            "result:",
            "output:",
            "solution:",
            "my answer is",
            "i think",
            "i believe",
            "in my opinion",
        ]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()

        # Normalize yes/no variations
        yes_variations = ["yes", "yeah", "yep", "yup", "affirmative", "correct", "true"]
        no_variations = ["no", "nope", "nah", "negative", "incorrect", "false"]

        if normalized in yes_variations:
            normalized = "yes"
        elif normalized in no_variations:
            normalized = "no"

        # Collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized)

        # Strip edge punctuation (but preserve internal punctuation)
        normalized = normalized.strip(".,;:!?\"'-")

        return normalized

    def _normalize_financial(self, answer: str) -> str:
        """
        Apply financial domain-specific normalization.

        Rules applied:
        - Remove currency symbols ($, €, £, ¥)
        - Remove thousand separators (commas)
        - Handle percentage signs
        - Preserve decimal points
        - Normalize number formats

        Args:
            answer: The answer string to normalize (should already be generically normalized)

        Returns:
            Financially normalized answer string

        Examples:
            >>> parser = AnswerParser()
            >>> parser._normalize_financial("$1,000.00")
            '1000.00'
            >>> parser._normalize_financial("25%")
            '25 percent'
        """
        normalized = answer

        # Remove currency symbols
        currency_symbols = ["$", "€", "£", "¥", "¢"]
        for symbol in currency_symbols:
            normalized = normalized.replace(symbol, "")

        # Remove thousand separators
        normalized = normalized.replace(",", "")

        # Normalize percentage
        if "%" in normalized:
            normalized = normalized.replace("%", " percent")

        # Handle financial notation (1.5M, 2.3K, etc.)
        # Must preserve the captured number and multiply by the appropriate factor
        def multiply_by_million(match):
            return str(float(match.group(1)) * 1000000)

        def multiply_by_thousand(match):
            return str(float(match.group(1)) * 1000)

        def multiply_by_billion(match):
            return str(float(match.group(1)) * 1000000000)

        normalized = re.sub(
            r"(\d+\.?\d*)\s*m\b", multiply_by_million, normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(
            r"(\d+\.?\d*)\s*k\b", multiply_by_thousand, normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(
            r"(\d+\.?\d*)\s*b\b", multiply_by_billion, normalized, flags=re.IGNORECASE
        )

        return normalized.strip()

    def _normalize_legal(self, answer: str) -> str:
        """
        Apply legal domain-specific normalization.

        Rules applied:
        - Normalize citation formats (42 U.S.C. § 1983 → 42 usc 1983)
        - Remove section symbols (§)
        - Normalize abbreviations (U.S.C., C.F.R., etc.)
        - Handle case citations

        Args:
            answer: The answer string to normalize (should already be generically normalized)

        Returns:
            Legally normalized answer string

        Examples:
            >>> parser = AnswerParser()
            >>> parser._normalize_legal("42 u.s.c. § 1983")
            '42 usc 1983'
        """
        normalized = answer

        # Remove section symbols
        normalized = normalized.replace("§", "").replace("section", "")

        # Normalize U.S.C. variations
        normalized = re.sub(r"u\.?s\.?c\.?", "usc", normalized)
        normalized = re.sub(r"c\.?f\.?r\.?", "cfr", normalized)
        normalized = re.sub(r"u\.?s\.?\b", "us", normalized)

        # Remove v. or vs. in case citations
        normalized = re.sub(r"\s+v\.?\s+", " v ", normalized)
        normalized = re.sub(r"\s+vs\.?\s+", " v ", normalized)

        # Collapse multiple spaces again after substitutions
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def _normalize_medical(self, answer: str) -> str:
        """
        Apply medical domain-specific normalization.

        Rules applied:
        - Normalize ICD codes (ICD-10, ICD-9)
        - Normalize medication names
        - Handle dosage formats
        - Normalize medical abbreviations

        Args:
            answer: The answer string to normalize (should already be generically normalized)

        Returns:
            Medically normalized answer string

        Examples:
            >>> parser = AnswerParser()
            >>> parser._normalize_medical("icd-10 code j44.0")
            'icd10 code j440'
        """
        normalized = answer

        # Normalize ICD codes
        normalized = re.sub(r"icd-?10", "icd10", normalized)
        normalized = re.sub(r"icd-?9", "icd9", normalized)

        # Remove dashes from codes (J44.0 → j440, but keep spaces)
        # This regex only affects code-like patterns (letter followed by digits and dots)
        normalized = re.sub(r"([a-z])(\d+)\.(\d+)", r"\1\2\3", normalized)

        # Normalize dosage units
        normalized = normalized.replace("mg", "milligram")
        normalized = normalized.replace("ml", "milliliter")
        normalized = normalized.replace("mcg", "microgram")

        # Common medical abbreviations
        normalized = normalized.replace("prn", "as needed")
        normalized = normalized.replace("qd", "daily")
        normalized = normalized.replace("bid", "twice daily")
        normalized = normalized.replace("tid", "three times daily")

        # Collapse multiple spaces again after substitutions
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def _similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity score between two strings using SequenceMatcher.

        Uses Python's difflib.SequenceMatcher for efficient similarity calculation.
        Ratio ranges from 0.0 (completely different) to 1.0 (identical).

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity ratio between 0.0 and 1.0

        Examples:
            >>> parser = AnswerParser()
            >>> parser._similarity("hello world", "hello world")
            1.0
            >>> parser._similarity("hello", "helo")
            0.8888888888888888
            >>> parser._similarity("completely", "different")
            0.18181818181818182
        """
        return SequenceMatcher(None, str1, str2).ratio()


# Convenience functions for common use cases


def normalize_answer(
    answer: str,
    domain: str = "generic",
    similarity_threshold: float = AnswerParser.DEFAULT_SIMILARITY_THRESHOLD,
) -> str:
    """
    Convenience function to normalize a single answer.

    Args:
        answer: The answer to normalize
        domain: Domain context for normalization
        similarity_threshold: Threshold for fuzzy matching (not used in this function)

    Returns:
        Normalized answer string
    """
    parser = AnswerParser(similarity_threshold=similarity_threshold)
    return parser.normalize(answer, domain=domain)


def check_answer_equivalence(
    given: str,
    ground_truth: str,
    domain: str = "generic",
    similarity_threshold: float = AnswerParser.DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    """
    Convenience function to check if two answers are equivalent.

    Args:
        given: The answer to check
        ground_truth: The reference answer
        domain: Domain context for normalization
        similarity_threshold: Minimum similarity for equivalence

    Returns:
        True if answers are equivalent, False otherwise
    """
    parser = AnswerParser(similarity_threshold=similarity_threshold)
    return parser.grade_answer(given, ground_truth, domain=domain)
