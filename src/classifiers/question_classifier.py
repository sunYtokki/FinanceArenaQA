"""Question type classifier for financial reasoning approach selection."""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of financial questions that determine reasoning approach."""

    TACTICAL_BASIC = "tactical_basic"        # Direct calculation with complete data
    TACTICAL_ASSUMPTION = "tactical_assumption"  # Calculation requiring assumptions
    CONCEPTUAL = "conceptual"               # Explanation or theory-based


@dataclass
class ClassificationResult:
    """Result of question classification."""

    question_type: QuestionType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    detected_indicators: List[str]
    missing_data_indicators: List[str] = None
    assumptions_needed: List[str] = None


class QuestionClassifier:
    """Classifies financial questions to determine reasoning approach."""

    def __init__(self):
        """Initialize classifier with keyword patterns and rules."""
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize keyword patterns for classification."""

        # Tactical-basic indicators (direct calculations with complete data)
        self.tactical_basic_keywords = {
            'calculation_terms': {
                'calculate', 'compute', 'determine', 'find', 'what is', 'how much',
                'total', 'sum', 'multiply', 'divide', 'percentage', 'percent'
            },
            'financial_metrics': {
                'roi', 'return on investment', 'npv', 'net present value', 'irr',
                'internal rate of return', 'ebitda', 'debt ratio', 'current ratio',
                'quick ratio', 'pe ratio', 'price to earnings', 'earnings per share',
                'eps', 'revenue', 'profit margin', 'gross margin', 'operating margin'
            },
            'direct_data_indicators': {
                'given', 'provided', 'the following data', 'based on', 'using',
                'with the following', 'from the table', 'from the statement'
            }
        }

        # Tactical-assumption indicators (missing data, need assumptions)
        self.tactical_assumption_keywords = {
            'assumption_triggers': {
                'assume', 'assuming', 'estimate', 'approximate', 'typical',
                'industry average', 'standard', 'reasonable', 'if we assume',
                'given that', 'suppose', 'hypothetically'
            },
            'missing_data_indicators': {
                'not provided', 'missing', 'unavailable', 'unknown', 'not given',
                'incomplete', 'partial data', 'limited information', 'no data on',
                'without knowing', 'if the data were', 'in the absence of'
            },
            'uncertainty_terms': {
                'likely', 'probably', 'might', 'could be', 'appears to be',
                'seems to', 'suggests', 'indicates', 'implies', 'estimated'
            }
        }

        # Conceptual indicators (explanations, theory, principles)
        self.conceptual_keywords = {
            'explanation_terms': {
                'explain', 'describe', 'what is', 'why', 'how does', 'what are',
                'define', 'meaning', 'concept', 'principle', 'theory', 'difference',
                'compare', 'contrast', 'advantages', 'disadvantages', 'benefits',
                'risks', 'implications'
            },
            'qualitative_analysis': {
                'analysis', 'evaluate', 'assess', 'discuss', 'consider',
                'implications', 'impact', 'effect', 'influence', 'relationship',
                'factors', 'considerations', 'approach', 'strategy', 'method'
            },
            'theoretical_terms': {
                'theory', 'model', 'framework', 'principle', 'concept',
                'methodology', 'approach', 'technique', 'best practice',
                'standard practice', 'convention', 'rule of thumb'
            }
        }

        # Financial domains that often require different approaches
        self.financial_domains = {
            'accounting': {'gaap', 'ifrs', 'accounting', 'financial statements', 'balance sheet', 'income statement'},
            'valuation': {'valuation', 'dcf', 'comparable', 'multiples', 'fair value'},
            'risk': {'risk', 'volatility', 'var', 'value at risk', 'beta', 'standard deviation'},
            'corporate_finance': {'wacc', 'cost of capital', 'capital structure', 'leverage'},
            'investment': {'portfolio', 'asset allocation', 'diversification', 'investment'},
            'derivatives': {'options', 'futures', 'swaps', 'derivatives', 'hedge'}
        }

    def classify(self, question: str) -> ClassificationResult:
        """Classify a financial question into its type.

        Args:
            question: The financial question to classify

        Returns:
            ClassificationResult with classification and confidence
        """
        question_lower = question.lower()
        detected_indicators = []

        # Score each question type
        tactical_basic_score = self._score_tactical_basic(question_lower, detected_indicators)
        tactical_assumption_score = self._score_tactical_assumption(question_lower, detected_indicators)
        conceptual_score = self._score_conceptual(question_lower, detected_indicators)

        # Determine question type based on highest score
        scores = {
            QuestionType.TACTICAL_BASIC: tactical_basic_score,
            QuestionType.TACTICAL_ASSUMPTION: tactical_assumption_score,
            QuestionType.CONCEPTUAL: conceptual_score
        }

        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]

        # Normalize confidence (simple approach)
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5

        # Generate reasoning
        reasoning = self._generate_reasoning(question, best_type, scores, detected_indicators)

        # Detect missing data indicators and assumptions for tactical-assumption type
        missing_data_indicators = []
        assumptions_needed = []

        if best_type == QuestionType.TACTICAL_ASSUMPTION:
            missing_data_indicators = self._detect_missing_data(question_lower)
            assumptions_needed = self._identify_assumptions_needed(question_lower)

        return ClassificationResult(
            question_type=best_type,
            confidence=confidence,
            reasoning=reasoning,
            detected_indicators=detected_indicators,
            missing_data_indicators=missing_data_indicators,
            assumptions_needed=assumptions_needed
        )

    def _score_tactical_basic(self, question: str, detected_indicators: List[str]) -> float:
        """Score question for tactical-basic classification."""
        score = 0.0

        # Check for calculation terms
        calc_matches = self._count_keyword_matches(question, self.tactical_basic_keywords['calculation_terms'])
        if calc_matches > 0:
            score += calc_matches * 2.0
            detected_indicators.append(f"calculation_terms: {calc_matches}")

        # Check for financial metrics
        metric_matches = self._count_keyword_matches(question, self.tactical_basic_keywords['financial_metrics'])
        if metric_matches > 0:
            score += metric_matches * 1.5
            detected_indicators.append(f"financial_metrics: {metric_matches}")

        # Check for direct data indicators
        data_matches = self._count_keyword_matches(question, self.tactical_basic_keywords['direct_data_indicators'])
        if data_matches > 0:
            score += data_matches * 1.0
            detected_indicators.append(f"direct_data_indicators: {data_matches}")

        # Bonus for numerical values (suggests concrete data)
        if re.search(r'\d+', question):
            score += 1.0
            detected_indicators.append("numerical_values_present")

        # Penalty for assumption/uncertainty indicators
        assumption_matches = self._count_keyword_matches(question, self.tactical_assumption_keywords['assumption_triggers'])
        if assumption_matches > 0:
            score -= assumption_matches * 0.5

        return max(0, score)

    def _score_tactical_assumption(self, question: str, detected_indicators: List[str]) -> float:
        """Score question for tactical-assumption classification."""
        score = 0.0

        # Check for assumption triggers
        assumption_matches = self._count_keyword_matches(question, self.tactical_assumption_keywords['assumption_triggers'])
        if assumption_matches > 0:
            score += assumption_matches * 3.0
            detected_indicators.append(f"assumption_triggers: {assumption_matches}")

        # Check for missing data indicators
        missing_matches = self._count_keyword_matches(question, self.tactical_assumption_keywords['missing_data_indicators'])
        if missing_matches > 0:
            score += missing_matches * 2.5
            detected_indicators.append(f"missing_data_indicators: {missing_matches}")

        # Check for uncertainty terms
        uncertainty_matches = self._count_keyword_matches(question, self.tactical_assumption_keywords['uncertainty_terms'])
        if uncertainty_matches > 0:
            score += uncertainty_matches * 1.5
            detected_indicators.append(f"uncertainty_terms: {uncertainty_matches}")

        # Check for calculation terms (still involves calculation)
        calc_matches = self._count_keyword_matches(question, self.tactical_basic_keywords['calculation_terms'])
        if calc_matches > 0:
            score += calc_matches * 1.0

        # Check for financial metrics
        metric_matches = self._count_keyword_matches(question, self.tactical_basic_keywords['financial_metrics'])
        if metric_matches > 0:
            score += metric_matches * 1.0

        return score

    def _score_conceptual(self, question: str, detected_indicators: List[str]) -> float:
        """Score question for conceptual classification."""
        score = 0.0

        # Check for explanation terms
        explain_matches = self._count_keyword_matches(question, self.conceptual_keywords['explanation_terms'])
        if explain_matches > 0:
            score += explain_matches * 2.5
            detected_indicators.append(f"explanation_terms: {explain_matches}")

        # Check for qualitative analysis terms
        qual_matches = self._count_keyword_matches(question, self.conceptual_keywords['qualitative_analysis'])
        if qual_matches > 0:
            score += qual_matches * 2.0
            detected_indicators.append(f"qualitative_analysis: {qual_matches}")

        # Check for theoretical terms
        theory_matches = self._count_keyword_matches(question, self.conceptual_keywords['theoretical_terms'])
        if theory_matches > 0:
            score += theory_matches * 1.5
            detected_indicators.append(f"theoretical_terms: {theory_matches}")

        # Penalty for numerical values (suggests calculation focus)
        if re.search(r'\d+', question):
            score -= 0.5

        # Penalty for calculation terms
        calc_matches = self._count_keyword_matches(question, self.tactical_basic_keywords['calculation_terms'])
        if calc_matches > 0:
            score -= calc_matches * 0.5

        return max(0, score)

    def _count_keyword_matches(self, text: str, keywords: Set[str]) -> int:
        """Count how many keywords from a set appear in the text."""
        return sum(1 for keyword in keywords if keyword in text)

    def _detect_missing_data(self, question: str) -> List[str]:
        """Detect indicators of missing data in the question."""
        indicators = []

        for keyword in self.tactical_assumption_keywords['missing_data_indicators']:
            if keyword in question:
                indicators.append(keyword)

        return indicators

    def _identify_assumptions_needed(self, question: str) -> List[str]:
        """Identify what types of assumptions might be needed."""
        assumptions = []

        # Common financial assumptions based on domain
        domain_assumptions = {
            'discount rate': ['discount', 'rate', 'cost of capital', 'wacc'],
            'growth rate': ['growth', 'increase', 'decline', 'trend'],
            'risk premium': ['risk', 'premium', 'volatility'],
            'tax rate': ['tax', 'taxes', 'after-tax'],
            'market conditions': ['market', 'economic', 'industry']
        }

        for assumption_type, keywords in domain_assumptions.items():
            if any(keyword in question for keyword in keywords):
                assumptions.append(assumption_type)

        return assumptions

    def _generate_reasoning(self, question: str, best_type: QuestionType,
                          scores: Dict[QuestionType, float],
                          detected_indicators: List[str]) -> str:
        """Generate human-readable reasoning for classification."""

        reasoning_parts = [
            f"Question classified as {best_type.value} with scores: "
            f"basic={scores[QuestionType.TACTICAL_BASIC]:.1f}, "
            f"assumption={scores[QuestionType.TACTICAL_ASSUMPTION]:.1f}, "
            f"conceptual={scores[QuestionType.CONCEPTUAL]:.1f}"
        ]

        if detected_indicators:
            reasoning_parts.append(f"Key indicators: {', '.join(detected_indicators[:3])}")

        # Add specific reasoning based on type
        if best_type == QuestionType.TACTICAL_BASIC:
            reasoning_parts.append("Direct calculation with available data expected.")
        elif best_type == QuestionType.TACTICAL_ASSUMPTION:
            reasoning_parts.append("Calculation requiring assumptions about missing data.")
        else:  # CONCEPTUAL
            reasoning_parts.append("Explanation or theoretical understanding required.")

        return " ".join(reasoning_parts)

    def batch_classify(self, questions: List[str]) -> List[ClassificationResult]:
        """Classify multiple questions at once."""
        return [self.classify(question) for question in questions]

    def get_reasoning_approach(self, question_type: QuestionType) -> Dict[str, Any]:
        """Get recommended reasoning approach for a question type."""

        approaches = {
            QuestionType.TACTICAL_BASIC: {
                "strategy": "direct_calculation",
                "tools_needed": ["financial_calculator", "data_parser"],
                "steps": [
                    "Parse and validate input data",
                    "Apply appropriate financial formulas",
                    "Calculate result with precision",
                    "Validate result reasonableness"
                ],
                "confidence_threshold": 0.9
            },
            QuestionType.TACTICAL_ASSUMPTION: {
                "strategy": "assumption_based_calculation",
                "tools_needed": ["financial_calculator", "assumption_generator", "data_parser"],
                "steps": [
                    "Identify missing data requirements",
                    "Generate reasonable assumptions",
                    "Document assumption rationale",
                    "Calculate with assumptions",
                    "Perform sensitivity analysis",
                    "Validate result with confidence bounds"
                ],
                "confidence_threshold": 0.7
            },
            QuestionType.CONCEPTUAL: {
                "strategy": "knowledge_based_explanation",
                "tools_needed": ["knowledge_base", "document_parser"],
                "steps": [
                    "Retrieve relevant financial concepts",
                    "Structure explanation logically",
                    "Provide examples if applicable",
                    "Include relevant context and caveats"
                ],
                "confidence_threshold": 0.8
            }
        }

        return approaches.get(question_type, approaches[QuestionType.CONCEPTUAL])