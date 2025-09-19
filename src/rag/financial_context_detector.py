"""Financial context detector with structured adjustment detection and confidence scoring."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .data_types import (
    AdjustmentSpec, AdjustmentType, ConfidenceThresholds, ADJUSTMENT_KEYWORDS
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of context detection."""
    signals: List[str]
    adjustment_specs: List[AdjustmentSpec]
    confidence: float
    detected_amounts: Dict[str, float]
    context_hints: List[str]


class FinancialContextDetector:
    """Detect financial context signals and generate adjustment specifications."""

    def __init__(self):
        """Initialize the detector."""
        self.amount_pattern = re.compile(r'\$?[\d,]+\.?\d*\s*(?:million|billion|thousand|m|b|k)?')
        self.percentage_pattern = re.compile(r'\d+\.?\d*\s*%')

    def detect_context(
        self,
        question: str,
        context: str = "",
        company: Optional[str] = None
    ) -> DetectionResult:
        """Detect financial context and generate adjustment specifications.

        Args:
            question: Financial question
            context: Optional context text
            company: Optional company name

        Returns:
            DetectionResult with signals and adjustment specs
        """
        combined_text = f"{question} {context}".lower()

        # Detect signals
        signals = self._detect_signals(combined_text)

        # Extract amounts
        amounts = self._extract_amounts(combined_text)

        # Detect context hints
        hints = self._detect_context_hints(combined_text)

        # Generate adjustment specifications
        adjustment_specs = self._generate_adjustment_specs(
            signals, amounts, hints, company
        )

        # Calculate overall confidence
        confidence = self._calculate_detection_confidence(
            signals, adjustment_specs, hints
        )

        return DetectionResult(
            signals=signals,
            adjustment_specs=adjustment_specs,
            confidence=confidence,
            detected_amounts=amounts,
            context_hints=hints
        )

    def _detect_signals(self, text: str) -> List[str]:
        """Detect financial adjustment signals in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected signals
        """
        signals = []

        # Check for each adjustment type's keywords
        for adj_type, keywords in ADJUSTMENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    signals.append(f"{adj_type.value}:{keyword}")

        # Check for specific calculation types
        if "adjusted" in text:
            if "ebitda" in text:
                signals.append("adjusted_ebitda")
            if "ebit" in text:
                signals.append("adjusted_ebit")
            if "margin" in text:
                signals.append("adjusted_margin")

        # Check for context indicators
        if "broken out" in text or "detailed" in text:
            signals.append("detailed_breakdown")

        if "expense" in text and ("lease" in text or "rent" in text):
            signals.append("lease_expense_context")

        if "compensation" in text and "stock" in text:
            signals.append("stock_compensation_context")

        return list(set(signals))  # Remove duplicates

    def _extract_amounts(self, text: str) -> Dict[str, float]:
        """Extract monetary amounts from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of detected amounts
        """
        amounts = {}

        # Find all monetary amounts
        amount_matches = self.amount_pattern.findall(text)

        for match in amount_matches:
            try:
                # Clean up the match
                clean_match = re.sub(r'[^\d.,]', '', match)
                if clean_match:
                    # Convert to float
                    amount = float(clean_match.replace(',', ''))

                    # Apply multipliers
                    if 'million' in match.lower() or 'm' in match.lower():
                        amount *= 1_000_000
                    elif 'billion' in match.lower() or 'b' in match.lower():
                        amount *= 1_000_000_000
                    elif 'thousand' in match.lower() or 'k' in match.lower():
                        amount *= 1_000

                    # Store with context
                    context_key = self._get_amount_context(text, match)
                    amounts[context_key] = amount

            except ValueError:
                continue

        return amounts

    def _get_amount_context(self, text: str, amount_match: str) -> str:
        """Get context for an amount in text.

        Args:
            text: Full text
            amount_match: The amount string that was matched

        Returns:
            Context key for the amount
        """
        # Find the position of the amount in text
        amount_pos = text.find(amount_match.lower())
        if amount_pos == -1:
            return "unknown_amount"

        # Look for context words before the amount
        context_window = text[max(0, amount_pos - 50):amount_pos + 50]

        if "lease" in context_window:
            return "lease_amount"
        elif "stock" in context_window and "compensation" in context_window:
            return "sbc_amount"
        elif "depreciation" in context_window:
            return "depreciation_amount"
        elif "amortization" in context_window:
            return "amortization_amount"
        elif "restructuring" in context_window:
            return "restructuring_amount"
        else:
            return "general_amount"

    def _detect_context_hints(self, text: str) -> List[str]:
        """Detect context hints that suggest specific adjustment approaches.

        Args:
            text: Text to analyze

        Returns:
            List of context hints
        """
        hints = []

        # Lease-specific hints
        if "lease expenses broken out" in text or "lease costs detailed" in text:
            hints.append("lease_detail_available")

        if "operating lease" in text and "variable lease" in text:
            hints.append("multiple_lease_types")

        # Stock compensation hints
        if "stock-based compensation" in text and "non-cash" in text:
            hints.append("sbc_non_cash_emphasis")

        # Calculation type hints
        if "calculate" in text or "compute" in text:
            hints.append("calculation_required")

        if "margin" in text:
            hints.append("margin_calculation")

        # Assumption hints
        if "assume" in text or "estimate" in text:
            hints.append("assumption_required")

        # GAAP vs non-GAAP hints
        if "non-gaap" in text or "adjusted" in text:
            hints.append("non_gaap_context")

        return hints

    def _generate_adjustment_specs(
        self,
        signals: List[str],
        amounts: Dict[str, float],
        hints: List[str],
        company: Optional[str]
    ) -> List[AdjustmentSpec]:
        """Generate adjustment specifications based on detected signals.

        Args:
            signals: Detected signals
            amounts: Detected amounts
            hints: Context hints
            company: Company name

        Returns:
            List of AdjustmentSpec objects
        """
        specs = []

        # Lease adjustments
        if any("lease_addback" in s for s in signals):
            confidence = self._calculate_lease_confidence(signals, hints, amounts)

            # Determine scope based on signals
            scope = "operating_leases"
            if any("variable lease" in s for s in signals):
                scope = "operating_and_variable_leases"

            spec = AdjustmentSpec(
                type=AdjustmentType.LEASE_ADDBACK,
                scope=scope,
                basis="ASC 842 lease treatment for non-GAAP metrics",
                source_ids=[],  # Will be filled by retrieval
                confidence=confidence,
                amount=amounts.get("lease_amount"),
                description=f"Add back {scope.replace('_', ' ')} for adjusted EBITDA calculation"
            )
            specs.append(spec)

        # Stock-based compensation adjustments
        if any("sbc_addback" in s for s in signals):
            confidence = self._calculate_sbc_confidence(signals, hints, amounts)

            spec = AdjustmentSpec(
                type=AdjustmentType.SBC_ADDBACK,
                scope="all_stock_compensation",
                basis="Non-cash expense exclusion",
                source_ids=[],
                confidence=confidence,
                amount=amounts.get("sbc_amount"),
                description="Add back stock-based compensation as non-cash expense"
            )
            specs.append(spec)

        # Depreciation and Amortization
        if any("depreciation" in s for s in signals) and "adjusted_ebitda" in signals:
            confidence = 0.9  # High confidence for D&A in EBITDA

            spec = AdjustmentSpec(
                type=AdjustmentType.DEPRECIATION_ADDBACK,
                scope="depreciation_and_amortization",
                basis="Standard EBITDA calculation",
                source_ids=[],
                confidence=confidence,
                amount=amounts.get("depreciation_amount"),
                description="Add back depreciation and amortization for EBITDA"
            )
            specs.append(spec)

        # Company-specific adjustments
        if company:
            company_specs = self._get_company_specific_adjustments(
                company, signals, hints, amounts
            )
            specs.extend(company_specs)

        return specs

    def _calculate_lease_confidence(
        self,
        signals: List[str],
        hints: List[str],
        amounts: Dict[str, float]
    ) -> float:
        """Calculate confidence for lease adjustments."""
        base_confidence = 0.6

        # Boost confidence if detailed lease info available
        if "lease_detail_available" in hints:
            base_confidence += 0.2

        # Boost if multiple lease types mentioned
        if "multiple_lease_types" in hints:
            base_confidence += 0.1

        # Boost if lease amount detected
        if "lease_amount" in amounts:
            base_confidence += 0.1

        # Boost for strong lease signals
        lease_signals = [s for s in signals if "lease" in s]
        if len(lease_signals) >= 2:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _calculate_sbc_confidence(
        self,
        signals: List[str],
        hints: List[str],
        amounts: Dict[str, float]
    ) -> float:
        """Calculate confidence for stock-based compensation adjustments."""
        base_confidence = 0.7

        # Boost if non-cash emphasized
        if "sbc_non_cash_emphasis" in hints:
            base_confidence += 0.1

        # Boost if SBC amount detected
        if "sbc_amount" in amounts:
            base_confidence += 0.1

        # Strong SBC signals
        sbc_signals = [s for s in signals if "sbc" in s or "stock" in s]
        if len(sbc_signals) >= 2:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _get_company_specific_adjustments(
        self,
        company: str,
        signals: List[str],
        hints: List[str],
        amounts: Dict[str, float]
    ) -> List[AdjustmentSpec]:
        """Get company-specific adjustment patterns.

        Args:
            company: Company name
            signals: Detected signals
            hints: Context hints
            amounts: Detected amounts

        Returns:
            List of company-specific adjustment specs
        """
        specs = []

        # Costco-specific patterns
        if company.lower() == "costco":
            # Costco often emphasizes lease adjustments
            if any("lease" in s for s in signals) and "lease_detail_available" in hints:
                # High confidence for Costco lease adjustments when detailed
                for spec in specs:
                    if spec.type == AdjustmentType.LEASE_ADDBACK:
                        spec.confidence = min(1.0, spec.confidence + 0.1)
                        spec.basis += " (Costco pattern: detailed lease disclosure)"

        # Tech companies - emphasize SBC
        tech_companies = ["apple", "microsoft", "google", "meta", "amazon"]
        if company.lower() in tech_companies:
            if any("sbc" in s for s in signals):
                # Boost SBC confidence for tech companies
                for spec in specs:
                    if spec.type == AdjustmentType.SBC_ADDBACK:
                        spec.confidence = min(1.0, spec.confidence + 0.15)
                        spec.basis += f" ({company} pattern: significant SBC usage)"

        return specs

    def _calculate_detection_confidence(
        self,
        signals: List[str],
        adjustment_specs: List[AdjustmentSpec],
        hints: List[str]
    ) -> float:
        """Calculate overall detection confidence.

        Args:
            signals: Detected signals
            adjustment_specs: Generated adjustment specs
            hints: Context hints

        Returns:
            Overall confidence score
        """
        if not signals and not adjustment_specs:
            return 0.0

        # Base confidence from signals
        signal_confidence = min(0.8, len(signals) * 0.1 + 0.2)

        # Adjustment spec confidence
        if adjustment_specs:
            spec_confidence = sum(spec.confidence for spec in adjustment_specs) / len(adjustment_specs)
        else:
            spec_confidence = 0.0

        # Hint bonus
        hint_bonus = min(0.2, len(hints) * 0.05)

        # Weighted average
        overall_confidence = (
            signal_confidence * 0.4 +
            spec_confidence * 0.5 +
            hint_bonus * 0.1
        )

        return min(1.0, overall_confidence)

    def should_use_rag(self, detection_result: DetectionResult) -> bool:
        """Determine if RAG should be used based on detection results.

        Args:
            detection_result: Detection result

        Returns:
            True if RAG should be used
        """
        return detection_result.confidence >= ConfidenceThresholds.RAG_MIN_CONFIDENCE

    def get_fallback_reason(self, detection_result: DetectionResult) -> Optional[str]:
        """Get reason for falling back to assumptions.

        Args:
            detection_result: Detection result

        Returns:
            Fallback reason if applicable
        """
        if detection_result.confidence < ConfidenceThresholds.RAG_MIN_CONFIDENCE:
            return f"Detection confidence {detection_result.confidence:.2f} below threshold {ConfidenceThresholds.RAG_MIN_CONFIDENCE}"

        if not detection_result.signals:
            return "No context signals detected"

        if not detection_result.adjustment_specs:
            return "No adjustment specifications generated"

        return None