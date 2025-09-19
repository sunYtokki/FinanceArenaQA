"""Load financial patterns from test data with ground truth contamination guards."""

import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..rag.data_types import ContextPattern, AdjustmentType, ChunkType
from ..rag.context_patterns import ContextPatternStore

logger = logging.getLogger(__name__)


class PatternLoader:
    """Load context patterns from test data with contamination prevention."""

    # Forbidden keys that indicate ground truth contamination
    FORBIDDEN_KEYS = {
        "answer", "ground_truth", "cot", "chain_of_thought", "solution",
        "final_answer", "predicted_answer", "correct_answer", "target"
    }

    def __init__(self, pattern_store: ContextPatternStore):
        """Initialize with pattern store.

        Args:
            pattern_store: ContextPatternStore to load patterns into
        """
        self.pattern_store = pattern_store

    def load_from_jsonl(self, file_path: str, max_patterns: Optional[int] = None) -> int:
        """Load patterns from JSONL file with ground truth guards.

        Args:
            file_path: Path to JSONL file
            max_patterns: Maximum number of patterns to load

        Returns:
            Number of patterns loaded

        Raises:
            ValueError: If ground truth contamination detected
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        patterns = []
        processed_count = 0
        skipped_count = 0

        logger.info(f"Loading patterns from {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_patterns and processed_count >= max_patterns:
                    break

                try:
                    data = json.loads(line.strip())

                    # Guard against ground truth contamination
                    if self._has_ground_truth_contamination(data):
                        logger.warning(f"Line {line_num}: Skipping due to ground truth contamination")
                        skipped_count += 1
                        continue

                    # Extract pattern from data
                    pattern = self._extract_pattern(data, line_num)
                    if pattern:
                        patterns.append(pattern)
                        processed_count += 1

                        # Batch store every 100 patterns
                        if len(patterns) >= 100:
                            self.pattern_store.store_patterns(patterns)
                            patterns = []

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                    skipped_count += 1
                except Exception as e:
                    logger.warning(f"Line {line_num}: Error processing - {e}")
                    skipped_count += 1

        # Store remaining patterns
        if patterns:
            self.pattern_store.store_patterns(patterns)

        logger.info(f"Loaded {processed_count} patterns, skipped {skipped_count}")
        return processed_count

    def _has_ground_truth_contamination(self, data: Dict[str, Any]) -> bool:
        """Check if data contains ground truth contamination.

        Args:
            data: Data dictionary to check

        Returns:
            True if contamination detected
        """
        # Check top-level keys
        for key in data.keys():
            if key.lower() in self.FORBIDDEN_KEYS:
                return True

        # Check nested dictionaries
        for value in data.values():
            if isinstance(value, dict):
                if self._has_ground_truth_contamination(value):
                    return True

        return False

    def _extract_pattern(self, data: Dict[str, Any], line_num: int) -> Optional[ContextPattern]:
        """Extract context pattern from data entry.

        Args:
            data: Data dictionary
            line_num: Line number for logging

        Returns:
            ContextPattern if extractable, None otherwise
        """
        try:
            # Required fields
            context = data.get("context", "")
            question = data.get("question", "")

            if not context or not question:
                return None

            # Extract company from context
            company = self._extract_company(context)
            if not company:
                company = "unknown"

            # Detect adjustment type and signals
            adjustment_type, signals = self._detect_adjustment_type(question, context)

            # Extract section information
            section = self._extract_section(context)

            # Calculate confidence based on signal strength
            confidence = self._calculate_confidence(signals, section)

            # Create pattern
            pattern = ContextPattern(
                id=str(uuid.uuid4()),
                company=company,
                context_signals=signals,
                adjustment_type=adjustment_type,
                evidence_text=context,
                section=section,
                confidence=confidence,
                metadata={
                    "source_line": line_num,
                    "question_type": data.get("question_type", "unknown"),
                    "loaded_at": "pattern_loader"
                }
            )

            return pattern

        except Exception as e:
            logger.warning(f"Line {line_num}: Failed to extract pattern - {e}")
            return None

    def _extract_company(self, context: str) -> Optional[str]:
        """Extract company name from context."""
        context_upper = context.upper()

        # Common company patterns
        company_patterns = {
            "COSTCO": ["COSTCO WHOLESALE", "COSTCO"],
            "Apple": ["APPLE INC", "APPLE"],
            "Microsoft": ["MICROSOFT", "MSFT"],
            "Amazon": ["AMAZON", "AMZN"],
            "Google": ["ALPHABET", "GOOGLE", "GOOGL"],
            "Meta": ["META", "FACEBOOK"],
            "Tesla": ["TESLA", "TSLA"],
            "Netflix": ["NETFLIX", "NFLX"]
        }

        for company, patterns in company_patterns.items():
            for pattern in patterns:
                if pattern in context_upper:
                    return company

        return None

    def _detect_adjustment_type(self, question: str, context: str) -> tuple[AdjustmentType, List[str]]:
        """Detect adjustment type and signals from question and context.

        Args:
            question: Question text
            context: Context text

        Returns:
            Tuple of (adjustment_type, signals)
        """
        text = (question + " " + context).lower()
        signals = []

        # Lease-related signals
        lease_keywords = ["lease", "operating lease", "variable lease", "right-of-use", "rou", "asc 842"]
        if any(keyword in text for keyword in lease_keywords):
            signals.extend([kw for kw in lease_keywords if kw in text])
            return AdjustmentType.LEASE_ADDBACK, signals

        # Stock-based compensation signals
        sbc_keywords = ["stock-based compensation", "stock based compensation", "equity compensation", "rsu", "stock option"]
        if any(keyword in text for keyword in sbc_keywords):
            signals.extend([kw for kw in sbc_keywords if kw in text])
            return AdjustmentType.SBC_ADDBACK, signals

        # Restructuring signals
        restructuring_keywords = ["restructuring", "severance", "reorganization", "workforce reduction"]
        if any(keyword in text for keyword in restructuring_keywords):
            signals.extend([kw for kw in restructuring_keywords if kw in text])
            return AdjustmentType.RESTRUCTURING_EXCL, signals

        # Impairment signals
        impairment_keywords = ["impairment", "write-down", "write-off", "asset impairment"]
        if any(keyword in text for keyword in impairment_keywords):
            signals.extend([kw for kw in impairment_keywords if kw in text])
            return AdjustmentType.IMPAIRMENT_EXCL, signals

        # Depreciation/Amortization signals
        if "depreciation" in text and "amortization" in text:
            signals.extend(["depreciation", "amortization"])
            return AdjustmentType.DEPRECIATION_ADDBACK, signals
        elif "depreciation" in text:
            signals.append("depreciation")
            return AdjustmentType.DEPRECIATION_ADDBACK, signals
        elif "amortization" in text:
            signals.append("amortization")
            return AdjustmentType.AMORTIZATION_ADDBACK, signals

        # Default to OTHER with any detected signals
        if "adjusted" in text:
            signals.append("adjusted")

        return AdjustmentType.OTHER, signals

    def _extract_section(self, context: str) -> str:
        """Extract section information from context."""
        context_lower = context.lower()

        # Common section patterns
        if "note" in context_lower and "lease" in context_lower:
            return "Notes - Leases"
        elif "note" in context_lower:
            return "Notes"
        elif "cash flow" in context_lower:
            return "Cash Flows"
        elif "income statement" in context_lower:
            return "Income Statement"
        elif "balance sheet" in context_lower:
            return "Balance Sheet"
        elif "consolidated" in context_lower:
            return "Consolidated Statements"
        else:
            return "Other"

    def _calculate_confidence(self, signals: List[str], section: str) -> float:
        """Calculate confidence score based on signals and section.

        Args:
            signals: List of detected signals
            section: Section name

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from number of signals
        base_confidence = min(0.8, len(signals) * 0.2 + 0.3)

        # Section bonus
        section_bonus = 0.0
        section_lower = section.lower()
        if "note" in section_lower or "lease" in section_lower:
            section_bonus = 0.2
        elif "cash flow" in section_lower or "income" in section_lower:
            section_bonus = 0.1

        return min(1.0, base_confidence + section_bonus)

    def load_sample_patterns(self) -> int:
        """Load a few sample patterns for testing.

        Returns:
            Number of patterns loaded
        """
        sample_patterns = [
            ContextPattern(
                id=str(uuid.uuid4()),
                company="Costco",
                context_signals=["operating lease", "lease expense"],
                adjustment_type=AdjustmentType.LEASE_ADDBACK,
                evidence_text="Operating lease costs of $284 million for facilities and equipment.",
                section="Notes - Leases",
                confidence=0.9,
                metadata={"source": "sample"}
            ),
            ContextPattern(
                id=str(uuid.uuid4()),
                company="Costco",
                context_signals=["stock-based compensation", "equity compensation"],
                adjustment_type=AdjustmentType.SBC_ADDBACK,
                evidence_text="Stock-based compensation expense of $818 million.",
                section="Notes - Stock Compensation",
                confidence=0.85,
                metadata={"source": "sample"}
            )
        ]

        self.pattern_store.store_patterns(sample_patterns)
        logger.info(f"Loaded {len(sample_patterns)} sample patterns")
        return len(sample_patterns)

    def get_load_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded patterns.

        Returns:
            Dictionary with loading statistics
        """
        return self.pattern_store.get_collection_stats()