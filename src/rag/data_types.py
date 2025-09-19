"""Structured data types for context-aware RAG system."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ChunkType(Enum):
    """Type of document chunk."""
    TABLE = "table"
    FOOTNOTE = "footnote"
    TEXT = "text"
    HEADER = "header"


class AdjustmentType(Enum):
    """Type of financial adjustment."""
    LEASE_ADDBACK = "lease_addback"
    SBC_ADDBACK = "sbc_addback"
    RESTRUCTURING_EXCL = "restructuring_excl"
    IMPAIRMENT_EXCL = "impairment_excl"
    DEPRECIATION_ADDBACK = "depreciation_addback"
    AMORTIZATION_ADDBACK = "amortization_addback"
    OTHER = "other"


@dataclass
class RAGEvidence:
    """Evidence retrieved from RAG system with metadata."""

    id: str
    text: str
    page: int
    section: str
    chunk_type: ChunkType
    confidence: float
    units: Optional[str] = None
    period: Optional[str] = None
    source_path: Optional[str] = None
    company: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "page": self.page,
            "section": self.section,
            "chunk_type": self.chunk_type.value,
            "confidence": self.confidence,
            "units": self.units,
            "period": self.period,
            "source_path": self.source_path,
            "company": self.company,
            "metadata": self.metadata
        }


@dataclass
class AdjustmentSpec:
    """Specification for a financial adjustment."""

    type: AdjustmentType
    scope: str  # Details like "operating+variable" or "all_leases"
    basis: str  # e.g., "ASC 842", "non-GAAP convention"
    source_ids: List[str]  # IDs of supporting evidence
    confidence: float
    amount: Optional[float] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "scope": self.scope,
            "basis": self.basis,
            "source_ids": self.source_ids,
            "confidence": self.confidence,
            "amount": self.amount,
            "description": self.description
        }


@dataclass
class RAGResult:
    """Complete result from RAG retrieval and processing."""

    signals: List[str]  # Detected context signals
    evidence: List[RAGEvidence]  # Supporting evidence
    proposed_adjustments: List[AdjustmentSpec]  # Suggested adjustments
    evidence_score: float  # Overall confidence in evidence
    retrieval_method: str = "hybrid"  # Method used for retrieval
    fallback_reason: Optional[str] = None  # Why fallback was needed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signals": self.signals,
            "evidence": [e.to_dict() for e in self.evidence],
            "proposed_adjustments": [a.to_dict() for a in self.proposed_adjustments],
            "evidence_score": self.evidence_score,
            "retrieval_method": self.retrieval_method,
            "fallback_reason": self.fallback_reason
        }


@dataclass
class ContextPattern:
    """Pattern for storing in ChromaDB collection."""

    id: str
    company: str
    context_signals: List[str]
    adjustment_type: AdjustmentType
    evidence_text: str
    section: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage."""
        return {
            "id": self.id,
            "company": self.company,
            "context_signals": self.context_signals,
            "adjustment_type": self.adjustment_type.value,
            "evidence_text": self.evidence_text,
            "section": self.section,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


# Confidence thresholds for decision making
class ConfidenceThresholds:
    """Confidence thresholds for RAG vs fallback decisions."""

    RAG_MIN_CONFIDENCE = 0.6  # Minimum to use RAG results
    HIGH_CONFIDENCE = 0.8     # High confidence threshold
    SECTION_BONUS = 0.1       # Bonus for evidence from key sections

    # Section importance weights
    SECTION_WEIGHTS = {
        "leases": 0.9,
        "note": 0.8,
        "footnote": 0.8,
        "cash flows": 0.7,
        "income statement": 0.6,
        "balance sheet": 0.6,
        "md&a": 0.4,
        "other": 0.3
    }


# Keywords for different adjustment types
ADJUSTMENT_KEYWORDS = {
    AdjustmentType.LEASE_ADDBACK: [
        "operating lease", "variable lease", "lease expense", "lease cost",
        "right-of-use", "rou asset", "lease liability", "asc 842"
    ],
    AdjustmentType.SBC_ADDBACK: [
        "stock-based compensation", "stock based compensation", "equity compensation",
        "restricted stock", "stock options", "rsu", "stock award"
    ],
    AdjustmentType.RESTRUCTURING_EXCL: [
        "restructuring", "severance", "reorganization", "facility closure",
        "workforce reduction", "restructuring charge"
    ],
    AdjustmentType.IMPAIRMENT_EXCL: [
        "impairment", "impairment charge", "asset impairment", "goodwill impairment",
        "write-down", "write-off"
    ],
    AdjustmentType.DEPRECIATION_ADDBACK: [
        "depreciation", "depreciation expense", "accumulated depreciation"
    ],
    AdjustmentType.AMORTIZATION_ADDBACK: [
        "amortization", "amortization expense", "intangible amortization"
    ]
}