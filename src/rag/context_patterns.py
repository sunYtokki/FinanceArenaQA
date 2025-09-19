"""ChromaDB-based context pattern storage and retrieval."""

import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime

from .data_types import (
    ContextPattern, RAGEvidence, RAGResult, AdjustmentSpec, AdjustmentType,
    ChunkType, ConfidenceThresholds, ADJUSTMENT_KEYWORDS
)

logger = logging.getLogger(__name__)


class ContextPatternStore:
    """ChromaDB-based storage for financial context patterns."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "financial_patterns"):
        """Initialize ChromaDB client and collection.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to store patterns
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection with metadata fields."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            # Create new collection with metadata
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Financial context patterns for adjustment detection",
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info(f"Created new collection: {self.collection_name}")

        return collection

    def store_patterns(self, patterns: List[ContextPattern]) -> None:
        """Store context patterns in ChromaDB.

        Args:
            patterns: List of context patterns to store
        """
        if not patterns:
            return

        # Prepare data for ChromaDB
        ids = [pattern.id for pattern in patterns]
        documents = [pattern.evidence_text for pattern in patterns]
        metadatas = []

        for pattern in patterns:
            metadata = {
                "company": pattern.company,
                "context_signals": json.dumps(pattern.context_signals),
                "adjustment_type": pattern.adjustment_type.value,
                "section": pattern.section,
                "confidence": pattern.confidence,
                "created_at": datetime.now().isoformat()
            }
            metadata.update(pattern.metadata)
            metadatas.append(metadata)

        # Store in ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Stored {len(patterns)} patterns in ChromaDB")
        except Exception as e:
            logger.error(f"Failed to store patterns: {e}")
            raise

    def retrieve_patterns(
        self,
        query: str,
        company: Optional[str] = None,
        adjustment_type: Optional[AdjustmentType] = None,
        n_results: int = 5
    ) -> List[RAGEvidence]:
        """Retrieve relevant patterns using hybrid search.

        Args:
            query: Search query
            company: Optional company filter
            adjustment_type: Optional adjustment type filter
            n_results: Number of results to return

        Returns:
            List of RAGEvidence objects
        """
        try:
            # Build where clause for filtering
            where_clause = {}
            if company:
                where_clause["company"] = company
            if adjustment_type:
                where_clause["adjustment_type"] = adjustment_type.value

            # Perform vector search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )

            # Convert to RAGEvidence objects
            evidence_list = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0

                    # Convert distance to confidence (1 - normalized_distance)
                    confidence = max(0.0, 1.0 - distance)

                    evidence = RAGEvidence(
                        id=results['ids'][0][i],
                        text=doc,
                        page=metadata.get('page', 0),
                        section=metadata.get('section', 'unknown'),
                        chunk_type=ChunkType.TEXT,  # Default, could be enhanced
                        confidence=confidence,
                        company=metadata.get('company'),
                        metadata=metadata
                    )
                    evidence_list.append(evidence)

            return evidence_list

        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        company: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        n_results: int = 5
    ) -> List[RAGEvidence]:
        """Perform hybrid search with keyword filtering.

        Args:
            query: Search query
            company: Optional company filter
            keywords: Optional keywords to filter by
            n_results: Number of results to return

        Returns:
            List of RAGEvidence objects
        """
        # First, try keyword filtering if keywords provided
        if keywords:
            # Build a where_document filter for keywords
            keyword_filter = " OR ".join([f'"{kw}"' for kw in keywords])

            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results * 2,  # Get more to filter
                    where_document={"$contains": keyword_filter} if keywords else None,
                    where={"company": company} if company else None
                )

                # Process results
                evidence_list = []
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                        distance = results['distances'][0][i] if results['distances'] else 1.0

                        # Calculate confidence with keyword bonus
                        confidence = max(0.0, 1.0 - distance)

                        # Add keyword bonus
                        if keywords:
                            doc_lower = doc.lower()
                            keyword_matches = sum(1 for kw in keywords if kw.lower() in doc_lower)
                            if keyword_matches > 0:
                                confidence = min(1.0, confidence + 0.1 * keyword_matches)

                        evidence = RAGEvidence(
                            id=results['ids'][0][i],
                            text=doc,
                            page=metadata.get('page', 0),
                            section=metadata.get('section', 'unknown'),
                            chunk_type=ChunkType.TEXT,
                            confidence=confidence,
                            company=metadata.get('company'),
                            metadata=metadata
                        )
                        evidence_list.append(evidence)

                # Sort by confidence and return top results
                evidence_list.sort(key=lambda x: x.confidence, reverse=True)
                return evidence_list[:n_results]

            except Exception as e:
                logger.warning(f"Keyword search failed, falling back to vector search: {e}")

        # Fallback to standard vector search
        return self.retrieve_patterns(query, company, None, n_results)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_patterns": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_patterns": 0, "collection_name": self.collection_name}

    def clear_collection(self) -> None:
        """Clear all patterns from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")


class ContextRetriever:
    """High-level interface for context retrieval and adjustment detection."""

    def __init__(self, pattern_store: ContextPatternStore):
        """Initialize with pattern store.

        Args:
            pattern_store: ContextPatternStore instance
        """
        self.pattern_store = pattern_store

    async def retrieve_context(self, question: str, context: str = "") -> RAGResult:
        """Retrieve context and detect adjustments for a financial question.

        Args:
            question: Financial question
            context: Optional context text

        Returns:
            RAGResult with evidence and proposed adjustments
        """
        try:
            # Detect signals in question and context
            signals = self._detect_signals(question + " " + context)

            # Determine search keywords based on signals
            keywords = self._get_keywords_for_signals(signals)

            # Extract company if possible
            company = self._extract_company(context)

            # Perform hybrid search
            evidence = self.pattern_store.hybrid_search(
                query=question,
                company=company,
                keywords=keywords,
                n_results=5
            )

            # Generate adjustment specifications
            adjustments = self._generate_adjustments(evidence, signals)

            # Calculate overall evidence score
            evidence_score = self._calculate_evidence_score(evidence, signals)

            # Determine if we should use RAG or fallback
            use_rag = evidence_score >= ConfidenceThresholds.RAG_MIN_CONFIDENCE

            return RAGResult(
                signals=signals,
                evidence=evidence,
                proposed_adjustments=adjustments if use_rag else [],
                evidence_score=evidence_score,
                retrieval_method="hybrid_search",
                fallback_reason=None if use_rag else f"Evidence score {evidence_score:.2f} below threshold {ConfidenceThresholds.RAG_MIN_CONFIDENCE}"
            )

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return RAGResult(
                signals=[],
                evidence=[],
                proposed_adjustments=[],
                evidence_score=0.0,
                retrieval_method="error_fallback",
                fallback_reason=f"Retrieval error: {str(e)}"
            )

    def _detect_signals(self, text: str) -> List[str]:
        """Detect context signals in text."""
        text_lower = text.lower()
        signals = []

        # Check for adjustment-related keywords
        for adj_type, keywords in ADJUSTMENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    signals.append(f"{adj_type.value}:{keyword}")

        # Check for specific patterns
        if "adjusted" in text_lower:
            signals.append("adjusted_metric")
        if "ebitda" in text_lower:
            signals.append("ebitda_calculation")
        if "margin" in text_lower:
            signals.append("margin_calculation")

        return list(set(signals))  # Remove duplicates

    def _get_keywords_for_signals(self, signals: List[str]) -> List[str]:
        """Get search keywords based on detected signals."""
        keywords = []

        for signal in signals:
            if "lease_addback" in signal:
                keywords.extend(ADJUSTMENT_KEYWORDS[AdjustmentType.LEASE_ADDBACK])
            elif "sbc_addback" in signal:
                keywords.extend(ADJUSTMENT_KEYWORDS[AdjustmentType.SBC_ADDBACK])
            elif "restructuring" in signal:
                keywords.extend(ADJUSTMENT_KEYWORDS[AdjustmentType.RESTRUCTURING_EXCL])
            elif "impairment" in signal:
                keywords.extend(ADJUSTMENT_KEYWORDS[AdjustmentType.IMPAIRMENT_EXCL])

        return list(set(keywords))  # Remove duplicates

    def _extract_company(self, context: str) -> Optional[str]:
        """Extract company name from context."""
        # Simple extraction - look for common patterns
        context_upper = context.upper()

        # Common company indicators
        if "COSTCO" in context_upper:
            return "Costco"
        elif "APPLE" in context_upper:
            return "Apple"
        elif "MICROSOFT" in context_upper:
            return "Microsoft"

        return None

    def _generate_adjustments(self, evidence: List[RAGEvidence], signals: List[str]) -> List[AdjustmentSpec]:
        """Generate adjustment specifications based on evidence and signals."""
        adjustments = []

        if not evidence:
            return adjustments

        # Group evidence by potential adjustment type
        lease_evidence = [e for e in evidence if any("lease" in s for s in signals)]
        sbc_evidence = [e for e in evidence if any("sbc" in s for s in signals)]

        # Generate lease adjustment if evidence found
        if lease_evidence:
            avg_confidence = sum(e.confidence for e in lease_evidence) / len(lease_evidence)
            adjustments.append(AdjustmentSpec(
                type=AdjustmentType.LEASE_ADDBACK,
                scope="operating_leases",
                basis="ASC 842 lease treatment",
                source_ids=[e.id for e in lease_evidence],
                confidence=avg_confidence,
                description="Add back lease expenses for adjusted EBITDA"
            ))

        # Generate SBC adjustment if evidence found
        if sbc_evidence:
            avg_confidence = sum(e.confidence for e in sbc_evidence) / len(sbc_evidence)
            adjustments.append(AdjustmentSpec(
                type=AdjustmentType.SBC_ADDBACK,
                scope="all_stock_compensation",
                basis="Non-cash expense adjustment",
                source_ids=[e.id for e in sbc_evidence],
                confidence=avg_confidence,
                description="Add back stock-based compensation"
            ))

        return adjustments

    def _calculate_evidence_score(self, evidence: List[RAGEvidence], signals: List[str]) -> float:
        """Calculate overall evidence score."""
        if not evidence:
            return 0.0

        # Base score from evidence confidence
        base_score = sum(e.confidence for e in evidence) / len(evidence)

        # Bonus for signal matches
        signal_bonus = 0.0
        if signals:
            signal_bonus = min(0.2, len(signals) * 0.05)

        # Section bonus
        section_bonus = 0.0
        for e in evidence:
            section_lower = e.section.lower()
            for section_name, weight in ConfidenceThresholds.SECTION_WEIGHTS.items():
                if section_name in section_lower:
                    section_bonus += weight * ConfidenceThresholds.SECTION_BONUS
                    break

        section_bonus = min(0.3, section_bonus / len(evidence))

        return min(1.0, base_score + signal_bonus + section_bonus)