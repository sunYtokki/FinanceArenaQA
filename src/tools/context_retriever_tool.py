"""Context Retriever Tool for RAG-based financial context and adjustment detection."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..agent.core import Tool
from ..rag.context_patterns import ContextPatternStore, ContextRetriever
from ..rag.data_types import RAGResult

logger = logging.getLogger(__name__)


class ContextRetrieverTool(Tool):
    """Tool implementation for retrieving financial context using RAG."""

    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        collection_name: str = "financial_patterns"
    ):
        """Initialize the context retriever tool.

        Args:
            chroma_persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
        """
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self._pattern_store = None
        self._context_retriever = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "context_retriever"

    @property
    def description(self) -> str:
        """Tool description for selection."""
        return "Retrieves relevant financial context and adjustment patterns from knowledge base"

    def _ensure_initialized(self) -> None:
        """Ensure the tool is properly initialized."""
        if not self._initialized:
            try:
                # Initialize pattern store
                self._pattern_store = ContextPatternStore(
                    persist_directory=self.chroma_persist_dir,
                    collection_name=self.collection_name
                )

                # Initialize context retriever
                self._context_retriever = ContextRetriever(self._pattern_store)

                self._initialized = True
                logger.info(f"ContextRetrieverTool initialized with collection: {self.collection_name}")

            except Exception as e:
                logger.error(f"Failed to initialize ContextRetrieverTool: {e}")
                raise

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the context retrieval tool.

        Args:
            input_data: Dictionary containing:
                - question (str): Financial question to analyze
                - context (str, optional): Additional context text

        Returns:
            Dictionary containing RAGResult data:
                - signals (List[str]): Detected context signals
                - evidence (List[Dict]): Retrieved evidence with confidence scores
                - proposed_adjustments (List[Dict]): Suggested financial adjustments
                - evidence_score (float): Overall confidence in evidence
                - retrieval_method (str): Method used for retrieval
                - fallback_reason (str, optional): Reason for fallback if applicable
                - use_rag (bool): Whether RAG should be used based on confidence
        """
        try:
            # Ensure tool is initialized
            self._ensure_initialized()

            # Extract input parameters
            question = input_data.get("question", "")
            context = input_data.get("context", "")

            if not question:
                logger.warning("No question provided to ContextRetrieverTool")
                return self._create_empty_result("No question provided")

            # Perform context retrieval
            rag_result = await self._context_retriever.retrieve_context(question, context)

            # Convert RAGResult to tool-compatible dictionary format
            result_dict = rag_result.to_dict()

            # Add tool-specific metadata
            result_dict["use_rag"] = rag_result.evidence_score >= 0.6  # ConfidenceThresholds.RAG_MIN_CONFIDENCE
            result_dict["tool_name"] = self.name
            result_dict["success"] = True

            logger.info(f"Context retrieval completed: {len(rag_result.evidence)} evidence items, "
                       f"score: {rag_result.evidence_score:.2f}, use_rag: {result_dict['use_rag']}")

            return result_dict

        except Exception as e:
            logger.error(f"ContextRetrieverTool execution failed: {e}")
            return self._create_error_result(str(e))

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate tool input data.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(input_data, dict):
            logger.warning("Input data must be a dictionary")
            return False

        question = input_data.get("question")
        if not question or not isinstance(question, str):
            logger.warning("Question must be a non-empty string")
            return False

        context = input_data.get("context")
        if context is not None and not isinstance(context, str):
            logger.warning("Context must be a string if provided")
            return False

        return True

    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create an empty RAG result for fallback scenarios.

        Args:
            reason: Reason for empty result

        Returns:
            Empty RAG result dictionary
        """
        return {
            "signals": [],
            "evidence": [],
            "proposed_adjustments": [],
            "evidence_score": 0.0,
            "retrieval_method": "fallback",
            "fallback_reason": reason,
            "use_rag": False,
            "tool_name": self.name,
            "success": True
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result for failed executions.

        Args:
            error_message: Error description

        Returns:
            Error result dictionary
        """
        return {
            "signals": [],
            "evidence": [],
            "proposed_adjustments": [],
            "evidence_score": 0.0,
            "retrieval_method": "error",
            "fallback_reason": f"Tool execution failed: {error_message}",
            "use_rag": False,
            "tool_name": self.name,
            "success": False,
            "error": error_message
        }

    def is_available(self) -> bool:
        """Check if the tool is available and can be used.

        Returns:
            True if tool dependencies are available, False otherwise
        """
        try:
            # Check if ChromaDB directory exists or can be created
            chroma_path = Path(self.chroma_persist_dir)
            if not chroma_path.exists():
                chroma_path.mkdir(parents=True, exist_ok=True)

            # Try to initialize if not already done
            if not self._initialized:
                self._ensure_initialized()

            return self._initialized

        except Exception as e:
            logger.warning(f"ContextRetrieverTool not available: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool's knowledge base.

        Returns:
            Dictionary with tool statistics
        """
        try:
            if not self._initialized:
                self._ensure_initialized()

            stats = self._pattern_store.get_collection_stats()
            stats["tool_name"] = self.name
            stats["available"] = True
            return stats

        except Exception as e:
            logger.warning(f"Failed to get tool stats: {e}")
            return {
                "tool_name": self.name,
                "available": False,
                "error": str(e)
            }