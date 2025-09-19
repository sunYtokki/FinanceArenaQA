"""Context-aware agent that integrates RAG system with simplified reasoning chain."""

import logging
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

from .core import SimplifiedAgent, Tool, ReasoningChain
from ..rag.context_patterns import ContextPatternStore, ContextRetriever
from ..rag.financial_context_detector import FinancialContextDetector

logger = logging.getLogger(__name__)


class ContextAwareAgent:
    """Agent with context-aware RAG system for financial question answering."""

    def __init__(
        self,
        model_manager,
        tools: Optional[List[Tool]] = None,
        config_path: str = "config/context_patterns.yaml",
        chroma_persist_dir: str = "./chroma_db"
    ):
        """Initialize context-aware agent.

        Args:
            model_manager: ModelManager instance for LLM access
            tools: List of available tools
            config_path: Path to configuration file
            chroma_persist_dir: Directory for ChromaDB persistence
        """
        self.model_manager = model_manager
        self.config = self._load_config(config_path)

        # Initialize RAG components
        self.pattern_store = ContextPatternStore(
            persist_directory=chroma_persist_dir,
            collection_name=self.config.get("chromadb", {}).get("collection_name", "financial_patterns")
        )

        self.context_detector = FinancialContextDetector()
        self.context_retriever = ContextRetriever(self.pattern_store)

        # Initialize core agent with context retriever
        self.core_agent = SimplifiedAgent(
            model_manager=model_manager,
            tools=tools,
            context_retriever=self.context_retriever
        )

        logger.info("Context-aware agent initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "confidence": {
                "rag_min_confidence": 0.45,
                "high_confidence": 0.8,
                "section_bonus": 0.1
            },
            "chromadb": {
                "collection_name": "financial_patterns",
                "max_results": 5
            },
            "retrieval": {
                "top_k": 5,
                "timeout_ms": 200
            }
        }

    async def answer_question(
        self,
        question: str,
        context: str = None,
        **kwargs
    ) -> ReasoningChain:
        """Answer a financial question using context-aware reasoning.

        Args:
            question: Financial question to answer
            context: Optional context text
            **kwargs: Additional parameters

        Returns:
            ReasoningChain with complete reasoning process
        """
        logger.info(f"Processing question: {question[:100]}...")

        # Use core agent with RAG integration
        result = await self.core_agent.answer_question(
            question=question,
            context=context,
            **kwargs
        )

        # Log RAG usage statistics
        self._log_rag_usage(result)

        return result

    def _log_rag_usage(self, chain: ReasoningChain) -> None:
        """Log RAG usage statistics.

        Args:
            chain: Completed reasoning chain
        """
        if not self.config.get("logging", {}).get("log_rag_decisions", False):
            return

        try:
            # Get RAG step
            context_step = chain.get_step_by_type("context_rag")  # Note: using string for now
            if context_step and context_step.output_data:
                rag_used = context_step.output_data.get("use_rag", False)
                confidence = context_step.output_data.get("rag_result", {}).get("evidence_score", 0.0)
                fallback_reason = context_step.output_data.get("fallback_reason")

                logger.info(f"RAG Usage - Used: {rag_used}, Confidence: {confidence:.2f}, "
                           f"Fallback: {fallback_reason or 'None'}")

                # Get synthesis step for transparency
                synthesis_step = chain.get_step_by_type("synthesis")  # Note: using string for now
                if synthesis_step and synthesis_step.output_data:
                    transparency = synthesis_step.output_data.get("reasoning_transparency", {})
                    adjustments = transparency.get("adjustments_applied", [])
                    if adjustments:
                        logger.info(f"Applied adjustments: {[adj.get('type') for adj in adjustments]}")

        except Exception as e:
            logger.warning(f"Failed to log RAG usage: {e}")


    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded patterns.

        Returns:
            Pattern statistics
        """
        return self.pattern_store.get_collection_stats()

    def clear_patterns(self) -> None:
        """Clear all patterns from storage."""
        logger.info("Clearing all patterns")
        self.pattern_store.clear_collection()

    async def test_context_detection(self, question: str, context: str = "") -> Dict[str, Any]:
        """Test context detection without full reasoning.

        Args:
            question: Financial question
            context: Optional context

        Returns:
            Detection results
        """
        detection_result = self.context_detector.detect_context(question, context)

        return {
            "signals": detection_result.signals,
            "adjustment_specs": [spec.to_dict() for spec in detection_result.adjustment_specs],
            "confidence": detection_result.confidence,
            "detected_amounts": detection_result.detected_amounts,
            "context_hints": detection_result.context_hints,
            "should_use_rag": self.context_detector.should_use_rag(detection_result),
            "fallback_reason": self.context_detector.get_fallback_reason(detection_result)
        }

    async def test_rag_retrieval(self, question: str, context: str = "") -> Dict[str, Any]:
        """Test RAG retrieval without full reasoning.

        Args:
            question: Financial question
            context: Optional context

        Returns:
            RAG retrieval results
        """
        rag_result = await self.context_retriever.retrieve_context(question, context)

        return {
            "rag_result": rag_result.to_dict(),
            "evidence_count": len(rag_result.evidence),
            "adjustment_count": len(rag_result.proposed_adjustments),
            "should_use_rag": rag_result.evidence_score >= 0.6,
            "method": rag_result.retrieval_method
        }

    # Delegate core agent methods
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.core_agent.add_tool(tool)

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent."""
        self.core_agent.remove_tool(tool_name)

    def list_tools(self) -> List[str]:
        """List available tools."""
        return self.core_agent.list_tools()

    def get_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Get reasoning chain by ID."""
        return self.core_agent.get_reasoning_chain(chain_id)

    def list_reasoning_chains(self) -> List[str]:
        """List all reasoning chain IDs."""
        return self.core_agent.list_reasoning_chains()


# Convenience function to create a fully configured agent
def create_context_aware_agent(
    model_manager,
    tools: Optional[List[Tool]] = None,
    config_path: str = "config/context_patterns.yaml"
) -> ContextAwareAgent:
    """Create a fully configured context-aware agent.

    Args:
        model_manager: ModelManager instance
        tools: Optional list of tools
        config_path: Path to configuration file

    Returns:
        Configured ContextAwareAgent
    """
    agent = ContextAwareAgent(
        model_manager=model_manager,
        tools=tools,
        config_path=config_path
    )

    # Note: Sample data loading removed with PatternLoader

    return agent