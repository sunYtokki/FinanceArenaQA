"""Unified Financial Agent with tool-based RAG architecture."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from .core import Tool, ReasoningChain, ReasoningStep, StepType, StepStatus
from ..rag.financial_rag import FinancialRAG, build_llm_context

logger = logging.getLogger(__name__)


class FinancialAgent:
    """Unified agent that combines financial reasoning with tool-based RAG architecture."""

    def __init__(
        self,
        model_manager,
        config: Optional[Dict[str, Any]] = None,
        enable_rag: bool = True
    ):
        """Initialize the financial agent with model manager and tools.

        Args:
            model_manager: ModelManager instance for LLM access
            config: Configuration dictionary with RAG and tool settings
            enable_rag: Whether to enable RAG tools (default: True)
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.enable_rag = enable_rag
        self.tools: Dict[str, Tool] = {}
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self._cleanup_called = False
        self.financial_rag: Optional[FinancialRAG] = None

        # Validate configuration
        self._validate_config()

        # Initialize RAG and tools
        self._initialize_financial_rag()
        # self._initialize_tools()

        logger.info(f"FinancialAgent initialized with {len(self.tools)} tools, RAG enabled: {enable_rag}")

    def cleanup(self) -> None:
        """Clean up agent resources including model manager connections."""
        try:
            # Mark cleanup as called
            self._cleanup_called = True

            # Clean up model manager if it has cleanup methods
            if hasattr(self.model_manager, 'cleanup'):
                self.model_manager.cleanup()
            elif hasattr(self.model_manager, 'close'):
                self.model_manager.close()
            elif hasattr(self.model_manager, 'close_all'):
                self.model_manager.close_all()

            # Clean up tools that might have resources
            for tool in self.tools.values():
                if hasattr(tool, 'cleanup'):
                    tool.cleanup()

            logger.info("FinancialAgent cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor - avoid async operations during garbage collection."""
        # NOTE: Cannot reliably run async cleanup during destruction
        # as event loop may be closed. Cleanup should be called explicitly
        # before destruction via context manager or explicit cleanup() call.
        try:
            # Log warning if resources might not be cleaned up
            if hasattr(self, '_cleanup_called') and not self._cleanup_called:
                logger.warning(
                    "FinancialAgent destroyed without explicit cleanup. "
                    "Call 'agent.cleanup()' explicitly before destruction."
                )
        except Exception:
            pass  # Ignore all errors in destructor

    def _validate_config(self) -> None:
        """Validate the configuration and apply simple defaults."""
        if not self.config:
            self.config = {}

        # Simple RAG configuration defaults
        if self.enable_rag:
            rag_config = self.config.setdefault("rag", {})
            rag_config.setdefault("enabled", True)
            rag_config.setdefault("chroma_persist_dir", "./chroma_db")
            rag_config.setdefault("collection_name", "financial_knowledge")
            rag_config.setdefault("max_results", 5)

            logger.info("Simple RAG configuration applied")

    def _initialize_financial_rag(self) -> None:
        """Initialize FinancialRAG for enhanced context retrieval."""
        if not self.enable_rag:
            logger.info("RAG disabled, skipping FinancialRAG initialization")
            return

        try:
            rag_config = self.config.get("rag", {})
            persist_directory = rag_config.get("chroma_persist_dir", "./chroma_db")
            collection_name = rag_config.get("collection_name", "financial_knowledge")

            self.financial_rag = FinancialRAG()

            if self.financial_rag.is_available():
                logger.info(f"FinancialRAG initialized successfully with collection '{collection_name}'")
            else:
                logger.warning("FinancialRAG initialization failed, continuing without enhanced RAG")
                self.financial_rag = None

        except Exception as e:
            logger.error(f"Failed to initialize FinancialRAG: {e}")
            self.financial_rag = None

    def answer_question(self, question: str, **kwargs) -> ReasoningChain:
        """Answer a financial question using simplified 3-step reasoning: Analysis → RAG → Synthesis.

        Args:
            question: The financial question to answer
            **kwargs: Additional parameters including context

        Returns:
            ReasoningChain containing the complete reasoning process
        """

        # Parse question to extract clean question (ignore any provided context)
        parsed_question, parsed_context = self._parse_question_and_context(question)
        question = parsed_question  # Use the clean question for analysis

        # Create new reasoning chain
        chain = ReasoningChain(question=question)
        self.reasoning_chains[chain.id] = chain

        start_time = datetime.now()

        try:
            # Step 1: Analysis - Question analysis and requirements identification
            analysis_step = self._execute_analysis(question)
            chain.add_step(analysis_step)

            # Step 2: RAG Search - Search Kaggle Q&A dataset for similar reasoning examples
            if analysis_step.status == StepStatus.COMPLETED:
                context_step = self._enhance_context(question=question, provided_context=parsed_context, analysis_output=analysis_step.output_data)
                chain.add_step(context_step)

            # Step 3: Synthesis - Generate final answer directly from analysis and RAG context
            synthesis_step = self._synthesis_answer(chain, parsed_context)
            chain.add_step(synthesis_step)

        except Exception as e:
            logger.error(f"Critical error in reasoning chain {chain.id}: {e}")
            # Add error step and ensure we have a final answer
            error_step = ReasoningStep(
                step_type=StepType.SYNTHESIS,
                status=StepStatus.FAILED,
                output_data=f"Error: {str(e)}"
            )
            chain.add_step(error_step)

        # Calculate timing and ensure final answer
        end_time = datetime.now()
        chain.total_execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        if chain.final_answer:
            logger.info(f"Reasoning chain {chain.id} completed in {chain.total_execution_time_ms}ms")

        return chain


    def _execute_analysis(self, question: str) -> ReasoningStep:
        """Step 1: Enhanced analysis with source routing."""
        step = ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="Enhanced question analysis with source routing",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": question}
        )

        start_time = datetime.now()

        try:
            # Import prompts
            from .prompts import get_enhanced_analysis_prompt

            # Use enhanced analysis prompt
            analysis_prompt = get_enhanced_analysis_prompt(question)

            response_content = self._safe_model_generate(
                analysis_prompt,
                "Enhanced analysis failed"
            )

            # Parse JSON response from LLM
            try:
                content = response_content

                # If the model uses <output>, only keep what's inside it
                if "<output>" in content:
                    content = content.split("<output>", 1)[1]
                    if "</output>" in content:
                        content = content.split("</output>", 1)[0]
                    content = content.strip()

                # Fallback: find the first top-level JSON object in the (possibly trimmed) content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end].strip()
                    parsed_analysis = json.loads(json_str)

                    step.output_data = {
                        "raw_response": response_content,
                        "analysis": parsed_analysis,
                        "question_type": parsed_analysis.get("question_type", "basic"),
                        "financial_concepts": parsed_analysis.get("financial_concepts", []),
                        "required_context": parsed_analysis.get("required_context", []),
                        "search_keywords": parsed_analysis.get("search_keywords", []),
                        "recommended_sources": parsed_analysis.get("recommended_sources", ["both"])
                    }
                    step.status = StepStatus.COMPLETED            

            except Exception as parse_error:
                # If JSON parsing fails, treat as analysis failure
                step.status = StepStatus.FAILED
                step.error_message = f"Failed to parse LLM analysis response: {parse_error}"
                step.output_data = {
                    "raw_response": response_content,
                    "error": str(parse_error)
                }

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.output_data = {
                "error": str(e)
            }

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    def _enhance_context(
        self,
        question: str,
        provided_context: Optional[str],
        analysis_output: Dict[str, Any]
    ) -> ReasoningStep:
        """Step 2: RAG search to enhance reasoning context."""
        step = ReasoningStep(
            step_type=StepType.CONTEXT_RAG,
            description="RAG search to enhance reasoning context",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": question}
        )

        start_time = datetime.now()

        try:
            # Always search RAG dataset for similar questions and reasoning
            if not self.financial_rag or not self.financial_rag.is_available():
                step.output_data = {
                    "context": "",
                    "context_source": "empty_no_rag",
                    "search_results": [],
                    "total_results": 0,
                    "strategy": "no_rag_fallback"
                }
                step.status = StepStatus.COMPLETED
                logger.warning("FinancialRAG not available, using empty context")
                return step

            # Prepare search query using analysis output
            search_query = question
            enhanced_search_terms = []

            if analysis_output:
                # Extract search keywords and financial concepts from analysis
                search_keywords = analysis_output.get("search_keywords", [])
                financial_concepts = analysis_output.get("financial_concepts", [])

                # Combine concepts and keywords for enhanced search
                if search_keywords:
                    enhanced_search_terms.extend(search_keywords)
                if financial_concepts:
                    enhanced_search_terms.extend(financial_concepts)

                # If we have analysis-derived terms, create enhanced query
                if enhanced_search_terms:
                    # Use the enhanced search functionality with multiple keywords
                    all_terms = [question] + enhanced_search_terms
                    search_query = all_terms
                    logger.info(f"Enhanced search with {len(enhanced_search_terms)} additional terms from analysis")

            # Search Kaggle Q&A dataset for similar financial questions
            search_results = self.financial_rag.search(search_query, source_filter='qa_dataset')

            if search_results:
                if provided_context:
                    reasoning_contexts = build_llm_context(search_results, question, provided_context, self.model_manager)

                else:
                    # Extract reasoning context from Kaggle Q&A pairs
                    reasoning_contexts = []
                    for result in search_results:
                        # Get reasoning context from metadata
                        metadata = result.get('metadata', {})
                        reasoning_context = metadata.get('reasoning_context', '')

                        if reasoning_context and reasoning_context.strip():
                            reasoning_contexts.append({
                                'question': metadata.get('question', ''),
                                'answer': metadata.get('answer', ''),
                                'reasoning': reasoning_context,
                                'score': result.get('score', 0)
                            })

                    # Build combined reasoning context
                    if reasoning_contexts:
                        reasoning_contexts = self._build_reasoning_context(reasoning_contexts)
                    else:
                        reasoning_contexts = "Found similar questions but no reasoning context available."

                step.output_data = {
                    # "context": combined_context,
                    "search_results": search_results,
                    "reasoning_contexts": reasoning_contexts,
                    "total_results": len(search_results),
                    "search_query": search_query,
                    "enhanced_search_used": isinstance(search_query, list),
                    "analysis_terms_added": len(enhanced_search_terms)
                }
                step.status = StepStatus.COMPLETED
                logger.info(f"Found {len(reasoning_contexts)} reasoning contexts from {len(search_results)} RAG search results")

            else:
                # No similar questions found - use empty context
                step.output_data = {
                    # "context": "",
                    "search_results": [],
                    "total_results": 0,
                    "search_query": search_query,
                    "enhanced_search_used": isinstance(search_query, list),
                    "analysis_terms_added": len(enhanced_search_terms)
                }
                step.status = StepStatus.COMPLETED
                logger.info("No result from RAG search, using original context")

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.output_data = {"error": str(e)}
            logger.error(f"Context step failed: {e}")

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    def _build_reasoning_context(self, reasoning_contexts: List[Dict]) -> str:
        """Build combined reasoning context from Kaggle Q&A pairs."""
        try:
            context_parts = []
            for i, ctx in enumerate(reasoning_contexts[:3], 1):  # Limit to top 3
                score = ctx.get('score', 0)
                question = ctx.get('question', '')
                reasoning = ctx.get('reasoning', '')

                if reasoning and reasoning.strip():
                    context_parts.append(
                        f"Similar Q{i} (score: {score:.3f}): {question}\n"
                        f"Reasoning: {reasoning}\n"
                    )

            if context_parts:
                return "Similar financial questions and reasoning:\n\n" + "\n".join(context_parts)
            else:
                return "No relevant reasoning context found from similar questions."

        except Exception as e:
            logger.error(f"Error building reasoning context: {e}")
            return "Error processing reasoning context from similar questions."

    def _parse_question_and_context(self, question: str) -> tuple[str, Optional[str]]:
        """Parse question to extract context if embedded in the question string."""
        try:
            if "original_question:" in question and "original_context:" in question:
                parts = question.split("original_context:")
                if len(parts) == 2:
                    question_part = parts[0].replace("original_question:", "").strip()
                    context_part = parts[1].strip()
                    return question_part, context_part
            return question, None
        except Exception as e:
            logger.warning(f"Failed to parse question and context: {e}")
            return question, None

    def _synthesis_answer(self, chain: ReasoningChain, original_context) -> ReasoningStep:
        """Generate final answer based on analysis and RAG context."""
        step = ReasoningStep(
            step_type=StepType.SYNTHESIS,
            description="Generate final answer based on analysis and RAG context",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Get results from previous steps
            analysis_step = chain.get_step_by_type(StepType.ANALYSIS)
            context_step = chain.get_step_by_type(StepType.CONTEXT_RAG)

            # Build comprehensive context for answer generation
            question = chain.question

            # Extract analysis information
            analysis_data = analysis_step.output_data if analysis_step else {}

            # Extract RAG context information
            # rag_context = ""
            reasoning_contexts = []
            if context_step and context_step.output_data:
                # rag_context = context_step.output_data.get('context', '')
                reasoning_contexts = context_step.output_data.get('reasoning_contexts', [])
                
            from .prompts import get_synthesis_prompt 
            # Use enhanced analysis prompt
            synthesis_prompt = get_synthesis_prompt(question, analysis_data, original_context, reasoning_contexts)

            response = self._safe_model_generate(
                synthesis_prompt,
                "Unable to generate answer"
            )

            # Extract final answer - look for common patterns
            final_answer = response.strip()

            # Try to extract more specific final answer if structured format is used
            for pattern in ["<final_answer>", "Final answer:", "Answer:", "final answer:", "answer:", "The answer is:", "Result:"]:
                if pattern in response:
                    potential_answer = response.split(pattern)[-1].strip()
                    if potential_answer and len(potential_answer) < len(final_answer):
                        final_answer = potential_answer
                    break

            # Clean up the final answer
            if final_answer.endswith('.'):
                pass  # Keep period
            final_answer = final_answer.strip()

            # Build reasoning transparency showing the simplified process
            transparency = {
                "raw_answer": response,
                "analysis_output": analysis_step.output_data,
                "rag_search": len(reasoning_contexts),
                "reasoning_contexts": reasoning_contexts,
                "context_source": context_step.output_data.get('context_source', 'none') if context_step else 'none'
            }

            # Set final answer on chain
            chain.final_answer = final_answer

            step.output_data = {
                "final_answer": final_answer,
                "full_response": response,
                "reasoning_transparency": transparency,
                "synthesis_method": "analysis_plus_rag"
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            # Ensure we have some answer even on error
            chain.final_answer = "Unable to generate answer due to processing error"

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    def _safe_model_generate(self, prompt: str, fallback_message: str = None, timeout: int = 60) -> str:
        """Safely generate LLM response with fallback and timeout."""
        try:
            # Use the synchronous wrapper method
            response = self.model_manager.generate_sync(prompt)
            if hasattr(response, 'content'):
                return response.content or fallback_message or "No response generated"
            return str(response) or fallback_message or "No response generated"
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return fallback_message or "Unable to generate response"
