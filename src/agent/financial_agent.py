"""Unified Financial Agent with tool-based RAG architecture."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .core import Tool, ReasoningChain, ReasoningStep, StepType, StepStatus
from ..tools.financial_calculator import FinancialCalculator
from ..tools.context_retriever_tool import ContextRetrieverTool

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

        # Validate configuration
        self._validate_config()

        # Initialize tools
        self._initialize_tools()

        logger.info(f"FinancialAgent initialized with {len(self.tools)} tools, RAG enabled: {enable_rag}")

    def _validate_config(self) -> None:
        """Validate the configuration and apply defaults."""
        if not self.config:
            self.config = {}

        # Validate and set RAG configuration defaults
        if self.enable_rag:
            rag_config = self.config.setdefault("rag", {})

            # Set sensible defaults
            rag_config.setdefault("enabled", True)
            rag_config.setdefault("chroma_persist_dir", "./chroma_db")
            rag_config.setdefault("collection_name", "financial_patterns")
            rag_config.setdefault("config_path", "config/context_patterns.yaml")

            # Validate confidence thresholds
            thresholds = rag_config.setdefault("confidence_thresholds", {})
            thresholds.setdefault("rag_min_confidence", 0.6)
            thresholds.setdefault("high_confidence", 0.8)

            # Validate threshold values
            if not (0.0 <= thresholds["rag_min_confidence"] <= 1.0):
                logger.warning(f"Invalid rag_min_confidence: {thresholds['rag_min_confidence']}, using 0.6")
                thresholds["rag_min_confidence"] = 0.6

            if not (0.0 <= thresholds["high_confidence"] <= 1.0):
                logger.warning(f"Invalid high_confidence: {thresholds['high_confidence']}, using 0.8")
                thresholds["high_confidence"] = 0.8

            # Validate retrieval settings
            retrieval = rag_config.setdefault("retrieval_settings", {})
            retrieval.setdefault("max_results", 10)
            retrieval.setdefault("similarity_threshold", 0.7)

            # Validate max_results
            if not isinstance(retrieval["max_results"], int) or retrieval["max_results"] <= 0:
                logger.warning(f"Invalid max_results: {retrieval['max_results']}, using 10")
                retrieval["max_results"] = 10

            # Validate fallback behavior
            fallback = rag_config.setdefault("fallback_behavior", {})
            fallback.setdefault("continue_without_rag", True)
            fallback.setdefault("log_failures", True)

            logger.info("RAG configuration validated and defaults applied")

    def answer_question(self, question: str, **kwargs) -> ReasoningChain:
        """Synchronous wrapper for answer_question to maintain protocol compatibility.

        Args:
            question: The financial question to answer
            **kwargs: Additional parameters including context

        Returns:
            ReasoningChain containing the complete reasoning process
        """
        import asyncio

        # Handle the async call in a sync wrapper
        try:
            # Get or create event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (e.g., in Jupyter), use run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.answer_question_async(question, **kwargs))
                    return future.result()
            else:
                # If no loop is running, use asyncio.run
                return asyncio.run(self.answer_question_async(question, **kwargs))
        except Exception as e:
            logger.error(f"Error in sync answer_question wrapper: {e}")
            # Return a basic error chain for compatibility
            chain = ReasoningChain(question=question)
            from .core import ReasoningStep, StepType, StepStatus
            error_step = ReasoningStep(
                step_type=StepType.SYNTHESIS,
                status=StepStatus.ERROR,
                output=f"Error: {str(e)}",
                metadata={"error": str(e)}
            )
            chain.add_step(error_step)
            chain.final_answer = f"Error processing question: {str(e)}"
            return chain

    def _initialize_tools(self) -> None:
        """Initialize and register available tools."""
        try:
            # Always add financial calculator
            financial_calc = FinancialCalculator()
            self.tools[financial_calc.name] = financial_calc
            logger.info(f"Registered tool: {financial_calc.name}")

            # Add RAG tool if enabled
            if self.enable_rag:
                rag_config = self.config.get("rag", {})
                context_tool = ContextRetrieverTool(
                    chroma_persist_dir=rag_config.get("chroma_persist_dir", "./chroma_db"),
                    collection_name=rag_config.get("collection_name", "financial_patterns")
                )

                # Check if RAG tool is available
                if context_tool.is_available():
                    self.tools[context_tool.name] = context_tool
                    logger.info(f"Registered tool: {context_tool.name}")
                else:
                    logger.warning("RAG tool not available, continuing without context retrieval")

        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            # Continue with available tools

    async def answer_question_async(self, question: str, context: str = None, **kwargs) -> ReasoningChain:
        """Answer a financial question using 5-step reasoning with tool integration.

        Args:
            question: The financial question to answer
            context: Optional context for the question
            **kwargs: Additional parameters for reasoning

        Returns:
            ReasoningChain containing the complete reasoning process
        """
        # Create new reasoning chain
        chain = ReasoningChain(question=question)
        if context:
            chain.metadata["context"] = context
        self.reasoning_chains[chain.id] = chain

        start_time = datetime.now()

        try:
            # Step 1: Analysis - Question analysis and requirements identification
            analysis_step = await self._execute_step_1_analysis(question, chain)
            chain.add_step(analysis_step)

            # Step 2: Context RAG - Tool-based context retrieval
            context_step = await self._execute_step_2_context_tools(question, context, chain)
            chain.add_step(context_step)

            # Step 3: Calculate - Execute financial calculations
            calculation_step = await self._execute_step_3_calculate(chain)
            chain.add_step(calculation_step)

            # Step 4: Validate - Validate results and reasoning
            validation_step = await self._execute_step_4_validate(chain)
            chain.add_step(validation_step)

            # Step 5: Answer - Synthesize final answer
            synthesis_step = await self._execute_step_5_answer(chain)
            chain.add_step(synthesis_step)

        except Exception as e:
            logger.error(f"Critical error in reasoning chain {chain.id}: {e}")
            # Add error step and ensure we have a final answer
            error_step = ReasoningStep(
                step_type=StepType.VALIDATION,
                description="Critical reasoning chain error",
                status=StepStatus.FAILED,
                error_message=str(e)
            )
            chain.add_step(error_step)

        finally:
            # Ensure we have a final answer
            self._ensure_final_answer(chain)

            # Calculate total execution time
            end_time = datetime.now()
            chain.completed_at = end_time
            chain.total_execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Log completion
            logger.info(f"Reasoning chain {chain.id} completed in {chain.total_execution_time_ms}ms")

            return chain

    async def _execute_step_1_analysis(self, question: str, chain: ReasoningChain) -> ReasoningStep:
        """Step 1: Analyze question type and requirements."""
        step = ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="Analyze question and identify requirements",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": question}
        )

        start_time = datetime.now()

        try:
            # Simple question analysis
            analysis_prompt = f"""
            Analyze this financial question briefly:
            1. Question type (calculation, conceptual, or assumption-based)
            2. Key financial concepts involved
            3. Required data or context needed

            Question: {question}

            Provide a concise analysis.
            """

            response_content = await self._safe_model_generate(
                analysis_prompt,
                "Question analysis failed"
            )

            step.output_data = {
                "analysis": response_content,
                "question_type": self._classify_question(question),
                "requires_context": "adjusted" in question.lower() or "assumption" in question.lower()
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _execute_step_2_context_tools(
        self,
        question: str,
        context: str,
        chain: ReasoningChain
    ) -> ReasoningStep:
        """Step 2: Execute context tools for information retrieval."""
        step = ReasoningStep(
            step_type=StepType.CONTEXT_RAG,
            description="Execute context tools for information retrieval",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": question, "context": context or ""}
        )

        start_time = datetime.now()

        try:
            # Get available context tools
            context_tools = [tool for tool in self.tools.values()
                           if tool.name == "context_retriever"]

            if context_tools:
                # Execute context tools in parallel
                tool_inputs = [
                    {"question": question, "context": context or ""}
                    for _ in context_tools
                ]

                tool_results = await asyncio.gather(
                    *[tool.execute(input_data) for tool, input_data
                      in zip(context_tools, tool_inputs)],
                    return_exceptions=True
                )

                # Process results
                successful_results = []
                for i, result in enumerate(tool_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Context tool {context_tools[i].name} failed: {result}")
                    else:
                        successful_results.append(result)

                # Use the best result (highest evidence score)
                if successful_results:
                    best_result = max(successful_results,
                                    key=lambda r: r.get("evidence_score", 0.0))

                    step.output_data = best_result
                    step.tool_used = "context_retriever"
                else:
                    # All tools failed
                    step.output_data = self._create_empty_context_result(
                        "All context tools failed"
                    )
            else:
                # No context tools available
                step.output_data = self._create_empty_context_result(
                    "No context tools available"
                )

            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.output_data = self._create_empty_context_result(f"Tool execution error: {e}")

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _execute_step_3_calculate(self, chain: ReasoningChain) -> ReasoningStep:
        """Step 3: Execute financial calculations with context from tools."""
        step = ReasoningStep(
            step_type=StepType.CALCULATION,
            description="Execute financial calculations",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Get context results from step 2
            context_step = chain.get_step_by_type(StepType.CONTEXT_RAG)
            use_rag = False
            rag_adjustments = []

            if context_step and context_step.output_data:
                use_rag = context_step.output_data.get("use_rag", False)
                rag_adjustments = context_step.output_data.get("proposed_adjustments", [])

            # Prepare calculation input
            calc_input = {
                "question": chain.question,
                "context": chain.metadata.get("context", ""),
                "analysis": chain.get_step_by_type(StepType.ANALYSIS).output_data,
                "use_rag": use_rag,
                "rag_adjustments": rag_adjustments,
                "fallback_to_assumptions": not use_rag
            }

            # Use financial calculator tool if available
            if "financial_calculator" in self.tools:
                result = await self.tools["financial_calculator"].execute(calc_input)
                step.output_data = result
                step.tool_used = "financial_calculator"
            else:
                # Enhanced LLM-based calculation with context
                calc_prompt = self._build_calculation_prompt(
                    chain.question,
                    chain.metadata.get("context", ""),
                    use_rag,
                    rag_adjustments
                )

                response = await self._safe_model_generate(
                    calc_prompt,
                    "Calculation failed"
                )

                step.output_data = {
                    "calculation_result": response,
                    "method": "llm_with_context" if use_rag else "llm_fallback",
                    "used_rag": use_rag,
                    "applied_adjustments": len(rag_adjustments)
                }

            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _execute_step_4_validate(self, chain: ReasoningChain) -> ReasoningStep:
        """Step 4: Validate calculation results."""
        step = ReasoningStep(
            step_type=StepType.VALIDATION,
            description="Validate calculation results",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            calculation_step = chain.get_step_by_type(StepType.CALCULATION)

            if calculation_step and calculation_step.status == StepStatus.COMPLETED:
                # Simple validation - check if we have a result
                has_result = (
                    calculation_step.output_data and
                    calculation_step.output_data.get("calculation_result")
                )

                step.output_data = {
                    "validation_passed": has_result,
                    "has_calculation": bool(calculation_step.output_data),
                    "validation_notes": "Basic result presence check"
                }
            else:
                step.output_data = {
                    "validation_passed": False,
                    "has_calculation": False,
                    "validation_notes": "No calculation step completed"
                }

            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _execute_step_5_answer(self, chain: ReasoningChain) -> ReasoningStep:
        """Step 5: Synthesize final answer."""
        step = ReasoningStep(
            step_type=StepType.SYNTHESIS,
            description="Synthesize final answer",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Get results from previous steps
            analysis_step = chain.get_step_by_type(StepType.ANALYSIS)
            calculation_step = chain.get_step_by_type(StepType.CALCULATION)
            validation_step = chain.get_step_by_type(StepType.VALIDATION)

            # Build context for synthesis
            context = f"""
            Question: {chain.question}

            Analysis: {analysis_step.output_data.get('analysis', '') if analysis_step else ''}

            Calculation: {calculation_step.output_data.get('calculation_result', '') if calculation_step else ''}

            Validation: {validation_step.output_data.get('validation_notes', '') if validation_step else ''}
            """

            synthesis_prompt = f"""
            Based on the analysis and calculations, provide a clear, concise final answer.

            {context}

            Instructions:
            1. Provide the final answer clearly
            2. Keep it concise and direct
            3. End with the exact final answer on a new line starting with '<final_answer>:'

            Response:
            """

            response = await self._safe_model_generate(
                synthesis_prompt,
                "Unable to synthesize answer"
            )

            # Extract final answer if present
            if "<final_answer>:" in response:
                final_answer = response.split("<final_answer>:")[-1].strip()
            else:
                final_answer = response.strip()

            # Add reasoning transparency
            transparency = self._build_reasoning_transparency(chain)

            # Set final answer on chain
            chain.final_answer = final_answer

            step.output_data = {
                "final_answer": final_answer,
                "full_response": response,
                "reasoning_transparency": transparency
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    def _classify_question(self, question: str) -> str:
        """Simple question classification."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["adjusted", "assume", "estimate", "typical"]):
            return "assumption"
        elif any(word in question_lower for word in ["calculate", "compute", "what is", "find"]):
            return "calculation"
        elif any(word in question_lower for word in ["explain", "why", "describe", "difference"]):
            return "conceptual"
        else:
            return "calculation"  # default

    def _build_calculation_prompt(
        self,
        question: str,
        context: str,
        use_rag: bool,
        rag_adjustments: List[Dict[str, Any]]
    ) -> str:
        """Build calculation prompt with context information."""
        base_prompt = f"""
        Answer this financial question with calculations:

        Question: {question}
        Context: {context or "Not provided"}
        """

        if use_rag and rag_adjustments:
            adjustment_text = "\n".join([
                f"- {adj.get('description', adj.get('type', 'Unknown adjustment'))}"
                for adj in rag_adjustments
            ])

            base_prompt += f"""

        Context-Detected Adjustments to Apply:
        {adjustment_text}

        Instructions: Use the detected adjustments above when calculating adjusted metrics.
        """
        else:
            base_prompt += """

        Instructions: If this requires assumptions about adjustments (like for adjusted EBITDA),
        use standard assumptions (e.g., add back stock-based compensation for adjusted metrics).
        """

        base_prompt += """

        Show your work and provide the final answer clearly.
        """

        return base_prompt

    def _build_reasoning_transparency(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Build reasoning transparency showing decision process."""
        transparency = {
            "question_type": "unknown",
            "rag_used": False,
            "adjustments_applied": [],
            "evidence_sources": [],
            "fallback_reason": None,
            "confidence_score": 0.0
        }

        # Get analysis step
        analysis_step = chain.get_step_by_type(StepType.ANALYSIS)
        if analysis_step and analysis_step.output_data:
            transparency["question_type"] = analysis_step.output_data.get("question_type", "unknown")

        # Get context step
        context_step = chain.get_step_by_type(StepType.CONTEXT_RAG)
        if context_step and context_step.output_data:
            transparency["rag_used"] = context_step.output_data.get("use_rag", False)
            transparency["fallback_reason"] = context_step.output_data.get("fallback_reason")
            transparency["confidence_score"] = context_step.output_data.get("evidence_score", 0.0)

            # Extract evidence sources
            evidence = context_step.output_data.get("evidence", [])
            transparency["evidence_sources"] = [
                {
                    "section": e.get("section", "unknown"),
                    "confidence": e.get("confidence", 0.0),
                    "text_preview": e.get("text", "")[:100] + "..." if len(e.get("text", "")) > 100 else e.get("text", "")
                }
                for e in evidence[:3]  # Top 3 sources
            ]

            # Extract applied adjustments
            adjustments = context_step.output_data.get("proposed_adjustments", [])
            transparency["adjustments_applied"] = [
                {
                    "type": adj.get("type", "unknown"),
                    "description": adj.get("description", ""),
                    "confidence": adj.get("confidence", 0.0)
                }
                for adj in adjustments
            ]

        return transparency

    def _create_empty_context_result(self, reason: str) -> Dict[str, Any]:
        """Create empty context result for fallback scenarios."""
        return {
            "signals": [],
            "evidence": [],
            "proposed_adjustments": [],
            "evidence_score": 0.0,
            "retrieval_method": "fallback",
            "fallback_reason": reason,
            "use_rag": False
        }

    async def _safe_model_generate(self, prompt: str, fallback_message: str = None, timeout: int = 60) -> str:
        """Safely generate LLM response with fallback and timeout."""
        try:
            response = await asyncio.wait_for(
                self.model_manager.generate(prompt),
                timeout=timeout
            )
            if hasattr(response, 'content'):
                return response.content or fallback_message or "No response generated"
            return str(response) or fallback_message or "No response generated"
        except asyncio.TimeoutError:
            logger.error(f"LLM generation timed out after {timeout}s")
            return fallback_message or "Request timed out"
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return fallback_message or "Unable to generate response"

    def _ensure_final_answer(self, chain: ReasoningChain) -> None:
        """Ensure chain always has a final answer."""
        if not chain.final_answer or chain.final_answer.strip() == "":
            # Try to extract answer from synthesis step
            synthesis_step = chain.get_step_by_type(StepType.SYNTHESIS)
            if synthesis_step and synthesis_step.output_data:
                chain.final_answer = synthesis_step.output_data.get("final_answer", "")

            # Final fallback
            if not chain.final_answer:
                chain.final_answer = "Unable to provide answer due to processing issues"

    # Tool management methods
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolbox."""
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent's toolbox."""
        if tool_name in self.tools:
            del self.tools[tool_name]

    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about available tools."""
        stats = {}
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'get_stats'):
                stats[tool_name] = tool.get_stats()
            else:
                stats[tool_name] = {"available": True, "type": type(tool).__name__}
        return stats

    # Agent management methods
    def get_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Get a reasoning chain by ID."""
        return self.reasoning_chains.get(chain_id)

    def list_reasoning_chains(self) -> List[str]:
        """Get list of all reasoning chain IDs."""
        return list(self.reasoning_chains.keys())