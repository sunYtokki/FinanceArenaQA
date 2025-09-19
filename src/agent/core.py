"""Simplified 5-step reasoning chain for financial question answering."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime

# Import structured data types
from ..rag.data_types import RAGResult, AdjustmentSpec, RAGEvidence

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a reasoning step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(Enum):
    """Type of reasoning step - simplified to 5 core types."""
    ANALYSIS = "analysis"           # Step 1: Question Analysis
    CONTEXT_RAG = "context_rag"     # Step 2: Context RAG Retrieval
    CALCULATION = "calculation"     # Step 3: Calculate
    VALIDATION = "validation"       # Step 4: Validate
    SYNTHESIS = "synthesis"         # Step 5: Answer


@dataclass
class ReasoningStep:
    """Data structure for tracking individual reasoning steps."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: StepType = StepType.ANALYSIS
    description: str = ""
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    tool_used: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    execution_time_ms: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "id": self.id,
            "step_type": self.step_type.value,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "tool_used": self.tool_used,
            "status": self.status.value,
            "error_message": self.error_message,
            "confidence_score": self.confidence_score,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ReasoningChain:
    """Container for a complete 5-step reasoning chain."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence_score: Optional[float] = None
    total_execution_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the chain."""
        self.steps.append(step)

    def get_step_by_type(self, step_type: StepType) -> Optional[ReasoningStep]:
        """Get the step of a specific type (only one per chain)."""
        for step in self.steps:
            if step.step_type == step_type:
                return step
        return None

    def is_complete(self) -> bool:
        """Check if the reasoning chain is complete."""
        if len(self.steps) != 5:
            return False
        return all(step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
                  for step in self.steps)

    def has_failures(self) -> bool:
        """Check if the reasoning chain has any failures."""
        return any(step.status == StepStatus.FAILED for step in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary for serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "confidence_score": self.confidence_score,
            "total_execution_time_ms": self.total_execution_time_ms,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class Tool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for selection."""
        pass

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with input data."""
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate tool input data."""
        return True


class SimplifiedAgent:
    """Agent with simplified 5-step linear reasoning chain."""

    def __init__(self, model_manager, tools: Optional[List[Tool]] = None, context_retriever=None):
        """Initialize agent with model manager and tools.

        Args:
            model_manager: ModelManager instance for LLM access
            tools: List of available tools for the agent
            context_retriever: Optional ContextRetriever for RAG functionality
        """
        self.model_manager = model_manager
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.context_retriever = context_retriever
        self.reasoning_chains: Dict[str, ReasoningChain] = {}

    async def answer_question(self, question: str, context: str = None, **kwargs) -> ReasoningChain:
        """Answer a financial question using simplified 5-step reasoning.

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

            # Step 2: Context RAG - Retrieve relevant context and patterns
            context_step = await self._execute_step_2_context_rag(chain)
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

    async def _execute_step_2_context_rag(self, chain: ReasoningChain) -> ReasoningStep:
        """Step 2: Retrieve relevant context using RAG."""
        step = ReasoningStep(
            step_type=StepType.CONTEXT_RAG,
            description="Retrieve relevant context and patterns",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": chain.question}
        )

        start_time = datetime.now()

        try:
            # Get context retriever if available
            if self.context_retriever:
                rag_result = await self.context_retriever.retrieve_context(
                    question=chain.question,
                    context=chain.metadata.get("context", "")
                )
                step.output_data = {
                    "rag_result": rag_result.to_dict(),
                    "method": "rag_retrieval",
                    "use_rag": rag_result.evidence_score >= 0.6,  # ConfidenceThresholds.RAG_MIN_CONFIDENCE
                    "fallback_reason": rag_result.fallback_reason
                }
                step.tool_used = "context_retriever"
            else:
                # Fallback: return empty RAG result
                rag_result = RAGResult(
                    signals=[],
                    evidence=[],
                    proposed_adjustments=[],
                    evidence_score=0.0,
                    retrieval_method="fallback",
                    fallback_reason="No context retriever available"
                )
                step.output_data = {
                    "rag_result": rag_result.to_dict(),
                    "method": "fallback",
                    "use_rag": False,
                    "fallback_reason": "No context retriever configured"
                }

            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _execute_step_3_calculate(self, chain: ReasoningChain) -> ReasoningStep:
        """Step 3: Execute financial calculations with RAG-based adjustments."""
        step = ReasoningStep(
            step_type=StepType.CALCULATION,
            description="Execute financial calculations",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Get RAG results from step 2
            context_step = chain.get_step_by_type(StepType.CONTEXT_RAG)
            use_rag = False
            rag_adjustments = []

            if context_step and context_step.output_data:
                use_rag = context_step.output_data.get("use_rag", False)
                rag_result_dict = context_step.output_data.get("rag_result", {})
                rag_adjustments = rag_result_dict.get("proposed_adjustments", [])

            # Prepare calculation input with RAG context
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
                # Enhanced LLM-based calculation with RAG context
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
                    "method": "llm_with_rag" if use_rag else "llm_fallback",
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
        """Build calculation prompt with RAG context.

        Args:
            question: Financial question
            context: Context text
            use_rag: Whether to use RAG adjustments
            rag_adjustments: List of RAG adjustment specifications

        Returns:
            Formatted prompt for calculation
        """
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

        RAG-Detected Adjustments to Apply:
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
        """Build reasoning transparency showing decision process.

        Args:
            chain: Reasoning chain

        Returns:
            Dictionary with transparency information
        """
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

        # Get RAG step
        context_step = chain.get_step_by_type(StepType.CONTEXT_RAG)
        if context_step and context_step.output_data:
            rag_result = context_step.output_data.get("rag_result", {})
            transparency["rag_used"] = context_step.output_data.get("use_rag", False)
            transparency["fallback_reason"] = context_step.output_data.get("fallback_reason")
            transparency["confidence_score"] = rag_result.get("evidence_score", 0.0)

            # Extract evidence sources
            evidence = rag_result.get("evidence", [])
            transparency["evidence_sources"] = [
                {
                    "section": e.get("section", "unknown"),
                    "confidence": e.get("confidence", 0.0),
                    "text_preview": e.get("text", "")[:100] + "..." if len(e.get("text", "")) > 100 else e.get("text", "")
                }
                for e in evidence[:3]  # Top 3 sources
            ]

            # Extract applied adjustments
            adjustments = rag_result.get("proposed_adjustments", [])
            transparency["adjustments_applied"] = [
                {
                    "type": adj.get("type", "unknown"),
                    "description": adj.get("description", ""),
                    "confidence": adj.get("confidence", 0.0)
                }
                for adj in adjustments
            ]

        return transparency

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

    def get_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Get a reasoning chain by ID."""
        return self.reasoning_chains.get(chain_id)

    def list_reasoning_chains(self) -> List[str]:
        """Get list of all reasoning chain IDs."""
        return list(self.reasoning_chains.keys())

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


# Backward compatibility alias
Agent = SimplifiedAgent