"""Core agent implementation with multi-step reasoning chain."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a reasoning step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(Enum):
    """Type of reasoning step."""
    ANALYSIS = "analysis"
    CALCULATION = "calculation"
    TOOL_USE = "tool_use"
    CLASSIFICATION = "classification"
    ASSUMPTION = "assumption"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


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
    parent_step_id: Optional[str] = None
    child_step_ids: List[str] = field(default_factory=list)

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
            "timestamp": self.timestamp.isoformat(),
            "parent_step_id": self.parent_step_id,
            "child_step_ids": self.child_step_ids
        }


@dataclass
class ReasoningChain:
    """Container for a complete reasoning chain."""

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

    def get_step_by_id(self, step_id: str) -> Optional[ReasoningStep]:
        """Get a specific step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_steps_by_type(self, step_type: StepType) -> List[ReasoningStep]:
        """Get all steps of a specific type."""
        return [step for step in self.steps if step.step_type == step_type]

    def get_failed_steps(self) -> List[ReasoningStep]:
        """Get all failed steps."""
        return [step for step in self.steps if step.status == StepStatus.FAILED]

    def is_complete(self) -> bool:
        """Check if the reasoning chain is complete."""
        if not self.steps:
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


class Agent:
    """Base agent class with reasoning chain interface."""

    def __init__(self, model_manager, tools: Optional[List[Tool]] = None, max_steps: int = 15):
        """Initialize agent with model manager and tools.

        Args:
            model_manager: ModelManager instance for LLM access
            tools: List of available tools for the agent
            max_steps: Maximum number of reasoning steps to prevent infinite loops
        """
        self.model_manager = model_manager
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.max_steps = max_steps

    async def answer_question(self, question: str, **kwargs) -> ReasoningChain:
        """Answer a financial question using multi-step reasoning.

        Args:
            question: The financial question to answer
            **kwargs: Additional parameters for reasoning

        Returns:
            ReasoningChain containing the complete reasoning process
        """
        # Create new reasoning chain
        chain = ReasoningChain(question=question)
        self.reasoning_chains[chain.id] = chain

        start_time = datetime.now()

        try:
            # Step 1: Analyze and decompose the question
            # analysis_step = await self._safe_execute_step(self._analyze_question, question, chain)
            # chain.add_step(analysis_step)

            # Check max steps limit
            # if len(chain.steps) >= self.max_steps:
            #     logger.warning(f"Reached max steps limit ({self.max_steps}), stopping early")
            #     return chain

            # Step 2: Generate reasoning plan
            # plan_step = await self._safe_execute_step(self._generate_reasoning_plan, chain)
            # chain.add_step(plan_step)

            # Check max steps limit
            # if len(chain.steps) >= self.max_steps:
            #     logger.warning(f"Reached max steps limit ({self.max_steps}), stopping early")
            #     return chain

            # Step 3: Execute reasoning steps
            # await self._execute_reasoning_plan(chain)

            # Check max steps limit
            # if len(chain.steps) >= self.max_steps:
            #     logger.warning(f"Reached max steps limit ({self.max_steps}), stopping early")
            #     return chain

            # Step 4: Synthesize final answer
            synthesis_step = await self._safe_execute_step(self._synthesize_answer, chain)
            chain.add_step(synthesis_step)

        except Exception as e:
            logger.error(f"Critical error in reasoning chain {chain.id}: {e}")

            # Create error step
            error_step = ReasoningStep(
                step_type=StepType.VALIDATION,
                description="Critical reasoning chain error",
                status=StepStatus.FAILED,
                error_message=str(e)
            )
            chain.add_step(error_step)

        finally:
            # Always ensure we have a final answer
            self._ensure_final_answer(chain)

            # Calculate total execution time
            end_time = datetime.now()
            chain.completed_at = end_time
            chain.total_execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Log reasoning chain completion
            logger.info(f"Reasoning chain {chain.id} completed with {len(chain.steps)} steps")

            return chain

    async def _analyze_question(self, question: str, chain: ReasoningChain) -> ReasoningStep:
        """Analyze the question to understand its type and requirements."""
        step = ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="Analyze question type and requirements",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": question}
        )

        start_time = datetime.now()

        try:
            # Use model to analyze the question
            analysis_prompt = f"""
            Analyze this financial question and identify:
            1. Question type (calculation, analysis, conceptual)
            2. Required information and data
            3. Potential tools or methods needed
            4. Expected answer format

            Question: {question}

            Provide a structured analysis.
            """

            response_content = await self._safe_model_generate(
                analysis_prompt,
                "Basic question analysis failed"
            )

            step.output_data = {
                "analysis": response_content,
                "question_type": "to_be_classified",  # Will be refined by classifier
                "raw_model_response": response_content
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _generate_reasoning_plan(self, chain: ReasoningChain) -> ReasoningStep:
        """Generate a step-by-step reasoning plan."""
        step = ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="Generate reasoning plan",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": chain.question, "analysis": chain.steps[-1].output_data}
        )

        start_time = datetime.now()

        try:
            # Create reasoning plan based on question analysis
            plan_prompt = f"""
            Based on the question analysis, create a step-by-step reasoning plan.

            Question: {chain.question}
            Analysis: {chain.steps[-1].output_data.get('analysis', '')}

            Available tools: {list(self.tools.keys())}

            Create a numbered list of reasoning steps needed to answer this question.
            """

            response = await self.model_manager.generate(plan_prompt)

            step.output_data = {
                "reasoning_plan": response.content,
                "available_tools": list(self.tools.keys()),
                "raw_model_response": response.to_dict() if hasattr(response, 'to_dict') else str(response)
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _execute_reasoning_plan(self, chain: ReasoningChain) -> None:
        """Execute the reasoning plan step by step with tool orchestration."""
        # Check max steps before proceeding
        if len(chain.steps) >= self.max_steps:
            logger.warning(f"Max steps reached, skipping reasoning plan execution")
            return

        # First, classify the question to determine approach
        await self._classify_and_route(chain)

        # Check max steps after classification
        if len(chain.steps) >= self.max_steps:
            logger.warning(f"Max steps reached after classification, stopping")
            return

        # Then execute the appropriate reasoning approach
        question_classification = self._get_question_classification(chain)

        if question_classification == "tactical_basic":
            await self._execute_tactical_basic_reasoning(chain)
        elif question_classification == "tactical_assumption":
            await self._execute_tactical_assumption_reasoning(chain)
        else:  # conceptual
            await self._execute_conceptual_reasoning(chain)

    async def _classify_and_route(self, chain: ReasoningChain) -> None:
        """Classify question and add classification step."""
        classification_step = ReasoningStep(
            step_type=StepType.CLASSIFICATION,
            description="Classify question type for routing",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": chain.question}
        )

        start_time = datetime.now()

        try:
            # Use question classifier if available
            if hasattr(self, 'question_classifier'):
                result = self.question_classifier.classify(chain.question)
                classification_step.output_data = {
                    "question_type": result.question_type.value,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "detected_indicators": result.detected_indicators,
                    "missing_data_indicators": result.missing_data_indicators,
                    "assumptions_needed": result.assumptions_needed
                }
            else:
                # Fallback: simple keyword-based classification
                classification_result = await self._simple_classification(chain.question)
                classification_step.output_data = classification_result

            classification_step.status = StepStatus.COMPLETED

        except Exception as e:
            classification_step.status = StepStatus.FAILED
            classification_step.error_message = str(e)
            # Default to conceptual on failure
            classification_step.output_data = {
                "question_type": "conceptual",
                "confidence": 0.5,
                "reasoning": f"Classification failed: {str(e)}, defaulting to conceptual"
            }

        finally:
            end_time = datetime.now()
            classification_step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        chain.add_step(classification_step)

    async def _simple_classification(self, question: str) -> Dict[str, Any]:
        """Simple keyword-based classification fallback."""
        question_lower = question.lower()

        # Simple heuristics
        calculation_keywords = ['calculate', 'compute', 'find', 'what is', 'how much']
        assumption_keywords = ['assume', 'estimate', 'typical', 'missing', 'not provided']
        conceptual_keywords = ['explain', 'why', 'what are', 'describe', 'difference']

        calc_score = sum(1 for kw in calculation_keywords if kw in question_lower)
        assumption_score = sum(1 for kw in assumption_keywords if kw in question_lower)
        concept_score = sum(1 for kw in conceptual_keywords if kw in question_lower)

        if assumption_score > 0:
            return {"question_type": "tactical_assumption", "confidence": 0.7}
        elif calc_score > concept_score:
            return {"question_type": "tactical_basic", "confidence": 0.6}
        else:
            return {"question_type": "conceptual", "confidence": 0.6}

    def _get_question_classification(self, chain: ReasoningChain) -> str:
        """Extract question classification from chain."""
        for step in reversed(chain.steps):
            if step.step_type == StepType.CLASSIFICATION and step.output_data:
                return step.output_data.get("question_type", "conceptual")
        return "conceptual"

    async def _execute_tactical_basic_reasoning(self, chain: ReasoningChain) -> None:
        """Execute reasoning for tactical-basic questions (direct calculations)."""
        # Step 1: Extract and validate data
        data_extraction_step = await self._extract_financial_data(chain)
        chain.add_step(data_extraction_step)

        # Step 2: Select appropriate tools
        tool_selection_step = await self._select_tools_for_calculation(chain)
        chain.add_step(tool_selection_step)

        # Step 3: Execute calculations
        calculation_step = await self._execute_financial_calculations(chain)
        chain.add_step(calculation_step)

        # Step 4: Validate results
        validation_step = await self._validate_calculation_results(chain)
        chain.add_step(validation_step)

    async def _execute_tactical_assumption_reasoning(self, chain: ReasoningChain) -> None:
        """Execute reasoning for tactical-assumption questions (calculations with assumptions)."""
        # Step 1: Identify missing data
        missing_data_step = await self._identify_missing_data(chain)
        chain.add_step(missing_data_step)

        # Step 2: Generate assumptions
        assumption_step = await self._generate_assumptions(chain)
        chain.add_step(assumption_step)

        # Step 3: Execute calculations with assumptions
        calculation_step = await self._execute_financial_calculations(chain)
        chain.add_step(calculation_step)

        # Step 4: Perform sensitivity analysis
        sensitivity_step = await self._perform_sensitivity_analysis(chain)
        chain.add_step(sensitivity_step)

    async def _execute_conceptual_reasoning(self, chain: ReasoningChain) -> None:
        """Execute reasoning for conceptual questions (explanations)."""
        # Step 1: Retrieve relevant knowledge
        knowledge_step = await self._retrieve_financial_knowledge(chain)
        chain.add_step(knowledge_step)

        # Step 2: Structure explanation
        structure_step = await self._structure_explanation(chain)
        chain.add_step(structure_step)

        # Step 3: Add examples if appropriate
        examples_step = await self._add_relevant_examples(chain)
        chain.add_step(examples_step)

    async def _extract_financial_data(self, chain: ReasoningChain) -> ReasoningStep:
        """Extract financial data from the question."""
        step = ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="Extract financial data from question",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": chain.question}
        )

        start_time = datetime.now()

        try:
            # Use document parser tool if available
            if "document_parser" in self.tools:
                result = await self.tools["document_parser"].execute({
                    "text": chain.question,
                    "extract_type": "financial_data"
                })
                step.output_data = result
                step.tool_used = "document_parser"
            else:
                # Fallback: regex-based extraction
                import re
                numbers = re.findall(r'\$?[\d,]+\.?\d*', chain.question)
                step.output_data = {
                    "extracted_numbers": numbers,
                    "method": "regex_fallback"
                }

            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _select_tools_for_calculation(self, chain: ReasoningChain) -> ReasoningStep:
        """Select appropriate tools based on the question requirements."""
        step = ReasoningStep(
            step_type=StepType.TOOL_USE,
            description="Select appropriate calculation tools",
            status=StepStatus.IN_PROGRESS,
            input_data={"available_tools": list(self.tools.keys())}
        )

        start_time = datetime.now()

        try:
            question_lower = chain.question.lower()
            selected_tools = []

            # Tool selection logic based on question keywords
            tool_keywords = {
                "financial_calculator": ["npv", "irr", "roi", "return", "calculate", "ratio"],
                "code_executor": ["complex", "multiple", "steps", "analysis"],
                "document_parser": ["statement", "document", "extract", "parse"]
            }

            for tool_name, keywords in tool_keywords.items():
                if tool_name in self.tools and any(kw in question_lower for kw in keywords):
                    selected_tools.append(tool_name)

            # Default to financial calculator if no specific tool selected
            if not selected_tools and "financial_calculator" in self.tools:
                selected_tools = ["financial_calculator"]

            step.output_data = {
                "selected_tools": selected_tools,
                "selection_reasoning": f"Selected based on keywords in question"
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _execute_financial_calculations(self, chain: ReasoningChain) -> ReasoningStep:
        """Execute financial calculations using selected tools."""
        step = ReasoningStep(
            step_type=StepType.CALCULATION,
            description="Execute financial calculations",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Get selected tools from previous step
            selected_tools = []
            for prev_step in reversed(chain.steps):
                if prev_step.step_type == StepType.TOOL_USE and prev_step.output_data:
                    selected_tools = prev_step.output_data.get("selected_tools", [])
                    break

            calculation_results = []

            # Execute calculations with each selected tool
            for tool_name in selected_tools:
                if tool_name in self.tools:
                    # Prepare input data for tool
                    tool_input = await self._prepare_tool_input(chain, tool_name)

                    # Execute tool safely with fallback
                    result = await self._safe_tool_execute(tool_name, tool_input, chain.question)

                    calculation_results.append(result)

            step.output_data = {
                "calculation_results": calculation_results,
                "tools_used": selected_tools
            }
            step.tool_used = ", ".join(selected_tools)
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _prepare_tool_input(self, chain: ReasoningChain, tool_name: str) -> Dict[str, Any]:
        """Prepare input data for a specific tool."""
        # Extract data from previous steps
        extracted_data = {}
        for step in chain.steps:
            if step.output_data and step.step_type == StepType.ANALYSIS:
                extracted_data.update(step.output_data)

        # Tool-specific input preparation
        if tool_name == "financial_calculator":
            return {
                "operation": self._infer_calculation_type(chain.question),
                "data": extracted_data,
                "question": chain.question
            }
        elif tool_name == "code_executor":
            return {
                "problem": chain.question,
                "data": extracted_data,
                "language": "python"
            }
        else:
            return {"question": chain.question, "data": extracted_data}

    def _infer_calculation_type(self, question: str) -> str:
        """Infer the type of calculation needed."""
        question_lower = question.lower()

        if any(term in question_lower for term in ["npv", "net present value"]):
            return "npv"
        elif any(term in question_lower for term in ["irr", "internal rate"]):
            return "irr"
        elif any(term in question_lower for term in ["roi", "return on investment"]):
            return "roi"
        elif any(term in question_lower for term in ["ratio", "current ratio", "debt"]):
            return "ratio"
        else:
            return "general"

    async def _validate_calculation_results(self, chain: ReasoningChain) -> ReasoningStep:
        """Validate calculation results for reasonableness."""
        step = ReasoningStep(
            step_type=StepType.VALIDATION,
            description="Validate calculation results",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Get calculation results from previous step
            calculation_results = None
            for prev_step in reversed(chain.steps):
                if prev_step.step_type == StepType.CALCULATION and prev_step.output_data:
                    calculation_results = prev_step.output_data.get("calculation_results", [])
                    break

            if not calculation_results:
                step.output_data = {"validation": "no_results_to_validate"}
                step.status = StepStatus.SKIPPED
                return step

            validation_results = []

            for calc_result in calculation_results:
                if calc_result["status"] == "success" and calc_result["result"]:
                    validation = await self._validate_single_result(calc_result)
                    validation_results.append(validation)

            step.output_data = {
                "validation_results": validation_results,
                "overall_validity": "valid" if all(v.get("valid", False) for v in validation_results) else "questionable"
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _validate_single_result(self, calc_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single calculation result."""
        # Simple validation logic - can be enhanced
        result_data = calc_result.get("result", {})

        validation = {
            "tool": calc_result["tool"],
            "valid": True,
            "warnings": []
        }

        # Check for common issues
        if isinstance(result_data, dict):
            if "error" in result_data:
                validation["valid"] = False
                validation["warnings"].append("Tool returned error")

            # Add more validation rules here

        return validation

    async def _identify_missing_data(self, chain: ReasoningChain) -> ReasoningStep:
        """Identify missing data for assumption-based questions."""
        step = ReasoningStep(
            step_type=StepType.ASSUMPTION,
            description="Identify missing data",
            status=StepStatus.IN_PROGRESS,
            input_data={"question": chain.question}
        )

        start_time = datetime.now()

        try:
            # Use model to identify missing data
            missing_data_prompt = f"""
            Analyze this financial question and identify what data is missing or not provided:

            Question: {chain.question}

            List the specific pieces of missing information needed for calculation.
            """

            response = await self.model_manager.generate(missing_data_prompt)

            step.output_data = {
                "missing_data_analysis": response.content,
                "method": "llm_analysis"
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _generate_assumptions(self, chain: ReasoningChain) -> ReasoningStep:
        """Generate reasonable assumptions for missing data."""
        step = ReasoningStep(
            step_type=StepType.ASSUMPTION,
            description="Generate reasonable assumptions",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Get missing data analysis
            missing_data = None
            for prev_step in reversed(chain.steps):
                if prev_step.step_type == StepType.ASSUMPTION and prev_step.output_data:
                    missing_data = prev_step.output_data.get("missing_data_analysis")
                    break

            assumption_prompt = f"""
            Based on the missing data analysis, generate reasonable assumptions:

            Missing Data: {missing_data}
            Question Context: {chain.question}

            Provide specific, reasonable assumptions with justification.
            """

            response = await self.model_manager.generate(assumption_prompt)

            step.output_data = {
                "generated_assumptions": response.content,
                "confidence": 0.7  # Lower confidence due to assumptions
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _perform_sensitivity_analysis(self, chain: ReasoningChain) -> ReasoningStep:
        """Perform sensitivity analysis for assumption-based calculations."""
        step = ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="Perform sensitivity analysis",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            step.output_data = {
                "sensitivity_analysis": "Sensitivity analysis placeholder - would test assumption variations",
                "confidence_bounds": "Would provide confidence intervals"
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _retrieve_financial_knowledge(self, chain: ReasoningChain) -> ReasoningStep:
        """Retrieve relevant financial knowledge for conceptual questions."""
        step = ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="Retrieve financial knowledge",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Use RAG system if available
            if "knowledge_base" in self.tools:
                result = await self.tools["knowledge_base"].execute({
                    "query": chain.question,
                    "retrieval_type": "conceptual"
                })
                step.output_data = result
                step.tool_used = "knowledge_base"
            else:
                # Fallback to model knowledge
                knowledge_prompt = f"""
                Provide relevant financial knowledge for this question:

                Question: {chain.question}

                Include key concepts, definitions, and relevant information.
                """

                response = await self.model_manager.generate(knowledge_prompt)
                step.output_data = {
                    "retrieved_knowledge": response.content,
                    "method": "model_knowledge"
                }

            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _structure_explanation(self, chain: ReasoningChain) -> ReasoningStep:
        """Structure the explanation for conceptual questions."""
        step = ReasoningStep(
            step_type=StepType.SYNTHESIS,
            description="Structure explanation",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            # Get knowledge from previous step
            knowledge = None
            for prev_step in reversed(chain.steps):
                if prev_step.step_type == StepType.ANALYSIS and prev_step.output_data:
                    knowledge = prev_step.output_data.get("retrieved_knowledge")
                    break

            structure_prompt = f"""
            Structure this knowledge into a clear, logical explanation:

            Question: {chain.question}
            Knowledge: {knowledge}

            Organize into clear sections with logical flow.
            """

            response = await self.model_manager.generate(structure_prompt)

            step.output_data = {
                "structured_explanation": response.content
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _add_relevant_examples(self, chain: ReasoningChain) -> ReasoningStep:
        """Add relevant examples to conceptual explanations."""
        step = ReasoningStep(
            step_type=StepType.SYNTHESIS,
            description="Add relevant examples",
            status=StepStatus.IN_PROGRESS
        )

        start_time = datetime.now()

        try:
            examples_prompt = f"""
            Add practical examples to illustrate the concepts in this question:

            Question: {chain.question}

            Provide 1-2 concrete, relevant examples.
            """

            response = await self.model_manager.generate(examples_prompt)

            step.output_data = {
                "examples": response.content
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

    async def _synthesize_answer(self, chain: ReasoningChain) -> ReasoningStep:
        """Synthesize final answer from reasoning steps."""
        step = ReasoningStep(
            step_type=StepType.SYNTHESIS,
            description="Synthesize final answer",
            status=StepStatus.IN_PROGRESS,
            input_data={"all_steps": [s.to_dict() for s in chain.steps]}
        )

        start_time = datetime.now()

        try:
            # Gather all intermediate results
            context = ""
            for reasoning_step in chain.steps:
                if reasoning_step.output_data:
                    context += f"Step: {reasoning_step.description}\n"
                    context += f"Result: {reasoning_step.output_data}\n\n"
                    
                    
            synthesis_prompt = f"""
            You are a financial reasoning agent. Based on all the original and reasoning context, 
            provide the final answer to the question in the required format.

            Question:
            {chain.question}

            Reasoning Context:
            {context}

            Instructions:
            1. Show your reasoning briefly (if necessary).
            2. Always end with the exact final answer on a new line, 
            starting with '<final_answer>:' and nothing else before it.
            3. The final answer must be concise, in the correct financial format 

            Now, provide your response.
            """

            response = await self.model_manager.generate(synthesis_prompt)

            # Set final answer on chain
            chain.final_answer = response.content

            step.output_data = {
                "final_answer": response.content,
                "confidence_assessment": "to_be_determined",
                "raw_model_response": response.to_dict() if hasattr(response, 'to_dict') else str(response)
            }
            step.status = StepStatus.COMPLETED

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)

        finally:
            end_time = datetime.now()
            step.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return step

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


    # Essential Error Handling Methods

    async def _safe_execute_step(self, step_func, *args, **kwargs) -> ReasoningStep:
        """Safely execute a reasoning step with error handling."""
        try:
            return await step_func(*args, **kwargs)
        except Exception as e:
            # Create failed step
            step = ReasoningStep(
                step_type=StepType.VALIDATION,
                description=f"Failed step: {step_func.__name__}",
                status=StepStatus.FAILED,
                error_message=str(e)
            )
            logger.error(f"Step {step_func.__name__} failed: {e}")
            return step

    async def _safe_model_generate(self, prompt: str, fallback_message: str = None, timeout: int = 60) -> str:
        """Safely generate LLM response with fallback and timeout."""
        try:
            # Add timeout to prevent hanging on model requests
            response = await asyncio.wait_for(
                self.model_manager.generate(prompt),
                timeout=timeout
            )
            if not response.content or response.content.strip() == "":
                logger.warning("Empty response from model, using fallback")
                return fallback_message or "Unable to generate response due to technical issues."
            return response.content
        except asyncio.TimeoutError:
            logger.error(f"LLM generation timed out after {timeout}s")
            return fallback_message or "Request timed out."
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return fallback_message or "Unable to generate response due to technical issues."

    async def _safe_tool_execute(self, tool_name: str, tool_input: Dict[str, Any],
                               question: str) -> Dict[str, Any]:
        """Safely execute tool with LLM fallback."""
        try:
            if tool_name in self.tools:
                result = await self.tools[tool_name].execute(tool_input)
                return {"status": "success", "result": result, "tool": tool_name}
        except Exception as e:
            logger.warning(f"Tool {tool_name} failed: {e}, falling back to LLM")

        # LLM fallback
        try:
            fallback_prompt = f"""
            The {tool_name} tool failed. Please answer this question directly:

            Question: {question}

            Provide the best answer you can.
            """

            llm_response = await self._safe_model_generate(
                fallback_prompt,
                "Unable to process this calculation."
            )

            return {
                "status": "llm_fallback",
                "result": {"response": llm_response},
                "tool": "llm"
            }

        except Exception as fallback_error:
            logger.error(f"Both tool and LLM fallback failed: {fallback_error}")
            return {
                "status": "failed",
                "result": {"error": "Both tool and fallback failed"},
                "tool": "none"
            }

    def _ensure_final_answer(self, chain: ReasoningChain) -> None:
        """Ensure chain always has a final answer."""
        if not chain.final_answer or chain.final_answer.strip() == "":
            # Check if any step has usable output
            for step in reversed(chain.steps):
                if step.status == StepStatus.COMPLETED and step.output_data:
                    if isinstance(step.output_data, dict):
                        # Extract any meaningful response
                        for key in ['final_answer', 'response', 'result', 'analysis']:
                            if key in step.output_data and step.output_data[key]:
                                chain.final_answer = str(step.output_data[key])
                                return

            # Last resort - generic message
            chain.final_answer = "Analysis incomplete due to technical issues. Please try again or consult financial documentation."