"""Simplified 5-step reasoning chain for financial question answering."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with input data."""
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate tool input data."""
        return True
