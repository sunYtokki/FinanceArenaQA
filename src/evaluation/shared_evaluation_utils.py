"""
Shared utilities for FinanceQA evaluation components.

This module provides common data classes, protocols, and utilities used across
the evaluation system components (agent evaluation, LLM evaluation, and orchestrator).
"""

import json
import time
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass

import pandas as pd


@dataclass
class EvaluationExample:
    """Represents a single evaluation example from the FinanceQA dataset."""
    context: str
    question: str
    chain_of_thought: str
    answer: str
    file_link: str
    file_name: str
    company: str
    question_type: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationExample':
        """Create an EvaluationExample from a dictionary."""
        return cls(
            context=data['context'],
            question=data['question'],
            chain_of_thought=data['chain_of_thought'],
            answer=data['answer'],
            file_link=data['file_link'],
            file_name=data['file_name'],
            company=data['company'],
            question_type=data['question_type']
        )


@dataclass
class AgentResponse:
    """Represents an agent's response to a question."""
    answer: str
    reasoning_steps: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None


@dataclass
class EvaluationResult:
    """Results of evaluating a single question (LLM-based evaluation only)."""
    example: EvaluationExample
    agent_response: AgentResponse
    processing_time: float
    error_message: Optional[str] = None
    llm_match: Optional[bool] = None
    llm_reasoning: Optional[str] = None


class FinancialAgent(Protocol):
    """Protocol defining the interface for financial QA agents."""

    def answer_question(self, question: str) -> Any:
        """
        Answer a financial question.

        Args:
            question: Question to answer (can include context)

        Returns:
            ReasoningChain or any object with final_answer and steps
        """
        ...


def load_dataset(dataset_path: str, split: str = "test") -> List[EvaluationExample]:
    """
    Load dataset examples from a specific split.

    Args:
        dataset_path: Path to the FinanceQA dataset directory
        split: Dataset split to load ("test", "train", "validation")

    Returns:
        List of evaluation examples
    """
    dataset_path = Path(dataset_path)
    jsonl_path = dataset_path / f"{split}.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(EvaluationExample.from_dict(data))

    return examples


def save_results(evaluation_output: Dict[str, Any], output_path: str):
    """
    Save evaluation results to a file.

    Args:
        evaluation_output: Output from evaluation functions
        output_path: Path to save results
    """
    # Convert results to serializable format
    serializable_results = []
    for result in evaluation_output['results']:
        serializable_result = {
            'question': result.example.question,
            'ground_truth': result.example.answer,
            'predicted_answer': result.agent_response.answer,
            'question_type': result.example.question_type,
            'company': result.example.company,
            'processing_time': result.processing_time,
            'error_message': result.error_message,
            'reasoning_steps': result.agent_response.reasoning_steps,
            'confidence_score': result.agent_response.confidence_score,
            'llm_match': result.llm_match,
            'llm_reasoning': result.llm_reasoning
        }
        serializable_results.append(serializable_result)

    output_data = {
        'evaluation_summary': evaluation_output['metrics'],
        'total_examples': evaluation_output['total_examples'],
        'total_processing_time': evaluation_output['total_processing_time'],
        'average_processing_time': evaluation_output['average_processing_time'],
        'detailed_results': serializable_results,
        'evaluation_timestamp': pd.Timestamp.now().isoformat()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to {output_path}")


def load_results(result_file_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from a file.

    Args:
        result_file_path: Path to the result file

    Returns:
        Dictionary containing evaluation results

    Raises:
        FileNotFoundError: If result file doesn't exist
        json.JSONDecodeError: If result file is malformed
    """
    result_path = Path(result_file_path)

    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_file_path}")

    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_result_file(result_data: Dict[str, Any]) -> bool:
    """
    Validate that a result file has the expected structure.

    Args:
        result_data: Loaded result file data

    Returns:
        True if valid, False otherwise
    """
    required_fields = ['detailed_results', 'evaluation_summary', 'total_examples']

    for field in required_fields:
        if field not in result_data:
            return False

    # Check that detailed_results has the expected structure
    if not isinstance(result_data['detailed_results'], list):
        return False

    if result_data['detailed_results']:
        result_fields = ['question', 'ground_truth', 'predicted_answer']
        sample_result = result_data['detailed_results'][0]

        for field in result_fields:
            if field not in sample_result:
                return False

    return True


def create_base_argument_parser(description: str) -> argparse.ArgumentParser:
    """
    Create a base argument parser with common arguments for evaluation scripts.

    Args:
        description: Description for the argument parser

    Returns:
        ArgumentParser with common evaluation arguments
    """
    parser = argparse.ArgumentParser(description=description)

    # Dataset and output arguments
    parser.add_argument("--dataset-path", default="data/datasets/financeqa",
                        help="Path to FinanceQA dataset directory")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--split", default="test", choices=["test", "train", "validation"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--output-file",
                        help="Custom output filename")
    parser.add_argument("--no-verbose", action="store_true",
                        help="Disable progress bars")

    return parser


def add_agent_evaluation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add agent evaluation specific arguments to the parser.

    Args:
        parser: Base argument parser

    Returns:
        Parser with agent evaluation arguments added
    """
    parser.add_argument("--evaluation-type", default="quick",
                        choices=["full", "quick", "by-type"],
                        help="Type of evaluation to run")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples for quick evaluation")
    parser.add_argument("--question-type",
                        help="Question type filter for by-type evaluation")

    return parser


def add_llm_evaluation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add LLM evaluation specific arguments to the parser.

    Args:
        parser: Base argument parser

    Returns:
        Parser with LLM evaluation arguments added
    """
    parser.add_argument("--result-file", required=True,
                        help="Path to result file to evaluate with LLM")
    parser.add_argument("--evaluation-model",
                        help="Model to use for LLM evaluation")
    parser.add_argument("--failed-examples", default="process",
                        choices=["skip", "process", "flag"],
                        help="How to handle failed agent examples")

    return parser


def add_orchestrator_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add orchestrator specific arguments to the parser.

    Args:
        parser: Base argument parser

    Returns:
        Parser with orchestrator arguments added
    """
    # Add both agent and LLM evaluation arguments
    parser = add_agent_evaluation_arguments(parser)

    parser.add_argument("--use-llm-evaluation", action="store_true",
                        help="Use LLM-based evaluation instead of exact match")
    parser.add_argument("--evaluation-model",
                        help="Model to use for LLM evaluation (if different from agent model)")

    return parser


def setup_model_manager(config_path: str = "config/model_config.json"):
    """
    Setup model manager from configuration file.

    Args:
        config_path: Path to model configuration file

    Returns:
        Tuple of (model_manager, config_dict)

    Raises:
        Exception: If model manager setup fails
    """
    try:
        from src.models.model_manager import create_model_manager_from_config

        # Load config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {"default_provider": "ollama"}

        # Create model manager
        model_manager = create_model_manager_from_config(config)
        return model_manager, config

    except Exception as e:
        raise Exception(f"Failed to setup model manager: {e}")


def setup_agent(model_manager, config: Dict[str, Any]):
    """
    Setup the FinanceQA agent with model manager and tools.

    Args:
        model_manager: Model manager instance
        config: Configuration dictionary

    Returns:
        Configured agent instance

    Raises:
        Exception: If agent setup fails
    """
    try:
        from src.agent.core import Agent
        from src.tools.financial_calculator import FinancialCalculator

        # Create tools
        financial_calc = FinancialCalculator()

        # Create agent
        agent = Agent(model_manager, tools=[financial_calc])

        # Add question classifier if possible
        try:
            from src.classifiers.question_classifier import QuestionClassifier
            classifier = QuestionClassifier()
            agent.question_classifier = classifier
        except Exception:
            pass  # Question classifier is optional

        return agent

    except Exception as e:
        raise Exception(f"Failed to setup agent: {e}")


def setup_llm_scorer(model_manager, evaluation_model: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """
    Setup LLM scorer for evaluation.

    Args:
        model_manager: Model manager instance
        evaluation_model: Specific model to use for evaluation (optional)
        config: Configuration dictionary with evaluation settings (optional)

    Returns:
        LLMScorer instance

    Raises:
        Exception: If LLM scorer setup fails
    """
    try:
        from .llm_scorer import LLMScorer

        # Use evaluation-specific model if specified in config or parameter
        eval_model = evaluation_model

        if config and 'evaluation' in config and not eval_model:
            eval_config = config['evaluation']
            eval_model = eval_config.get('model_name')

            # Log evaluation config being used
            print(f"📊 Using evaluation-specific config:")
            print(f"   Model: {eval_model}")
            print(f"   Provider: {eval_config.get('provider', 'default')}")
            if eval_config.get('base_url'):
                print(f"   Base URL: {eval_config['base_url']}")

        scorer = LLMScorer(model_manager, model_name=eval_model, config=config)
        return scorer

    except Exception as e:
        raise Exception(f"Failed to setup LLM scorer: {e}")


def compute_metrics_llm_only(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Compute evaluation metrics for LLM-based evaluation only.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary of metrics (LLM-based only, no exact match metrics)
    """
    if not results:
        return {}

    # Overall LLM accuracy
    llm_matches = sum(1 for r in results if r.llm_match is True)
    errors = sum(1 for r in results if r.error_message is not None)
    total = len(results)

    metrics = {
        'llm_match_accuracy': llm_matches / total if total > 0 else 0.0,
        'error_rate': errors / total if total > 0 else 0.0,
        'total_examples': total,
        'llm_matches': llm_matches,
        'errors': errors
    }

    # Accuracy by question type
    question_types = {}
    for result in results:
        qtype = result.example.question_type
        if qtype not in question_types:
            question_types[qtype] = {'total': 0, 'llm': 0, 'errors': 0}

        question_types[qtype]['total'] += 1
        if result.llm_match is True:
            question_types[qtype]['llm'] += 1
        if result.error_message:
            question_types[qtype]['errors'] += 1

    # Compute per-type accuracies
    by_question_type = {}
    for qtype, counts in question_types.items():
        type_metrics = {
            'llm_match_accuracy': counts['llm'] / counts['total'] if counts['total'] > 0 else 0.0,
            'error_rate': counts['errors'] / counts['total'] if counts['total'] > 0 else 0.0,
            'total_examples': counts['total']
        }
        by_question_type[qtype] = type_metrics

    metrics['by_question_type'] = by_question_type

    return metrics