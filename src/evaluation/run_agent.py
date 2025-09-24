#!/usr/bin/env python3
"""
FinanceQA Agent Result Generation Script

Standalone script for running agents on the FinanceQA benchmark to generate
result files. This script only calls agents and saves their responses -
no evaluation/scoring is performed here.
"""

import sys
import time
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from src.agent.financial_agent import FinancialAgent

import pandas as pd
from tqdm import tqdm


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

    parser.add_argument("--question-type", type=str,
                        help="Filter dataset by question type (e.g., 'calculation', 'conceptual')")
    parser.add_argument("--num-samples", type=int,
                        help="Limit processing to N samples (default: process all available)")

    # RAG control arguments
    parser.add_argument("--disable-rag", action="store_true",
                        help="Disable RAG (Retrieval-Augmented Generation) tools")

    return parser


def validate_evaluation_arguments(args) -> None:
    """
    Validate evaluation arguments for consistency.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid
    """
    if hasattr(args, 'num_samples') and args.num_samples and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    if hasattr(args, 'question_type') and args.question_type and not args.question_type.strip():
        raise ValueError("--question-type cannot be empty")

    # Log RAG status if disable_rag flag is present
    if hasattr(args, 'disable_rag') and args.disable_rag:
        print("🔧 RAG tools disabled via --disable-rag flag")


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


def setup_agent(model_manager, config: Dict[str, Any], enable_rag: bool = True):
    """
    Setup the FinanceQA agent with model manager and tools.

    Args:
        model_manager: Model manager instance
        config: Configuration dictionary
        enable_rag: Whether to enable RAG tools (default: True)

    Returns:
        Configured FinancialAgent instance
    """

    return FinancialAgent(model_manager, config=config, enable_rag=enable_rag)


class AgentRunner:
    """Runner for executing agents on FinanceQA dataset and saving results."""

    def __init__(self, dataset_path: str = "data/datasets/financeqa"):
        """
        Initialize the agent runner.

        Args:
            dataset_path: Path to the FinanceQA dataset directory
        """
        self.dataset_path = Path(dataset_path)

    def _prepare_dataset(self,
                        split: str,
                        question_type: Optional[str] = None,
                        num_samples: Optional[int] = None) -> Tuple[List, str]:
        """
        Load and filter dataset examples based on criteria.

        Args:
            split: Dataset split to use ("test", "train", "validation")
            question_type: Filter by question type (optional)
            num_samples: Limit to N samples (optional)

        Returns:
            Tuple of (filtered_examples, run_description)
        """
        # Load all examples
        all_examples = load_dataset(str(self.dataset_path), split)

        # Apply question type filter
        if question_type:
            examples = [ex for ex in all_examples if ex.question_type == question_type]
            print(f"Found {len(examples)} {question_type} examples out of {len(all_examples)} total")
            if not examples:
                print(f"No examples found for question type: {question_type}")
                return [], ""
        else:
            examples = all_examples

        # Apply sample limit
        if num_samples:
            examples = examples[:num_samples]

        # Generate run description
        desc_parts = []
        if question_type:
            desc_parts.append(question_type.upper())
        if num_samples:
            desc_parts.append(f"{len(examples)} samples")
        else:
            desc_parts.append("FULL")
        run_description = " - ".join(desc_parts)

        return examples, run_description

    def _call_agent(self, agent: FinancialAgent, example):
        """
        Call agent.answer_question synchronously.

        Args:
            agent: Agent implementing the FinancialAgent protocol
            question: Question to ask the agent

        Returns:
            Result from agent.answer_question
        """

        # Combine context and question
        full_question = f"original_question: {example.question} \n original_context: {example.context}\n" if example.context else f"original_question: {example.question}"
        return agent.answer_question(full_question)

    def run_agent_on_dataset(
        self,
        agent: FinancialAgent,
        split: str = "test",
        question_type: Optional[str] = None,
        num_samples: Optional[int] = None,
        output_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run agent on FinanceQA dataset with flexible filtering and output options.

        Args:
            agent: Agent implementing the FinancialAgent protocol
            split: Dataset split to use ("test", "train", "validation")
            question_type: Filter by question type (optional)
            num_samples: Limit to N samples (optional)
            output_file: Custom output filename (optional)
            verbose: Whether to show progress bars

        Returns:
            Dictionary containing agent results
        """
        # Prepare dataset
        examples, run_description = self._prepare_dataset(split, question_type, num_samples)

        # Handle case where no examples found
        if not examples:
            return {}

        print(f"Running agent on {len(examples)} examples from {split} split ({run_description})")

        # Process examples
        results = []
        total_processing_time = 0.0
        iterator = tqdm(examples, desc="Running Agent") if verbose else examples

        for example in iterator:
            try:

                # Time the agent response
                start_time = time.time()
                reasoning_chain = self._call_agent(agent, example)
                processing_time = time.time() - start_time

                # Extract answer from reasoning chain
                answer = getattr(reasoning_chain, 'final_answer', '') or ''

                # Extract reasoning steps and confidence
                reasoning_steps = []
                confidence_score = 0.5

                if hasattr(reasoning_chain, 'steps'):
                    reasoning_steps = [step.description for step in reasoning_chain.steps
                                     if hasattr(step, 'description')]

                    failed_steps = len([s for s in reasoning_chain.steps
                                      if hasattr(s, 'status') and s.status.value == 'failed'])
                    total_steps = len(reasoning_chain.steps)
                    confidence_score = max(0.1, 1.0 - (failed_steps / total_steps)) if total_steps > 0 else 0.5

                agent_response = AgentResponse(
                    answer=answer,
                    reasoning_steps=reasoning_steps,
                    confidence_score=confidence_score,
                    processing_time=processing_time
                )

                result = EvaluationResult(
                    example=example,
                    agent_response=agent_response,
                    processing_time=processing_time,
                    llm_match=None,
                    llm_reasoning=None
                )

            except Exception as e:
                result = EvaluationResult(
                    example=example,
                    agent_response=AgentResponse(answer=""),
                    processing_time=0.0,
                    error_message=str(e),
                    llm_match=None,
                    llm_reasoning=None
                )

            results.append(result)
            total_processing_time += result.processing_time

        # Generate output filename if not provided
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename_parts = ["financeqa_agent_results", split]
            if question_type:
                filename_parts.append(question_type)
            if num_samples:
                filename_parts.append(str(num_samples))
            else:
                filename_parts.append("full")
            filename_parts.append(timestamp)
            output_file = "_".join(filename_parts) + ".json"

        # Save results
        output_path = Path("results") / output_file
        output_path.parent.mkdir(exist_ok=True)

        # Simple summary metrics
        errors = sum(1 for r in results if r.error_message is not None)
        metrics = {
            'total_examples': len(results),
            'errors': errors,
            'error_rate': errors / len(results) if results else 0.0,
            'agent_run_complete': True,
            'evaluation_pending': True
        }

        final_results = {
            'results': results,
            'metrics': metrics,
            'total_examples': len(results),
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(results) if results else 0.0
        }

        save_results(final_results, str(output_path))

        # Print summary
        self._print_summary(final_results, split, run_description)

        return final_results


    def _print_summary(self, results: Dict[str, Any], split: str, run_type: str):
        """Print run summary."""
        metrics = results['metrics']

        print(f"\nAGENT RUN SUMMARY - {split.upper()} SPLIT - {run_type}")
        print("=" * 60)
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Agent Errors: {metrics['errors']} ({metrics['error_rate']:.1%})")
        print(f"Avg Processing Time: {results['average_processing_time']:.2f}s")
        print(f"Total Processing Time: {results['total_processing_time']:.1f}s")
        print(f"✅ Agent results generated successfully")
        print(f"📝 Next step: Use run_llm_evaluation.py to evaluate these results")


def main():
    """Command-line interface for agent result generation."""
    # Create argument parser
    parser = create_base_argument_parser("FinanceQA Agent Result Generation")
    parser = add_agent_evaluation_arguments(parser)

    args = parser.parse_args()

    try:
        # Validate arguments
        validate_evaluation_arguments(args)

        # Setup model manager with context manager for automatic cleanup
        print("🔧 Setting up model manager...")
        model_manager, config = setup_model_manager()
        print(f"✅ Model manager setup complete using {config['default_provider']}")

        with model_manager as mm:
            print("🤖 Setting up FinanceQA agent...")
            agent = setup_agent(mm, config)
            print("✅ FinanceQA agent setup complete")

            # Initialize runner
            runner = AgentRunner(dataset_path=args.dataset_path)
            verbose = not args.no_verbose

            try:
                # Use consolidated interface
                runner.run_agent_on_dataset(
                    agent=agent,
                    split=args.split,
                    question_type=getattr(args, 'question_type', None),
                    num_samples=getattr(args, 'num_samples', None),
                    output_file=args.output_file,
                    verbose=verbose
                )

            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Make sure the FinanceQA dataset is downloaded and in the correct location.")
                sys.exit(1)
            except Exception as e:
                print(f"Agent run failed: {e}")
                sys.exit(1)

            print(f"\n✅ Agent run completed successfully!")
            print(f"📄 Results saved to results/ directory")
            print(f"🔄 Next step: Use run_llm_evaluation.py to evaluate these results")

    except Exception as e:
        print(f"❌ Failed to setup agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()