"""
FinanceQA Benchmark Evaluation Harness

This module provides functionality to evaluate financial QA agents on the FinanceQA benchmark
dataset with exact match scoring and detailed performance metrics.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

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
    """Results of evaluating a single question."""
    example: EvaluationExample
    agent_response: AgentResponse
    exact_match: bool
    normalized_match: bool
    processing_time: float
    error_message: Optional[str] = None


class FinancialAgent(Protocol):
    """Protocol defining the interface for financial QA agents."""

    async def answer_question(self, question: str) -> Any:
        """
        Answer a financial question.

        Args:
            question: Question to answer (can include context)

        Returns:
            ReasoningChain or any object with final_answer and steps
        """
        ...


class ExactMatchScorer:
    """Exact match scorer with normalization for financial answers."""

    @staticmethod


    def normalize_financial_answer(answer: str) -> str:
        """Normalize financial answers (e.g., '$1,000 mln' -> '1000 million')."""
        if not answer:
            return ""

        s = answer.lower().strip()

        # Remove currency symbols and thousands separators (don't insert spaces)
        s = re.sub(r'[$,]', '', s)

        # Remove notes like "(in millions)"
        s = re.sub(r'\(in\s+millions?\)', '', s)

        # Abbreviation & plural normalization
        s = re.sub(r'\bmln\b', 'million', s)
        s = re.sub(r'\bbln\b', 'billion', s)
        s = re.sub(r'\bmillion[s]?\b', 'million', s)
        s = re.sub(r'\bbillion[s]?\b', 'billion', s)
        s = re.sub(r'\bthousand[s]?\b', 'thousand', s)

        # Whitespace to single spaces
        s = re.sub(r'\s+', ' ', s).strip()

        return s
    

    @staticmethod
    def extract_numeric_value(answer: str) -> Optional[float]:
        """
        Extract the primary numeric value from a financial answer.

        Args:
            answer: Financial answer string

        Returns:
            Extracted numeric value or None
        """
        # Remove common financial formatting
        cleaned = re.sub(r'[,$\s]+', '', answer)

        # Find all numbers (including decimals and negatives)
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)

        if not numbers:
            return None

        # Return the first/main number found
        try:
            return float(numbers[0])
        except ValueError:
            return None

    def compute_exact_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Compute exact match between predicted and ground truth answers.

        Args:
            predicted: Agent's predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if exact match, False otherwise
        """
        return predicted.strip() == ground_truth.strip()

    def compute_normalized_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Compute normalized match between predicted and ground truth answers.

        Args:
            predicted: Agent's predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if normalized match, False otherwise
        """
        norm_pred = self.normalize_financial_answer(predicted)
        norm_gt = self.normalize_financial_answer(ground_truth)

        return norm_pred == norm_gt


class FinanceQAEvaluator:
    """Main evaluator class for the FinanceQA benchmark."""

    def __init__(self, dataset_path: str = "data/datasets/financeqa"):
        """
        Initialize the evaluator.

        Args:
            dataset_path: Path to the FinanceQA dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.scorer = ExactMatchScorer()

    def load_dataset(self, split: str = "test") -> List[EvaluationExample]:
        """
        Load dataset examples from a specific split.

        Args:
            split: Dataset split to load ("test", "train", "validation")

        Returns:
            List of evaluation examples
        """
        jsonl_path = self.dataset_path / f"{split}.jsonl"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

        examples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(EvaluationExample.from_dict(data))

        return examples

    async def evaluate_agent_async(
        self,
        agent: FinancialAgent,
        examples: Optional[List[EvaluationExample]] = None,
        split: str = "test",
        max_examples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a financial agent on the FinanceQA benchmark.

        Args:
            agent: Agent implementing the FinancialAgent protocol
            examples: Specific examples to evaluate (if None, loads from split)
            split: Dataset split to use if examples not provided
            max_examples: Maximum number of examples to evaluate
            verbose: Whether to show progress bars

        Returns:
            Dictionary containing evaluation results and metrics
        """
        if examples is None:
            examples = self.load_dataset(split)

        if max_examples:
            examples = examples[:max_examples]

        results = []
        total_processing_time = 0.0

        # Progress bar
        iterator = tqdm(examples, desc="Evaluating") if verbose else examples

        # Handle async execution
        import asyncio

        async def evaluate_single_example(example):
            try:
                # Combine context and question
                full_question = f"Context: {example.context}\n\nQuestion: {example.question}" if example.context else example.question

                # Time the agent response
                start_time = time.time()
                reasoning_chain = await agent.answer_question(full_question)
                processing_time = time.time() - start_time

                # Extract answer from reasoning chain
                answer = getattr(reasoning_chain, 'final_answer', '') or ''

                # Create agent response for compatibility
                reasoning_steps = []
                confidence_score = 0.5

                if hasattr(reasoning_chain, 'steps'):
                    # Extract reasoning steps
                    reasoning_steps = [step.description for step in reasoning_chain.steps
                                     if hasattr(step, 'description')]

                    # Calculate confidence based on success rate
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

                # Compute scores
                exact_match = self.scorer.compute_exact_match(answer, example.answer)
                normalized_match = self.scorer.compute_normalized_match(answer, example.answer)

                return EvaluationResult(
                    example=example,
                    agent_response=agent_response,
                    exact_match=exact_match,
                    normalized_match=normalized_match,
                    processing_time=processing_time
                )

            except Exception as e:
                # Handle agent errors gracefully
                return EvaluationResult(
                    example=example,
                    agent_response=AgentResponse(answer=""),
                    exact_match=False,
                    normalized_match=False,
                    processing_time=0.0,
                    error_message=str(e)
                )

        # Run async evaluation with concurrency limits to prevent resource exhaustion
        async def run_evaluation():
            # Limit concurrent evaluations to prevent Ollama server overload
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent evaluations

            async def evaluate_with_semaphore(example):
                async with semaphore:
                    return await evaluate_single_example(example)

            # Add timeout per evaluation to prevent infinite loops
            async def evaluate_with_timeout(example):
                try:
                    return await asyncio.wait_for(evaluate_with_semaphore(example), timeout=300.0)  # 5 min timeout
                except asyncio.TimeoutError:
                    return EvaluationResult(
                        example=example,
                        agent_response=AgentResponse(answer="Timeout: Agent took too long to respond"),
                        exact_match=False,
                        normalized_match=False,
                        processing_time=300.0,
                        error_message="Evaluation timeout after 300 seconds"
                    )

            tasks = [evaluate_with_timeout(example) for example in examples]
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Execute async evaluation
        try:
            results = await run_evaluation()
            total_processing_time = sum(r.processing_time for r in results)
        finally:
            # Clean up any sessions in the model manager
            if hasattr(agent, 'model_manager') and hasattr(agent.model_manager, 'close_all'):
                try:
                    await agent.model_manager.close_all()
                except Exception as e:
                    # Don't fail the evaluation for cleanup issues
                    pass

        # Compute overall metrics
        metrics = self.compute_metrics(results)

        return {
            'results': results,
            'metrics': metrics,
            'total_examples': len(results),
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(results) if results else 0.0
        }

    def evaluate_agent(
        self,
        agent: FinancialAgent,
        examples: Optional[List[EvaluationExample]] = None,
        split: str = "test",
        max_examples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_agent_async to maintain backwards compatibility.

        Args:
            agent: Agent implementing the FinancialAgent protocol
            examples: Specific examples to evaluate (if None, loads from split)
            split: Dataset split to use if examples not provided
            max_examples: Maximum number of examples to evaluate
            verbose: Whether to show progress bars

        Returns:
            Dictionary containing evaluation results and metrics
        """
        try:
            # Try to get the current running loop
            loop = asyncio.get_running_loop()
            # If we're in a loop, we need to run in a new thread to avoid deadlock
            import concurrent.futures
            import threading

            def run_in_thread():
                # Create new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self.evaluate_agent_async(agent, examples, split, max_examples, verbose)
                    )
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()

        except RuntimeError:
            # No running loop, can use asyncio.run directly
            return asyncio.run(
                self.evaluate_agent_async(agent, examples, split, max_examples, verbose)
            )

    def compute_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Compute evaluation metrics from results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary of metrics
        """
        if not results:
            return {}

        # Overall accuracy
        exact_matches = sum(1 for r in results if r.exact_match)
        normalized_matches = sum(1 for r in results if r.normalized_match)
        errors = sum(1 for r in results if r.error_message is not None)

        total = len(results)

        metrics = {
            'exact_match_accuracy': exact_matches / total,
            'normalized_match_accuracy': normalized_matches / total,
            'error_rate': errors / total,
            'total_examples': total,
            'exact_matches': exact_matches,
            'normalized_matches': normalized_matches,
            'errors': errors
        }

        # Accuracy by question type
        question_types = {}
        for result in results:
            qtype = result.example.question_type
            if qtype not in question_types:
                question_types[qtype] = {'total': 0, 'exact': 0, 'normalized': 0, 'errors': 0}

            question_types[qtype]['total'] += 1
            if result.exact_match:
                question_types[qtype]['exact'] += 1
            if result.normalized_match:
                question_types[qtype]['normalized'] += 1
            if result.error_message:
                question_types[qtype]['errors'] += 1

        # Compute per-type accuracies
        by_question_type = {}
        for qtype, counts in question_types.items():
            by_question_type[qtype] = {
                'exact_match_accuracy': counts['exact'] / counts['total'],
                'normalized_match_accuracy': counts['normalized'] / counts['total'],
                'error_rate': counts['errors'] / counts['total'],
                'total_examples': counts['total']
            }

        metrics['by_question_type'] = by_question_type

        return metrics

    def save_results(self, evaluation_output: Dict[str, Any], output_path: str):
        """
        Save evaluation results to a file.

        Args:
            evaluation_output: Output from evaluate_agent()
            output_path: Path to save results
        """
        # Convert results to serializable format
        serializable_results = []
        for result in evaluation_output['results']:
            serializable_result = {
                'question': result.example.question,
                'ground_truth': result.example.answer,
                'predicted_answer': result.agent_response.answer,
                'exact_match': result.exact_match,
                'normalized_match': result.normalized_match,
                'question_type': result.example.question_type,
                'company': result.example.company,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'reasoning_steps': result.agent_response.reasoning_steps,
                'confidence_score': result.agent_response.confidence_score
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


class DummyAgent:
    """Dummy agent for testing the evaluation harness."""

    async def answer_question(self, question: str):
        """
        Dummy implementation that returns a simple response.

        Args:
            question: Question to answer (including context)

        Returns:
            Mock reasoning chain with final_answer
        """
        # Simple pattern matching for demonstration
        class MockChain:
            def __init__(self, answer):
                self.final_answer = answer
                self.steps = []

        if "gross profit" in question.lower():
            return MockChain("$32,095 (in millions)")
        elif "revenue" in question.lower() or "total revenue" in question.lower():
            return MockChain("$254,453")
        else:
            return MockChain("Unable to determine")


class BenchmarkRunner:
    """High-level runner for FinanceQA benchmark evaluation."""

    def __init__(self,
                 dataset_path: str = "data/datasets/financeqa",
                 output_dir: str = "results"):
        """
        Initialize benchmark runner.

        Args:
            dataset_path: Path to FinanceQA dataset directory
            output_dir: Directory to save evaluation results
        """
        self.evaluator = FinanceQAEvaluator(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_full_evaluation(self,
                          agent,
                          split: str = "test",
                          output_file: Optional[str] = None,
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Run evaluation on the full dataset.

        Args:
            agent: Agent to evaluate
            split: Dataset split to use ("test", "train", "validation")
            output_file: Custom output filename
            verbose: Show progress bars

        Returns:
            Evaluation results
        """
        print(f"Running FULL evaluation on {split} split...")

        # Load all examples
        examples = self.evaluator.load_dataset(split)
        print(f"Loaded {len(examples)} examples from {split} split")

        # Run evaluation on all examples
        results = self.evaluator.evaluate_agent(
            agent=agent,
            examples=examples,
            verbose=verbose
        )

        # Save results
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"financeqa_full_evaluation_{split}_{timestamp}.json"

        output_path = self.output_dir / output_file
        self.evaluator.save_results(results, str(output_path))

        # Print summary
        self._print_evaluation_summary(results, split, "FULL")

        return results

    def run_quick_evaluation(self,
                           agent,
                           split: str = "test",
                           num_samples: int = 50,
                           output_file: Optional[str] = None,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Run evaluation on a subset of the dataset for quick testing.

        Args:
            agent: Agent to evaluate
            split: Dataset split to use
            num_samples: Number of samples to evaluate
            output_file: Custom output filename
            verbose: Show progress bars

        Returns:
            Evaluation results
        """
        print(f"Running QUICK evaluation on {num_samples} samples from {split} split...")

        # Load examples
        examples = self.evaluator.load_dataset(split)
        print(f"Loaded {len(examples)} examples, using first {num_samples}")

        # Run evaluation on subset
        results = self.evaluator.evaluate_agent(
            agent=agent,
            examples=examples,
            max_examples=num_samples,
            verbose=verbose
        )

        # Save results
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"financeqa_quick_evaluation_{split}_{num_samples}_{timestamp}.json"

        output_path = self.output_dir / output_file
        self.evaluator.save_results(results, str(output_path))

        # Print summary
        self._print_evaluation_summary(results, split, f"QUICK ({num_samples} samples)")

        return results

    def run_by_question_type(self,
                           agent,
                           question_type: str,
                           split: str = "test",
                           output_file: Optional[str] = None,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Run evaluation filtered by specific question type.

        Args:
            agent: Agent to evaluate
            question_type: Filter by this question type
            split: Dataset split to use
            output_file: Custom output filename
            verbose: Show progress bars

        Returns:
            Evaluation results
        """
        print(f"Running evaluation on {question_type} questions from {split} split...")

        # Load and filter examples
        all_examples = self.evaluator.load_dataset(split)
        examples = [ex for ex in all_examples if ex.question_type == question_type]

        print(f"Found {len(examples)} examples of type '{question_type}' out of {len(all_examples)} total")

        if not examples:
            print(f"No examples found for question type: {question_type}")
            return {}

        # Run evaluation
        results = self.evaluator.evaluate_agent(
            agent=agent,
            examples=examples,
            verbose=verbose
        )

        # Save results
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"financeqa_by_type_{question_type}_{timestamp}.json"

        output_path = self.output_dir / output_file
        self.evaluator.save_results(results, str(output_path))

        # Print summary
        self._print_evaluation_summary(results, split, f"BY TYPE ({question_type})")

        return results

    def _print_evaluation_summary(self, results: Dict[str, Any], split: str, evaluation_type: str):
        """Print evaluation summary."""
        metrics = results['metrics']

        print(f"\n{evaluation_type} EVALUATION SUMMARY - {split.upper()} SPLIT")
        print("=" * 60)
        print(f"Total Examples: {metrics['total_examples']}")
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.1%}")
        print(f"Normalized Match Accuracy: {metrics['normalized_match_accuracy']:.1%}")
        print(f"Error Rate: {metrics['error_rate']:.1%}")
        print(f"Avg Processing Time: {results['average_processing_time']:.2f}s")
        print(f"Total Processing Time: {results['total_processing_time']:.1f}s")

        # Per question type breakdown
        if 'by_question_type' in metrics and metrics['by_question_type']:
            print("\nAccuracy by Question Type:")
            print("-" * 40)
            for qtype, type_metrics in metrics['by_question_type'].items():
                print(f"{qtype:20s}: {type_metrics['exact_match_accuracy']:.1%} "
                      f"({type_metrics['total_examples']} examples)")

def main():
    """Command-line interface for benchmark evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="FinanceQA Benchmark Evaluation")
    parser.add_argument("--dataset-path", default="data/datasets/financeqa",
                        help="Path to FinanceQA dataset directory")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--split", default="test", choices=["test", "train", "validation"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--evaluation-type", default="quick",
                        choices=["full", "quick", "by-type"],
                        help="Type of evaluation to run")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples for quick evaluation")
    parser.add_argument("--question-type",
                        help="Question type filter for by-type evaluation")
    parser.add_argument("--output-file",
                        help="Custom output filename")
    parser.add_argument("--no-verbose", action="store_true",
                        help="Disable progress bars")

    args = parser.parse_args()

    # Initialize benchmark runner
    runner = BenchmarkRunner(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )

    # Create FinanceQA agent
    try:
        from src.agent.core import Agent
        from src.models.model_manager import create_model_manager_from_config
        from src.tools.financial_calculator import FinancialCalculator
        from src.classifiers.question_classifier import QuestionClassifier
        import os
        import json

        # Try to load custom config if available
        if os.path.exists("config/model_config.json"):
            try:
                with open("config/model_config.json", 'r') as f:
                    config = json.load(f)
                print(f"✅ Loaded custom model configuration")
            except Exception:
                print("⚠️  Failed to load custom config")

        # Create model manager
        model_manager = create_model_manager_from_config(config)

        # Create tools
        financial_calc = FinancialCalculator()

        # Create agent
        agent = Agent(model_manager, tools=[financial_calc])

        # Add question classifier if possible
        try:
            classifier = QuestionClassifier()
            agent.question_classifier = classifier
            print("✅ Added question classifier")
        except Exception as e:
            print(f"⚠️  Question classifier not available: {e}")

        print(f"✅ Created FinanceQA agent with {config['default_provider']} model")

    except Exception as e:
        print(f"❌ Failed to create FinanceQA agent: {e}")
        print("Falling back to DummyAgent")
        agent = DummyAgent()

    verbose = not args.no_verbose

    try:
        if args.evaluation_type == "full":
            runner.run_full_evaluation(
                agent=agent,
                split=args.split,
                output_file=args.output_file,
                verbose=verbose
            )
        elif args.evaluation_type == "quick":
            runner.run_quick_evaluation(
                agent=agent,
                split=args.split,
                num_samples=args.num_samples,
                output_file=args.output_file,
                verbose=verbose
            )
        elif args.evaluation_type == "by-type":
            if not args.question_type:
                print("--question-type is required for by-type evaluation")
                return
            runner.run_by_question_type(
                agent=agent,
                question_type=args.question_type,
                split=args.split,
                output_file=args.output_file,
                verbose=verbose
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the FinanceQA dataset is downloaded and in the correct location.")
        return
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return

    print(f"\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()