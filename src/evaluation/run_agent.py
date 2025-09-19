#!/usr/bin/env python3
"""
FinanceQA Agent Result Generation Script

Standalone script for running agents on the FinanceQA benchmark to generate
result files. This script only calls agents and saves their responses -
no evaluation/scoring is performed here.
"""

import sys
import time
import asyncio
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures

import pandas as pd
from tqdm import tqdm

# Import shared utilities
from .shared_evaluation_utils import (
    EvaluationExample, AgentResponse, EvaluationResult, FinancialAgent,
    load_dataset, save_results,
    create_base_argument_parser, add_agent_evaluation_arguments,
    setup_model_manager, setup_agent, validate_evaluation_arguments
)


class AgentRunner:
    """Runner for executing agents on FinanceQA dataset and saving results."""

    def __init__(self, dataset_path: str = "data/datasets/financeqa"):
        """
        Initialize the agent runner.

        Args:
            dataset_path: Path to the FinanceQA dataset directory
        """
        self.dataset_path = Path(dataset_path)

    def _call_agent(self, agent: FinancialAgent, question: str):
        """
        Call agent.answer_question, handling both sync and async methods.

        Args:
            agent: Agent implementing the FinancialAgent protocol
            question: Question to ask the agent

        Returns:
            Result from agent.answer_question
        """
        # Check if the answer_question method is async
        if inspect.iscoroutinefunction(agent.answer_question):
            # Agent is async, run it in sync context
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, need to run in executor
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, agent.answer_question(question))
                        return future.result()
                else:
                    # No running loop, safe to use run
                    return loop.run_until_complete(agent.answer_question(question))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(agent.answer_question(question))
        else:
            # Agent is sync, call directly
            return agent.answer_question(question)

    def run_agent_on_dataset(
        self,
        agent: FinancialAgent,
        examples: List[EvaluationExample],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run agent on dataset examples and collect results.

        Args:
            agent: Agent implementing the FinancialAgent protocol
            examples: List of examples to process
            verbose: Whether to show progress bars

        Returns:
            Dictionary containing agent results
        """
        results = []
        total_processing_time = 0.0

        # Progress bar
        iterator = tqdm(examples, desc="Running Agent") if verbose else examples

        # Process each example
        for example in iterator:
            try:
                # Combine context and question
                full_question = f"Original Question: {example.question} \n Original Context: {example.context}\n" if example.context else example.question

                # Time the agent response
                start_time = time.time()
                reasoning_chain = self._call_agent(agent, full_question)
                processing_time = time.time() - start_time

                # Extract answer from reasoning chain
                answer = getattr(reasoning_chain, 'final_answer', '') or ''

                # Extract reasoning steps and confidence
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

                # Create result record (no scoring - just agent output)
                result = EvaluationResult(
                    example=example,
                    agent_response=agent_response,
                    processing_time=processing_time,
                    llm_match=None,  # Will be filled by LLM evaluation script
                    llm_reasoning=None
                )

            except Exception as e:
                # Handle agent errors gracefully
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

        # Simple summary metrics (no evaluation scores)
        errors = sum(1 for r in results if r.error_message is not None)
        metrics = {
            'total_examples': len(results),
            'errors': errors,
            'error_rate': errors / len(results) if results else 0.0,
            'agent_run_complete': True,
            'evaluation_pending': True
        }

        return {
            'results': results,
            'metrics': metrics,
            'total_examples': len(results),
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(results) if results else 0.0
        }

    def run_full(self,
                 agent: FinancialAgent,
                 split: str = "test",
                 output_file: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run agent on the full dataset.

        Args:
            agent: Agent to run
            split: Dataset split to use ("test", "train", "validation")
            output_file: Custom output filename
            verbose: Show progress bars

        Returns:
            Agent results
        """
        print(f"Running agent on FULL {split} split...")

        # Load all examples
        examples = load_dataset(str(self.dataset_path), split)
        print(f"Loaded {len(examples)} examples from {split} split")

        # Run agent on all examples
        results = self.run_agent_on_dataset(
            agent=agent,
            examples=examples,
            verbose=verbose
        )

        # Generate output filename if not provided
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"financeqa_agent_results_{split}_full_{timestamp}.json"

        # Save results
        output_path = Path("results") / output_file
        output_path.parent.mkdir(exist_ok=True)
        save_results(results, str(output_path))

        # Print summary
        self._print_summary(results, split, "FULL")

        return results

    def run_quick(self,
                  agent: FinancialAgent,
                  split: str = "test",
                  num_samples: int = 50,
                  output_file: Optional[str] = None,
                  verbose: bool = True) -> Dict[str, Any]:
        """
        Run agent on a subset of the dataset for quick testing.

        Args:
            agent: Agent to run
            split: Dataset split to use
            num_samples: Number of samples to process
            output_file: Custom output filename
            verbose: Show progress bars

        Returns:
            Agent results
        """
        print(f"Running agent on {num_samples} samples from {split} split...")

        # Load examples
        examples = load_dataset(str(self.dataset_path), split)
        examples = examples[:num_samples]
        print(f"Processing {len(examples)} examples")

        # Run agent on subset
        results = self.run_agent_on_dataset(
            agent=agent,
            examples=examples,
            verbose=verbose
        )

        # Generate output filename if not provided
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"financeqa_agent_results_{split}_{num_samples}_{timestamp}.json"

        # Save results
        output_path = Path("results") / output_file
        output_path.parent.mkdir(exist_ok=True)
        save_results(results, str(output_path))

        # Print summary
        self._print_summary(results, split, f"QUICK ({num_samples} samples)")

        return results

    def run_by_type(self,
                    agent: FinancialAgent,
                    question_type: str,
                    split: str = "test",
                    output_file: Optional[str] = None,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Run agent on examples filtered by question type.

        Args:
            agent: Agent to run
            question_type: Filter by this question type
            split: Dataset split to use
            output_file: Custom output filename
            verbose: Show progress bars

        Returns:
            Agent results
        """
        print(f"Running agent on {question_type} questions from {split} split...")

        # Load and filter examples
        all_examples = load_dataset(str(self.dataset_path), split)
        examples = [ex for ex in all_examples if ex.question_type == question_type]

        print(f"Found {len(examples)} examples of type '{question_type}' out of {len(all_examples)} total")

        if not examples:
            print(f"No examples found for question type: {question_type}")
            return {}

        # Run agent
        results = self.run_agent_on_dataset(
            agent=agent,
            examples=examples,
            verbose=verbose
        )

        # Generate output filename if not provided
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"financeqa_agent_results_{question_type}_{timestamp}.json"

        # Save results
        output_path = Path("results") / output_file
        output_path.parent.mkdir(exist_ok=True)
        save_results(results, str(output_path))

        # Print summary
        self._print_summary(results, split, f"BY TYPE ({question_type})")

        return results

    def run_evaluation(self,
                      agent: FinancialAgent,
                      question_type: Optional[str] = None,
                      num_samples: Optional[int] = None,
                      split: str = "test",
                      output_file: Optional[str] = None,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Run agent evaluation with modern interface parameters.

        Args:
            agent: Agent to run
            question_type: Filter by question type
            num_samples: Limit to N samples
            split: Dataset split to use
            output_file: Custom output filename
            verbose: Show progress bars

        Returns:
            Agent results
        """
        if question_type and num_samples:
            # Combined filtering: by type + sample limit
            print(f"Running agent on {num_samples} {question_type} questions from {split} split...")

            # Load and filter examples
            all_examples = load_dataset(str(self.dataset_path), split)
            examples = [ex for ex in all_examples if ex.question_type == question_type]

            if not examples:
                print(f"No examples found for question type: {question_type}")
                return {}

            # Apply sample limit
            examples = examples[:num_samples]
            print(f"Processing {len(examples)} examples")

            # Run agent
            results = self.run_agent_on_dataset(
                agent=agent,
                examples=examples,
                verbose=verbose
            )

            # Generate output filename
            if output_file is None:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"financeqa_agent_results_{question_type}_{num_samples}_{timestamp}.json"

            # Save results
            output_path = Path("results") / output_file
            output_path.parent.mkdir(exist_ok=True)
            save_results(results, str(output_path))

            # Print summary
            self._print_summary(results, split, f"{question_type.upper()} ({num_samples} samples)")

        elif question_type:
            # Filter by question type only
            results = self.run_by_type(
                agent=agent,
                question_type=question_type,
                split=split,
                output_file=output_file,
                verbose=verbose
            )

        elif num_samples:
            # Limit samples only
            results = self.run_quick(
                agent=agent,
                split=split,
                num_samples=num_samples,
                output_file=output_file,
                verbose=verbose
            )

        else:
            # Full dataset
            results = self.run_full(
                agent=agent,
                split=split,
                output_file=output_file,
                verbose=verbose
            )

        return results

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


async def main():
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

        async with model_manager as mm:
            print("🤖 Setting up FinanceQA agent...")
            agent = setup_agent(mm, config)
            print("✅ FinanceQA agent setup complete")

            # Initialize runner
            runner = AgentRunner(dataset_path=args.dataset_path)
            verbose = not args.no_verbose

            try:
                # Use modern unified evaluation interface
                runner.run_evaluation(
                    agent=agent,
                    question_type=getattr(args, 'question_type', None),
                    num_samples=getattr(args, 'num_samples', None),
                    split=args.split,
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
    asyncio.run(main())