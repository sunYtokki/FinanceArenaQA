#!/usr/bin/env python3
"""
FinanceQA Benchmark Runner (Orchestrator)

This module orchestrates the complete FinanceQA evaluation workflow by coordinating
the agent evaluation and LLM evaluation components. It maintains backward compatibility
with the original benchmark runner while enabling separated execution.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Import evaluation components
from .run_agent import (
    AgentRunner, create_base_argument_parser,
    setup_model_manager, setup_agent, validate_evaluation_arguments,
    add_agent_evaluation_arguments
)
from .run_llm_evaluation import LLMEvaluator, setup_llm_scorer


def add_orchestrator_arguments(parser):
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


class BenchmarkOrchestrator:
    """Orchestrator for the complete FinanceQA evaluation workflow."""

    def __init__(self,
                 dataset_path: str = "data/datasets/financeqa",
                 output_dir: str = "results",
                 use_llm_evaluation: bool = False,
                 evaluation_model: Optional[str] = None,
                 model_manager=None,
                 config=None):
        """
        Initialize the benchmark orchestrator.

        Args:
            dataset_path: Path to FinanceQA dataset directory
            output_dir: Directory to save evaluation results
            use_llm_evaluation: Whether to run LLM evaluation after agent evaluation
            evaluation_model: Model to use for LLM evaluation (if different from agent model)
            model_manager: Optional pre-initialized model manager to reuse
            config: Optional pre-loaded config to reuse
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_llm_evaluation = use_llm_evaluation
        self.evaluation_model = evaluation_model
        self.model_manager = model_manager
        self.config = config

    def run_evaluation(self,
                       agent,
                       question_type: Optional[str] = None,
                       num_samples: Optional[int] = None,
                       split: str = "test",
                       output_file: Optional[str] = None,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Run evaluation workflow (agent + optional LLM evaluation).

        Args:
            agent: Agent to evaluate
            question_type: Filter by question type (optional)
            num_samples: Limit to N samples (optional)
            split: Dataset split to use
            output_file: Custom output filename (optional)
            verbose: Show progress bars

        Returns:
            Complete evaluation results
        """
        # Generate evaluation type for logging and filenames
        if question_type and num_samples:
            evaluation_type = f"{question_type.upper()} ({num_samples} samples)"
            filename_suffix = f"{question_type}_{num_samples}"
        elif question_type:
            evaluation_type = f"BY TYPE ({question_type.upper()})"
            filename_suffix = f"by_type_{question_type}"
        elif num_samples:
            evaluation_type = f"QUICK ({num_samples} samples)"
            filename_suffix = f"quick_{num_samples}"
        else:
            evaluation_type = "FULL"
            filename_suffix = "full"

        print(f"🚀 Starting {evaluation_type} evaluation workflow from {split} split...")

        # Step 1: Run agent evaluation
        print("\n📋 STEP 1: Running Agent Evaluation")
        print("-" * 50)

        agent_runner = AgentRunner(self.dataset_path)

        # Generate temporary filename for agent results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        temp_agent_file = f"temp_agent_results_{filename_suffix}_{timestamp}.json"

        agent_results = agent_runner.run_agent_on_dataset(
            agent=agent,
            question_type=question_type,
            num_samples=num_samples,
            split=split,
            output_file=temp_agent_file,
            verbose=verbose
        )

        if not agent_results:
            print(f"No examples found for {evaluation_type} evaluation")
            return {}

        # Step 2: Run LLM evaluation if requested
        final_results = agent_results
        if self.use_llm_evaluation:
            print("\n🧠 STEP 2: Running LLM Evaluation")
            print("-" * 50)

            final_results = self._run_llm_evaluation_step(
                agent_result_file=str(self.output_dir / temp_agent_file),
                verbose=verbose
            )

            # Clean up temporary agent file
            temp_file_path = self.output_dir / temp_agent_file
            if temp_file_path.exists():
                temp_file_path.unlink()
        else:
            print("\n⏭️  STEP 2: Skipping LLM Evaluation (--use-llm-evaluation not specified)")

        # Step 3: Save final results
        if output_file is None:
            eval_type = "llm_evaluated" if self.use_llm_evaluation else "agent_only"
            output_file = f"financeqa_{filename_suffix}_{eval_type}_{timestamp}.json"

        final_output_path = self.output_dir / output_file

        # Save results
        if self.use_llm_evaluation:
            model_manager, config = self._get_model_manager_and_config()
            evaluator = LLMEvaluator(model_manager, self.evaluation_model, config)
            evaluator.save_evaluated_results(final_results, str(final_output_path))
        else:
            from .run_agent import save_results
            save_results(final_results, str(final_output_path))

        # Print final summary
        self._print_final_summary(final_results, split, evaluation_type, self.use_llm_evaluation)

        return final_results

    def _get_model_manager_and_config(self):
        """Get model manager and config, using stored instances if available."""
        if self.model_manager and self.config:
            return self.model_manager, self.config
        else:
            return setup_model_manager()

    def _run_llm_evaluation_step(self, agent_result_file: str, verbose: bool) -> Dict[str, Any]:
        """
        Run LLM evaluation on agent results.

        Args:
            agent_result_file: Path to agent results file
            verbose: Show progress bars

        Returns:
            LLM evaluation results
        """
        try:
            # Get model manager and config
            model_manager, config = self._get_model_manager_and_config()

            # Use evaluation-specific model if configured
            eval_model = self.evaluation_model
            if not eval_model and 'evaluation' in config:
                eval_model = config['evaluation'].get('model_name')

            evaluator = LLMEvaluator(model_manager, eval_model, config)

            # Run LLM evaluation
            llm_results = evaluator.evaluate_result_file(
                result_file_path=agent_result_file,
                failed_examples="process",  # Process all examples
                verbose=verbose
            )

            return llm_results

        except Exception as e:
            print(f"❌ LLM evaluation failed: {e}")
            print("📋 Returning agent-only results")

            # Fallback to agent-only results
            from .run_llm_evaluation import load_results
            return load_results(agent_result_file)

    def _print_final_summary(self, results: Dict[str, Any], split: str, evaluation_type: str, used_llm: bool):
        """Print final evaluation summary."""
        metrics = results['metrics']

        eval_mode = "LLM EVALUATION" if used_llm else "AGENT EVALUATION"
        print(f"\n🎯 FINAL {eval_mode} SUMMARY - {split.upper()} SPLIT - {evaluation_type}")
        print("=" * 60)
        print(f"Total Examples: {metrics['total_examples']}")

        if used_llm and 'llm_match_accuracy' in metrics:
            print(f"LLM Match Accuracy: {metrics['llm_match_accuracy']:.1%}")
            print(f"LLM Matches: {metrics['llm_matches']}")

        print(f"Error Rate: {metrics['error_rate']:.1%}")
        print(f"Total Processing Time: {results.get('total_processing_time', 0):.1f}s")

        # Per question type breakdown
        if 'by_question_type' in metrics and metrics['by_question_type']:
            accuracy_label = "LLM Accuracy" if used_llm else "Success Rate"
            print(f"\n{accuracy_label} by Question Type:")
            print("-" * 50)
            for qtype, type_metrics in metrics['by_question_type'].items():
                if used_llm and 'llm_match_accuracy' in type_metrics:
                    accuracy = type_metrics['llm_match_accuracy']
                else:
                    accuracy = 1.0 - type_metrics['error_rate']  # Success rate
                print(f"{qtype:20s}: {accuracy:.1%} ({type_metrics['total_examples']} examples)")


def main():
    """Command-line interface for benchmark orchestration."""
    # Create argument parser with orchestrator arguments
    parser = create_base_argument_parser("FinanceQA Benchmark Evaluation")
    parser = add_orchestrator_arguments(parser)

    args = parser.parse_args()

    try:
        # Validate arguments
        validate_evaluation_arguments(args)

        # Setup model manager and agent
        print("🔧 Setting up model manager...")
        model_manager, config = setup_model_manager()
        print(f"✅ Model manager setup complete using {config['default_provider']}")

        print("🤖 Setting up FinanceQA agent...")
        agent = setup_agent(model_manager, config)
        print("✅ FinanceQA agent setup complete")

        # Validate LLM evaluation setup if requested
        if args.use_llm_evaluation:
            try:
                setup_llm_scorer(model_manager, args.evaluation_model, config)
                print("✅ LLM evaluation setup validated")
            except Exception as e:
                print(f"⚠️  LLM evaluation setup failed: {e}")
                print("Continuing with agent-only evaluation")
                args.use_llm_evaluation = False

    except Exception as e:
        print(f"❌ Failed to setup components: {e}")
        sys.exit(1)

    # Initialize orchestrator with model manager for reuse
    orchestrator = BenchmarkOrchestrator(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        use_llm_evaluation=args.use_llm_evaluation,
        evaluation_model=args.evaluation_model,
        model_manager=model_manager,
        config=config
    )

    verbose = not args.no_verbose

    try:
        # Use modern unified evaluation interface
        orchestrator.run_evaluation(
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
        print(f"Evaluation failed: {e}")
        sys.exit(1)
    finally:
        # Clean up agent resources
        if hasattr(agent, 'cleanup'):
            try:
                agent.cleanup()
                print("🧹 Agent cleanup completed")
            except Exception as e:
                print(f"⚠️ Warning: Agent cleanup error: {e}")

    print(f"\n✅ Benchmark evaluation completed successfully!")
    if args.use_llm_evaluation:
        print(f"📊 Complete LLM-based evaluation results saved")
    else:
        print(f"📋 Agent results saved - use --use-llm-evaluation for complete evaluation")


if __name__ == "__main__":
    main()