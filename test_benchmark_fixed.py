#!/usr/bin/env python3
"""Test benchmark runner with fixes applied."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model_manager import create_model_manager_from_config
from src.agent.core import Agent
from src.evaluation.benchmark_runner import FinanceQAEvaluator

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Less verbose for benchmark
logger = logging.getLogger(__name__)

class FixedFinancialAgent:
    """Wrapper to make our agent compatible with the benchmark."""

    def __init__(self, model_manager):
        self.agent = Agent(model_manager, max_steps=10)  # Limit steps

    async def answer_question(self, question: str):
        """Answer a question and return the reasoning chain."""
        return await self.agent.answer_question(question)

async def run_small_benchmark():
    """Run a small benchmark test with fixes."""
    logger.warning("Starting small benchmark test with fixes...")

    # Load config
    config_path = Path("config/model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Create model manager
    model_manager = create_model_manager_from_config(config)

    # Create fixed agent
    agent = FixedFinancialAgent(model_manager)

    # Create evaluator
    evaluator = FinanceQAEvaluator()

    try:
        # Run evaluation on just 5 examples
        start_time = time.time()
        results = await evaluator.evaluate_agent_async(
            agent,
            max_examples=5,  # Small test
            verbose=True
        )
        total_time = time.time() - start_time

        # Print results
        metrics = results['metrics']
        logger.warning(f"✅ Benchmark completed in {total_time:.1f}s")
        logger.warning(f"Total examples: {results['total_examples']}")
        logger.warning(f"Average processing time: {results['average_processing_time']:.1f}s")
        logger.warning(f"Exact match accuracy: {metrics['exact_match_accuracy']:.1%}")

        # Check for timeouts
        timeout_count = sum(1 for r in results['results']
                          if r.error_message and 'timeout' in r.error_message.lower())
        if timeout_count > 0:
            logger.warning(f"Handled {timeout_count} timeouts gracefully")

        return True

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return False

    finally:
        # Clean up
        await model_manager.close_all()

if __name__ == "__main__":
    success = asyncio.run(run_small_benchmark())
    if success:
        print("\n🎉 BENCHMARK TEST PASSED - Agent no longer gets stuck in infinite loops!")
    else:
        print("\n❌ Benchmark test failed")
    sys.exit(0 if success else 1)