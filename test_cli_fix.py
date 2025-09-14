#!/usr/bin/env python3
"""Test CLI functionality with the fixes."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model_manager import create_model_manager_from_config
from src.agent.core import Agent
from src.evaluation.benchmark_runner import BenchmarkRunner

class MockAgent:
    """Mock agent for testing."""

    async def answer_question(self, question: str):
        """Mock answer method."""
        from src.agent.core import ReasoningChain, ReasoningStep, StepType, StepStatus

        chain = ReasoningChain(question=question)
        step = ReasoningStep(
            step_type=StepType.ANALYSIS,
            description="Mock analysis",
            status=StepStatus.COMPLETED
        )
        step.output_data = {"mock": "response"}
        chain.add_step(step)
        chain.final_answer = "Mock answer for testing."
        return chain

def test_cli_runner():
    """Test that CLI runner works with the fixes."""
    print("Testing CLI runner with fixes...")

    # Create benchmark runner
    runner = BenchmarkRunner()

    # Create mock agent
    agent = MockAgent()

    try:
        # Test run_quick_evaluation (the one that was failing)
        print("Running quick evaluation...")
        results = runner.run_quick_evaluation(
            agent=agent,
            num_samples=2,  # Very small test
            verbose=False
        )

        print("✅ CLI quick evaluation completed!")
        print(f"   Results type: {type(results)}")
        print(f"   Has metrics: {'metrics' in results}")
        return True

    except Exception as e:
        print(f"❌ CLI runner failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cli_runner()
    if success:
        print("\n🎉 CLI FIX WORKING - Original error should be resolved!")
    else:
        print("\n❌ CLI still has issues")
    sys.exit(0 if success else 1)