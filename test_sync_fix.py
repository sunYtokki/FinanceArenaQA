#!/usr/bin/env python3
"""Test the sync/async fix for benchmark runner."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model_manager import create_model_manager_from_config
from src.agent.core import Agent
from src.evaluation.benchmark_runner import FinanceQAEvaluator

class TestAgent:
    """Simple test agent for verification."""

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
        chain.final_answer = "Mock answer: Unable to determine from provided context."
        return chain

def test_sync_call():
    """Test that synchronous calls work."""
    print("Testing synchronous evaluate_agent call...")

    evaluator = FinanceQAEvaluator()
    agent = TestAgent()

    try:
        # This should work now with the sync wrapper
        results = evaluator.evaluate_agent(
            agent,
            max_examples=2,
            verbose=False
        )

        print(f"✅ Sync call succeeded!")
        print(f"   Total examples: {results['total_examples']}")
        print(f"   Average time: {results['average_processing_time']:.2f}s")
        return True

    except Exception as e:
        print(f"❌ Sync call failed: {e}")
        return False

if __name__ == "__main__":
    success = test_sync_call()
    if success:
        print("\n🎉 SYNC FIX WORKING - CLI should work now!")
    else:
        print("\n❌ Still has issues")
    sys.exit(0 if success else 1)