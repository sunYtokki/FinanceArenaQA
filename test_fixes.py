#!/usr/bin/env python3
"""Test the fixes for infinite loop issues."""

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_fixed_agent():
    """Test the fixed agent with timeout controls."""

    # Load config
    config_path = Path("config/model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Create model manager
    model_manager = create_model_manager_from_config(config)

    # Create agent with max_steps limit
    agent = Agent(model_manager, max_steps=10)

    # Test problematic financial question
    question = "What is unadjusted EBITDA for the year ending in 2024?"
    logger.info(f"Testing fixed agent with: {question}")

    start_time = time.time()

    try:
        chain = await agent.answer_question(question)

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"✅ Agent completed in {duration:.2f} seconds")
        logger.info(f"Final answer: {chain.final_answer[:200]}...")
        logger.info(f"Number of steps: {len(chain.steps)}")

        # Check for early termination due to max steps
        if len(chain.steps) >= agent.max_steps:
            logger.warning("Agent hit max steps limit - this is expected behavior now")

        # Check for timeout errors
        timeout_steps = [s for s in chain.steps if s.error_message and "timeout" in s.error_message.lower()]
        if timeout_steps:
            logger.info(f"Found {len(timeout_steps)} timeout-related steps (handled gracefully)")

        return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"❌ Agent failed after {duration:.2f} seconds: {e}")
        return False

    finally:
        # Clean up sessions
        if hasattr(model_manager, 'close_all'):
            await model_manager.close_all()

async def test_multiple_concurrent():
    """Test multiple concurrent requests (simulating benchmark scenario)."""
    logger.info("\nTesting concurrent requests...")

    # Load config
    config_path = Path("config/model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Create model manager
    model_manager = create_model_manager_from_config(config)

    # Create multiple agents
    agents = [Agent(model_manager, max_steps=8) for _ in range(3)]

    questions = [
        "What is 2 + 2?",
        "What is gross profit?",
        "Calculate ROI for an investment."
    ]

    async def test_single_agent(agent, question, index):
        start_time = time.time()
        try:
            chain = await agent.answer_question(question)
            duration = time.time() - start_time
            logger.info(f"Agent {index} completed in {duration:.2f}s")
            return True
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Agent {index} failed after {duration:.2f}s: {e}")
            return False

    # Run concurrently
    start_time = time.time()
    tasks = [test_single_agent(agents[i], questions[i], i+1) for i in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_duration = time.time() - start_time

    success_count = sum(1 for r in results if r is True)
    logger.info(f"✅ Concurrent test: {success_count}/3 succeeded in {total_duration:.2f}s")

    # Clean up
    await model_manager.close_all()

    return success_count >= 2  # Allow one failure

async def main():
    """Main test function."""
    logger.info("=== Testing Infinite Loop Fixes ===")

    # Test 1: Fixed agent behavior
    logger.info("\n1. Testing fixed agent with timeout controls...")
    test1_success = await test_fixed_agent()

    # Test 2: Concurrent requests
    logger.info("\n2. Testing concurrent request handling...")
    test2_success = await test_multiple_concurrent()

    # Summary
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Fixed Agent Test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    logger.info(f"Concurrent Test: {'✅ PASS' if test2_success else '❌ FAIL'}")

    if test1_success and test2_success:
        logger.info("🎉 ALL TESTS PASSED - Fixes appear to be working!")
        return True
    else:
        logger.error("❌ Some tests failed - more investigation needed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)