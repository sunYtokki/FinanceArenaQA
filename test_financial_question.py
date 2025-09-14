#!/usr/bin/env python3
"""Test with financial question to reproduce the infinite loop issue."""

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
logging.basicConfig(level=logging.WARNING)  # Less verbose
logger = logging.getLogger(__name__)

async def test_financial_question():
    """Test with the type of financial question that causes issues."""

    # Load config
    config_path = Path("config/model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Create model manager
    model_manager = create_model_manager_from_config(config)

    # Create agent
    agent = Agent(model_manager)

    # Test question from the evaluation results
    question = "What is unadjusted EBITDA for the year ending in 2024?"
    logger.warning(f"Testing financial question: {question}")

    start_time = time.time()
    timeout = 60  # 60 second timeout

    try:
        # Use asyncio.wait_for to enforce timeout
        chain = await asyncio.wait_for(
            agent.answer_question(question),
            timeout=timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.warning(f"Question completed in {duration:.2f} seconds")
        logger.warning(f"Final answer: {chain.final_answer[:200]}...")
        logger.warning(f"Number of steps: {len(chain.steps)}")

        # Print failed steps
        failed_steps = [s for s in chain.steps if s.status.value == "failed"]
        if failed_steps:
            logger.warning(f"Failed steps: {len(failed_steps)}")
            for step in failed_steps:
                logger.warning(f"  - {step.description}: {step.error_message}")

        return chain

    except asyncio.TimeoutError:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"TIMEOUT after {duration:.2f} seconds - Infinite loop confirmed!")

        # Get partial results
        if hasattr(agent, 'reasoning_chains') and agent.reasoning_chains:
            chain_id = list(agent.reasoning_chains.keys())[-1]
            partial_chain = agent.reasoning_chains[chain_id]
            logger.warning(f"Partial results - {len(partial_chain.steps)} steps completed:")

            # Count steps by status
            status_counts = {}
            for step in partial_chain.steps:
                status = step.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            logger.warning(f"Step status counts: {status_counts}")

            # Show in_progress steps (these are likely stuck)
            in_progress_steps = [s for s in partial_chain.steps if s.status.value == "in_progress"]
            if in_progress_steps:
                logger.error(f"STUCK STEPS ({len(in_progress_steps)}):")
                for step in in_progress_steps:
                    logger.error(f"  - {step.description}")

        return None

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"Agent failed after {duration:.2f} seconds: {e}")
        return None

async def main():
    """Main test function."""
    logger.warning("=== Testing Financial Question for Infinite Loop ===")

    result = await test_financial_question()

    if result is None:
        logger.error("CONFIRMED: Agent has infinite loop issues with financial questions!")
    else:
        logger.warning("Agent completed successfully - issue may be intermittent")

if __name__ == "__main__":
    asyncio.run(main())