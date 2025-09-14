#!/usr/bin/env python3
"""Debug script to test agent behavior and identify infinite loop issues."""

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

async def test_simple_question():
    """Test a simple question with timeout to detect infinite loops."""

    # Load config
    config_path = Path("config/model_config.json")
    if not config_path.exists():
        logger.error("Config file not found")
        return

    with open(config_path) as f:
        config = json.load(f)

    # Create model manager
    model_manager = create_model_manager_from_config(config)

    # Create agent
    agent = Agent(model_manager)

    # Test question
    question = "What is 2 + 2?"
    logger.info(f"Testing simple question: {question}")

    start_time = time.time()
    timeout = 30  # 30 second timeout

    try:
        # Use asyncio.wait_for to enforce timeout
        chain = await asyncio.wait_for(
            agent.answer_question(question),
            timeout=timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Question completed in {duration:.2f} seconds")
        logger.info(f"Final answer: {chain.final_answer}")
        logger.info(f"Number of steps: {len(chain.steps)}")

        # Print step details
        for i, step in enumerate(chain.steps):
            logger.info(f"Step {i+1}: {step.description} ({step.status.value})")
            if step.error_message:
                logger.warning(f"  Error: {step.error_message}")

        return chain

    except asyncio.TimeoutError:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"TIMEOUT after {duration:.2f} seconds - Agent appears to be in infinite loop!")

        # Try to get partial results
        if hasattr(agent, 'reasoning_chains') and agent.reasoning_chains:
            chain_id = list(agent.reasoning_chains.keys())[-1]
            partial_chain = agent.reasoning_chains[chain_id]
            logger.info(f"Partial results - {len(partial_chain.steps)} steps completed:")

            for i, step in enumerate(partial_chain.steps):
                logger.info(f"  Step {i+1}: {step.description} ({step.status.value})")
                if step.error_message:
                    logger.warning(f"    Error: {step.error_message}")

        return None

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"Agent failed after {duration:.2f} seconds: {e}")
        return None

async def test_ollama_direct():
    """Test Ollama provider directly to isolate issues."""
    logger.info("Testing Ollama provider directly...")

    config_path = Path("config/model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    model_manager = create_model_manager_from_config(config)

    try:
        response = await model_manager.generate("What is 2 + 2? Answer briefly.")
        logger.info(f"Direct Ollama response: {response.content[:200]}...")
        return True
    except Exception as e:
        logger.error(f"Direct Ollama test failed: {e}")
        return False

async def main():
    """Main debug function."""
    logger.info("=== FinanceQA Agent Debug Session ===")

    # Test 1: Direct Ollama communication
    logger.info("\n1. Testing direct Ollama communication...")
    ollama_works = await test_ollama_direct()

    if not ollama_works:
        logger.error("Ollama communication failed - fix this first!")
        return

    logger.info("✓ Ollama communication works")

    # Test 2: Simple agent question with timeout
    logger.info("\n2. Testing simple agent question with timeout...")
    result = await test_simple_question()

    if result is None:
        logger.error("Agent has timeout/infinite loop issues!")
    else:
        logger.info("✓ Agent completed successfully")

if __name__ == "__main__":
    asyncio.run(main())