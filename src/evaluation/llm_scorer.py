"""
LLM-based scoring system for FinanceQA evaluation.

This module provides an LLM-based scorer that replaces exact match scoring
with intelligent evaluation using a language model to assess financial
answer correctness.
"""

import json
import time
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.models.model_manager import ModelManager


@dataclass
class LLMEvaluationResult:
    """Result of LLM evaluation for a single answer pair."""
    is_correct: bool
    reasoning: str
    confidence: float
    processing_time: float
    error_message: Optional[str] = None


class LLMScorer:
    """LLM-based scorer for financial QA evaluation."""

    def __init__(self, model_manager: ModelManager, model_name: Optional[str] = None):
        """
        Initialize the LLM scorer.

        Args:
            model_manager: Model manager for LLM inference
            model_name: Specific model to use for evaluation (if None, uses default)
        """
        self.model_manager = model_manager
        self.model_name = model_name

        # Financial evaluation prompt template
        self.evaluation_prompt = """You are an expert financial analyst evaluating the correctness of financial answers.

Your task is to determine if a predicted answer matches the ground truth answer in the context of financial data.

Consider these factors when evaluating:
1. Numerical accuracy (exact values, proper units, reasonable approximations)
2. Financial terminology (consistent use of financial concepts)
3. Contextual correctness (answer fits the question context)
4. Format equivalence (different valid representations of same value)

Examples of CORRECT matches:
- Predicted: "$1,000 million", Ground Truth: "$1 billion" (equivalent values)
- Predicted: "32.5%", Ground Truth: "32.5 percent" (same value, different format)
- Predicted: "Decreased by 15%", Ground Truth: "15% decline" (same meaning)

Examples of INCORRECT matches:
- Predicted: "$500 million", Ground Truth: "$1 billion" (different values)
- Predicted: "Increased", Ground Truth: "Decreased" (opposite meaning)
- Predicted: "Q3 2023", Ground Truth: "Q4 2023" (different time periods)

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Evaluate whether the predicted answer is correct compared to the ground truth.
Provide your response in the following JSON format:

{{
  "is_correct": true/false,
  "reasoning": "Detailed explanation of your evaluation decision",
  "confidence": 0.95
}}

Response:"""

    def compute_llm_match(self, predicted: str, ground_truth: str, question: str = "") -> bool:
        """
        Compute LLM-based match between predicted and ground truth answers.

        Args:
            predicted: Agent's predicted answer
            ground_truth: Ground truth answer
            question: Original question for context (optional)

        Returns:
            True if LLM considers answers equivalent, False otherwise
        """
        try:
            result = self.evaluate_answer_pair(predicted, ground_truth, question)
            return result.is_correct
        except Exception:
            # Fallback to exact match on LLM failure
            return predicted.strip().lower() == ground_truth.strip().lower()

    def evaluate_answer_pair(self, predicted: str, ground_truth: str, question: str = "") -> LLMEvaluationResult:
        """
        Evaluate a single predicted vs ground truth answer pair using LLM.

        Args:
            predicted: Agent's predicted answer
            ground_truth: Ground truth answer
            question: Original question for context

        Returns:
            LLMEvaluationResult with evaluation details
        """
        try:
            # Handle event loop more robustly
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Create new event loop if closed
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run the async function
                if loop.is_running():
                    # If loop is running, we need to use run_until_complete in a thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._evaluate_answer_pair_async(predicted, ground_truth, question)
                        )
                        return future.result()
                else:
                    # Loop exists but not running, safe to use
                    return loop.run_until_complete(
                        self._evaluate_answer_pair_async(predicted, ground_truth, question)
                    )
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self._evaluate_answer_pair_async(predicted, ground_truth, question))

        except Exception as e:
            # Return conservative result on error
            return LLMEvaluationResult(
                is_correct=False,
                reasoning=f"Evaluation failed: {str(e)}",
                confidence=0.0,
                processing_time=0.0,
                error_message=str(e)
            )

    async def _evaluate_answer_pair_async(self, predicted: str, ground_truth: str, question: str = "") -> LLMEvaluationResult:
        """
        Async implementation of evaluate_answer_pair.

        Args:
            predicted: Agent's predicted answer
            ground_truth: Ground truth answer
            question: Original question for context

        Returns:
            LLMEvaluationResult with evaluation details
        """
        start_time = time.time()

        try:
            # Format the evaluation prompt
            prompt = self.evaluation_prompt.format(
                question=question.strip(),
                ground_truth=ground_truth.strip(),
                predicted=predicted.strip()
            )

            # Get LLM response using the correct async method
            response = await self.model_manager.generate(
                prompt=prompt,
                provider_name=None,  # Use default provider
                temperature=0.1,
                max_tokens=512
            )

            processing_time = time.time() - start_time

            # Parse JSON response from response.content
            evaluation_data = self._parse_llm_response(response.content)

            return LLMEvaluationResult(
                is_correct=evaluation_data["is_correct"],
                reasoning=evaluation_data["reasoning"],
                confidence=evaluation_data.get("confidence", 0.5),
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time

            # Return conservative result on error
            return LLMEvaluationResult(
                is_correct=False,
                reasoning=f"Evaluation failed: {str(e)}",
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )

    def evaluate_batch(self, answer_pairs: List[Dict[str, str]],
                       max_retries: int = 3,
                       retry_delay: float = 1.0) -> List[LLMEvaluationResult]:
        """
        Evaluate multiple answer pairs efficiently with retry logic.

        Args:
            answer_pairs: List of dicts with 'predicted', 'ground_truth', and optional 'question' keys
            max_retries: Maximum number of retries for failed evaluations
            retry_delay: Base delay between retries (exponential backoff)

        Returns:
            List of LLMEvaluationResult objects
        """
        results = []

        for i, pair in enumerate(answer_pairs):
            result = self._evaluate_with_retry(
                predicted=pair["predicted"],
                ground_truth=pair["ground_truth"],
                question=pair.get("question", ""),
                max_retries=max_retries,
                retry_delay=retry_delay,
                pair_index=i
            )
            results.append(result)

        return results

    def _evaluate_with_retry(self, predicted: str, ground_truth: str, question: str = "",
                           max_retries: int = 3, retry_delay: float = 1.0,
                           pair_index: int = 0) -> LLMEvaluationResult:
        """
        Evaluate a single answer pair with exponential backoff retry.

        Args:
            predicted: Agent's predicted answer
            ground_truth: Ground truth answer
            question: Original question for context
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            pair_index: Index of this pair for logging

        Returns:
            LLMEvaluationResult with evaluation details
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = self.evaluate_answer_pair(predicted, ground_truth, question)

                # If we had previous failures but this succeeded, note it
                if attempt > 0:
                    result.reasoning += f" [Succeeded on attempt {attempt + 1}]"

                return result

            except Exception as e:
                last_error = e

                if attempt < max_retries:
                    # Exponential backoff
                    delay = retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed, return conservative result
                    return LLMEvaluationResult(
                        is_correct=False,
                        reasoning=f"LLM evaluation failed after {max_retries + 1} attempts: {str(last_error)}",
                        confidence=0.0,
                        processing_time=0.0,
                        error_message=f"Pair {pair_index}: {str(last_error)}"
                    )

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract evaluation data.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed evaluation data

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Try to find JSON in the response
            response = response.strip()

            # Look for JSON block
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Validate required fields
            if "is_correct" not in data:
                raise ValueError("Missing 'is_correct' field")
            if "reasoning" not in data:
                raise ValueError("Missing 'reasoning' field")

            # Ensure proper types
            data["is_correct"] = bool(data["is_correct"])
            data["reasoning"] = str(data["reasoning"])
            data["confidence"] = float(data.get("confidence", 0.5))

            return data

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def compute_exact_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Backward compatibility method for exact match computation.

        Args:
            predicted: Agent's predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if exact match, False otherwise
        """
        return predicted.strip() == ground_truth.strip()

    def compute_normalized_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Backward compatibility method for normalized match computation.
        Uses LLM evaluation by default.

        Args:
            predicted: Agent's predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if normalized match, False otherwise
        """
        return self.compute_llm_match(predicted, ground_truth)