"""
Unit tests for the FinanceQA benchmark evaluation harness.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.evaluation.benchmark_runner import (
    EvaluationExample,
    AgentResponse,
    EvaluationResult,
    ExactMatchScorer,
    FinanceQAEvaluator,
    DummyAgent
)


class TestEvaluationExample:
    """Test the EvaluationExample dataclass."""

    def test_from_dict(self):
        """Test creating EvaluationExample from dictionary."""
        data = {
            'context': 'Financial context',
            'question': 'What is revenue?',
            'chain_of_thought': 'Step 1: Find revenue',
            'answer': '$100',
            'file_link': 'http://example.com',
            'file_name': 'file.pdf',
            'company': 'TestCorp',
            'question_type': 'basic'
        }

        example = EvaluationExample.from_dict(data)

        assert example.context == 'Financial context'
        assert example.question == 'What is revenue?'
        assert example.answer == '$100'
        assert example.company == 'TestCorp'
        assert example.question_type == 'basic'


class TestExactMatchScorer:
    """Test the ExactMatchScorer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = ExactMatchScorer()

    def test_normalize_financial_answer(self):
        """Test financial answer normalization."""
        # Test basic normalization
        assert self.scorer.normalize_financial_answer("$1,000") == "1000"
        assert self.scorer.normalize_financial_answer("$1,000 million") == "1000 million"

        # Test case insensitive
        assert self.scorer.normalize_financial_answer("$1,000 Million") == "1000 million"

        # Test whitespace normalization
        assert self.scorer.normalize_financial_answer("  $1,000   million  ") == "1000 million"

        # Test abbreviations
        assert self.scorer.normalize_financial_answer("$1,000 mln") == "1000 million"

    def test_extract_numeric_value(self):
        """Test numeric value extraction."""
        assert self.scorer.extract_numeric_value("$1,000") == 1000.0
        assert self.scorer.extract_numeric_value("$1,500.50") == 1500.5
        assert self.scorer.extract_numeric_value("-$500") == -500.0
        assert self.scorer.extract_numeric_value("No numbers here") is None
        assert self.scorer.extract_numeric_value("") is None

    def test_compute_exact_match(self):
        """Test exact match computation."""
        assert self.scorer.compute_exact_match("$1,000", "$1,000") is True
        assert self.scorer.compute_exact_match("$1,000", "$1,001") is False
        assert self.scorer.compute_exact_match("  $1,000  ", "$1,000") is True  # Strips whitespace

    def test_compute_normalized_match(self):
        """Test normalized match computation."""
        assert self.scorer.compute_normalized_match("$1,000", "$1,000") is True
        assert self.scorer.compute_normalized_match("$1,000", "$1,001") is False
        assert self.scorer.compute_normalized_match("$1,000 million", "$1,000 mln") is True
        assert self.scorer.compute_normalized_match("$1,000 Million", "$1,000 million") is True


class TestFinanceQAEvaluator:
    """Test the FinanceQAEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.evaluator = FinanceQAEvaluator(self.temp_dir)

    def create_test_dataset(self, examples_data):
        """Create a test dataset file."""
        test_file = Path(self.temp_dir) / "test.jsonl"
        with open(test_file, 'w', encoding='utf-8') as f:
            for example in examples_data:
                json.dump(example, f)
                f.write('\n')

    def test_load_dataset(self):
        """Test dataset loading."""
        # Create test data
        examples_data = [
            {
                'context': 'Context 1',
                'question': 'Question 1?',
                'chain_of_thought': 'Step 1',
                'answer': '$100',
                'file_link': 'http://example.com/1',
                'file_name': 'file1.pdf',
                'company': 'Corp1',
                'question_type': 'basic'
            },
            {
                'context': 'Context 2',
                'question': 'Question 2?',
                'chain_of_thought': 'Step 2',
                'answer': '$200',
                'file_link': 'http://example.com/2',
                'file_name': 'file2.pdf',
                'company': 'Corp2',
                'question_type': 'advanced'
            }
        ]

        self.create_test_dataset(examples_data)

        # Load dataset
        examples = self.evaluator.load_dataset("test")

        assert len(examples) == 2
        assert examples[0].question == 'Question 1?'
        assert examples[0].answer == '$100'
        assert examples[1].company == 'Corp2'

    def test_load_dataset_file_not_found(self):
        """Test loading non-existent dataset."""
        with pytest.raises(FileNotFoundError):
            self.evaluator.load_dataset("nonexistent")

    def test_compute_metrics(self):
        """Test metrics computation."""
        # Create mock results
        examples = [
            EvaluationExample(
                context="ctx", question="q1", chain_of_thought="", answer="$100",
                file_link="", file_name="", company="Corp1", question_type="basic"
            ),
            EvaluationExample(
                context="ctx", question="q2", chain_of_thought="", answer="$200",
                file_link="", file_name="", company="Corp2", question_type="basic"
            ),
            EvaluationExample(
                context="ctx", question="q3", chain_of_thought="", answer="$300",
                file_link="", file_name="", company="Corp3", question_type="advanced"
            ),
        ]

        results = [
            EvaluationResult(
                example=examples[0],
                agent_response=AgentResponse(answer="$100"),
                exact_match=True,
                normalized_match=True,
                processing_time=1.0
            ),
            EvaluationResult(
                example=examples[1],
                agent_response=AgentResponse(answer="$201"),
                exact_match=False,
                normalized_match=False,
                processing_time=1.5
            ),
            EvaluationResult(
                example=examples[2],
                agent_response=AgentResponse(answer="$300"),
                exact_match=True,
                normalized_match=True,
                processing_time=2.0,
                error_message="Test error"
            ),
        ]

        metrics = self.evaluator.compute_metrics(results)

        assert metrics['exact_match_accuracy'] == 2/3  # 2 out of 3 correct
        assert metrics['normalized_match_accuracy'] == 2/3
        assert metrics['error_rate'] == 1/3  # 1 error
        assert metrics['total_examples'] == 3

        # Check by question type
        assert 'basic' in metrics['by_question_type']
        assert 'advanced' in metrics['by_question_type']
        assert metrics['by_question_type']['basic']['exact_match_accuracy'] == 0.5  # 1/2
        assert metrics['by_question_type']['advanced']['exact_match_accuracy'] == 1.0  # 1/1

    def test_evaluate_agent_with_mock(self):
        """Test agent evaluation with mock agent."""
        # Create test data
        examples_data = [
            {
                'context': 'Revenue is $100',
                'question': 'What is revenue?',
                'chain_of_thought': 'Look for revenue',
                'answer': '$100',
                'file_link': 'http://example.com',
                'file_name': 'file.pdf',
                'company': 'TestCorp',
                'question_type': 'basic'
            }
        ]

        self.create_test_dataset(examples_data)

        # Create mock agent
        mock_agent = Mock()
        mock_agent.answer_question.return_value = AgentResponse(answer="$100")

        # Evaluate
        results = self.evaluator.evaluate_agent(mock_agent, verbose=False)

        # Verify
        assert results['total_examples'] == 1
        assert results['metrics']['exact_match_accuracy'] == 1.0
        assert mock_agent.answer_question.called
        assert len(results['results']) == 1

    def test_save_results(self):
        """Test saving evaluation results."""
        # Create mock evaluation output
        example = EvaluationExample(
            context="ctx", question="q", chain_of_thought="", answer="$100",
            file_link="", file_name="", company="Corp", question_type="basic"
        )

        result = EvaluationResult(
            example=example,
            agent_response=AgentResponse(answer="$100", reasoning_steps=["step1"]),
            exact_match=True,
            normalized_match=True,
            processing_time=1.0
        )

        evaluation_output = {
            'results': [result],
            'metrics': {'exact_match_accuracy': 1.0},
            'total_examples': 1,
            'total_processing_time': 1.0,
            'average_processing_time': 1.0
        }

        # Save results
        output_path = Path(self.temp_dir) / "test_results.json"
        self.evaluator.save_results(evaluation_output, str(output_path))

        # Verify file was created and contains expected data
        assert output_path.exists()

        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        assert 'evaluation_summary' in saved_data
        assert 'detailed_results' in saved_data
        assert len(saved_data['detailed_results']) == 1
        assert saved_data['detailed_results'][0]['exact_match'] is True


class TestDummyAgent:
    """Test the DummyAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = DummyAgent()

    def test_answer_question_gross_profit(self):
        """Test answering gross profit questions."""
        response = self.agent.answer_question("context", "What is gross profit?")
        assert response.answer == "$32,095 (in millions)"
        assert response.confidence_score == 0.8

    def test_answer_question_revenue(self):
        """Test answering revenue questions."""
        response = self.agent.answer_question("context", "What is revenue?")
        assert response.answer == "$254,453"
        assert response.confidence_score == 0.9

    def test_answer_question_unknown(self):
        """Test answering unknown questions."""
        response = self.agent.answer_question("context", "What is the meaning of life?")
        assert response.answer == "Unable to determine"
        assert response.confidence_score == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])