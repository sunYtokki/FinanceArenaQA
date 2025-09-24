"""Simple data processor for financial Q&A dataset preparation.

This module handles parsing, cleaning, and preprocessing of Q&A datasets
for ingestion into ChromaDB for RAG retrieval.
"""

import re
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class FinancialQAProcessor:
    """Simple processor for financial Q&A datasets."""

    def __init__(self):
        """Initialize the processor."""
        self.processed_count = 0
        self.skipped_count = 0

    def load_kaggle_dataset(self, file_path: str) -> pd.DataFrame:
        """Load Kaggle financial Q&A dataset from CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with Q&A pairs
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset from {file_path}: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)\:\;]', '', text)

        # Normalize financial terms
        text = re.sub(r'\$(\d+)', r'$\1', text)  # Ensure $ is attached to numbers
        text = re.sub(r'(\d+)%', r'\1%', text)    # Ensure % is attached to numbers

        return text

    def extract_qa_pairs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract Q&A pairs from dataset with basic validation.

        Args:
            df: DataFrame containing Q&A data

        Returns:
            List of validated Q&A pairs
        """
        qa_pairs = []

        # Try to identify question and answer columns
        question_col = self._find_column(df, ['question', 'q', 'query', 'prompt'])
        answer_col = self._find_column(df, ['answer', 'a', 'response', 'text'])
        context_col = self._find_column(df, ['context', 'c', 'reasoning', 'reason'])

        if not question_col or not answer_col:
            logger.error(f"Could not identify Q&A columns. Available: {list(df.columns)}")
            raise ValueError("Could not identify question and answer columns")

        logger.info(f"Using columns: question='{question_col}', answer='{answer_col}'")

        for idx, row in df.iterrows():
            try:
                question = self.clean_text(row[question_col])
                answer = self.clean_text(row[answer_col])
                context = self.clean_text(row[context_col])

                if self._is_valid_qa_pair(question, answer):
                    qa_pair = {
                        'id': f"kaggle_qa_{idx}",
                        'question': question,
                        'answer': answer,
                        'context': context,
                        'source': 'kaggle_financial_qa',
                        'metadata': {
                            'original_index': idx,
                            'question_length': len(question.split()),
                            'answer_length': len(answer.split()),
                            'context_length': len(context.split())
                            # 'topics': self._extract_topics(question, answer)
                        }
                    }
                    qa_pairs.append(qa_pair)
                    self.processed_count += 1
                else:
                    self.skipped_count += 1

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                self.skipped_count += 1

        logger.info(f"Processed {self.processed_count} Q&A pairs, skipped {self.skipped_count}")
        return qa_pairs

    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column by checking possible names (case-insensitive).

        Args:
            df: DataFrame to search
            possible_names: List of possible column names

        Returns:
            Column name if found, None otherwise
        """
        df_columns = [col.lower() for col in df.columns]

        for name in possible_names:
            if name.lower() in df_columns:
                return df.columns[df_columns.index(name.lower())]

        return None

    def _is_valid_qa_pair(self, question: str, answer: str) -> bool:
        """Validate Q&A pair for minimum quality.

        Args:
            question: Question text
            answer: Answer text

        Returns:
            True if valid, False otherwise
        """
        # Check minimum lengths
        if len(question.strip()) < 10 or len(answer.strip()) < 5:
            return False

        # Check for reasonable word counts
        question_words = len(question.split())
        answer_words = len(answer.split())

        if question_words < 3 or answer_words < 2:
            return False

        # Check maximum lengths to avoid extremely long content
        if question_words > 200 or answer_words > 1000:
            return False

        # Check for basic financial content indicators
        financial_terms = [
            'financial', 'finance', 'money', 'revenue', 'profit', 'loss',
            'cash', 'debt', 'equity', 'investment', 'return', 'cost',
            'ebitda', 'margin', 'ratio', 'balance', 'income', 'statement'
        ]

        text_combined = (question + " " + answer).lower()
        has_financial_content = any(term in text_combined for term in financial_terms)

        return has_financial_content

    def _extract_topics(self, question: str, answer: str) -> List[str]:
        """Extract financial topics from Q&A content.

        Args:
            question: Question text
            answer: Answer text

        Returns:
            List of identified topics
        """
        topics = []
        text_combined = (question + " " + answer).lower()

        # Financial topic patterns
        topic_patterns = {
            'valuation': ['valuation', 'npv', 'dcf', 'wacc', 'value'],
            'ratios': ['ratio', 'debt to equity', 'current ratio', 'pe ratio'],
            'profitability': ['profit', 'margin', 'ebitda', 'ebit', 'roi', 'return'],
            'cash_flow': ['cash flow', 'operating cash', 'free cash flow', 'fcf'],
            'balance_sheet': ['balance sheet', 'assets', 'liabilities', 'equity'],
            'income_statement': ['income statement', 'revenue', 'expenses', 'net income'],
            'investment': ['investment', 'irr', 'payback', 'capital'],
            'risk': ['risk', 'beta', 'volatility', 'standard deviation'],
            'growth': ['growth', 'compound', 'cagr', 'expansion'],
            'accounting': ['accounting', 'gaap', 'depreciation', 'amortization']
        }

        for topic, keywords in topic_patterns.items():
            if any(keyword in text_combined for keyword in keywords):
                topics.append(topic)

        return topics[:5]  # Limit to top 5 topics

    def save_processed_data(self, qa_pairs: List[Dict[str, Any]], output_path: str) -> bool:
        """Save processed Q&A pairs to JSON file.

        Args:
            qa_pairs: List of processed Q&A pairs
            output_path: Path to save the processed data

        Returns:
            True if successful, False otherwise
        """
        try:
            import json

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'dataset_info': {
                        'total_pairs': len(qa_pairs),
                        'processed_count': self.processed_count,
                        'skipped_count': self.skipped_count,
                        'source': 'kaggle_financial_qa'
                    },
                    'qa_pairs': qa_pairs
                }, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(qa_pairs)} processed Q&A pairs to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            'processed_count': self.processed_count,
            'skipped_count': self.skipped_count,
            'success_rate': self.processed_count / (self.processed_count + self.skipped_count) if (self.processed_count + self.skipped_count) > 0 else 0
        }


def process_kaggle_dataset(input_file: str, output_file: str) -> bool:
    """Main function to process Kaggle financial Q&A dataset.

    Args:
        input_file: Path to input CSV file
        output_file: Path to save processed JSON file

    Returns:
        True if processing successful, False otherwise
    """
    try:
        processor = FinancialQAProcessor()

        # Load dataset
        df = processor.load_kaggle_dataset(input_file)

        # Extract and clean Q&A pairs
        qa_pairs = processor.extract_qa_pairs(df)

        if not qa_pairs:
            logger.error("No valid Q&A pairs extracted")
            return False

        # Save processed data
        success = processor.save_processed_data(qa_pairs, output_file)

        # Print statistics
        stats = processor.get_processing_stats()
        logger.info(f"Processing complete. Stats: {stats}")

        return success

    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    input_path = "data/datasets/financial_qa/Financial-QA-10k.csv"
    output_path = "data/datasets/financial_qa/processed_qa_pairs.json"

    success = process_kaggle_dataset(input_path, output_path)
    if success:
        print("✓ Dataset processing completed successfully")
    else:
        print("✗ Dataset processing failed")