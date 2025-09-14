#!/usr/bin/env python3
"""
Download and prepare FinanceQA benchmark dataset from HuggingFace.

This script downloads the FinanceQA dataset and prepares it for evaluation.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

try:
    from datasets import load_dataset
except ImportError:
    print("Please install the datasets library: pip install datasets")
    exit(1)


def download_financeqa_dataset(output_dir: str = "data/datasets/financeqa") -> Dict[str, Any]:
    """
    Download the FinanceQA dataset from HuggingFace and save locally.

    Args:
        output_dir: Directory to save the dataset

    Returns:
        Dataset statistics and info
    """
    print("Downloading FinanceQA dataset from HuggingFace...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset from HuggingFace
    dataset = load_dataset("AfterQuery/FinanceQA")

    stats = {}

    # Process each split (train, test, validation if available)
    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split with {len(split_data)} examples...")

        # Convert to pandas DataFrame for easier manipulation
        df = split_data.to_pandas()

        # Save as JSON Lines format for easier loading
        jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                json.dump(row.to_dict(), f, ensure_ascii=False)
                f.write('\n')

        # Save as CSV for human inspection
        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # Collect statistics
        stats[split_name] = {
            'total_examples': len(df),
            'columns': list(df.columns),
            'sample_example': df.iloc[0].to_dict() if len(df) > 0 else {}
        }

        print(f"Saved {split_name} split to {jsonl_path} and {csv_path}")

    # Save dataset info and statistics
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_name': 'FinanceQA',
            'source': 'https://huggingface.co/datasets/AfterQuery/FinanceQA',
            'splits': stats,
            'download_timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print(f"Dataset info saved to {info_path}")
    print("\nDataset Statistics:")
    for split_name, split_stats in stats.items():
        print(f"  {split_name}: {split_stats['total_examples']} examples")
        print(f"    Columns: {split_stats['columns']}")

    return stats


def analyze_dataset_structure(dataset_dir: str = "data/datasets/financeqa") -> Dict[str, Any]:
    """
    Analyze the structure and content of the downloaded dataset.

    Args:
        dataset_dir: Directory containing the dataset

    Returns:
        Analysis results
    """
    print("\nAnalyzing dataset structure...")

    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        print(f"Dataset info not found at {info_path}")
        return {}

    with open(info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    analysis = {
        'dataset_info': dataset_info,
        'question_types': {},
        'answer_patterns': {}
    }

    # Analyze each split
    for split_name in dataset_info['splits']:
        jsonl_path = os.path.join(dataset_dir, f"{split_name}.jsonl")

        if os.path.exists(jsonl_path):
            print(f"\nAnalyzing {split_name} split...")

            # Load examples
            examples = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line.strip()))

            # Analyze question types and patterns
            question_lengths = []
            answer_lengths = []

            for example in examples[:100]:  # Analyze first 100 examples
                if 'question' in example:
                    question_lengths.append(len(example['question'].split()))
                if 'answer' in example:
                    answer_lengths.append(len(str(example['answer']).split()))

            analysis['question_types'][split_name] = {
                'avg_question_length': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
                'avg_answer_length': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
                'sample_questions': [ex.get('question', '') for ex in examples[:3]]
            }

            print(f"  Average question length: {analysis['question_types'][split_name]['avg_question_length']:.1f} words")
            print(f"  Average answer length: {analysis['question_types'][split_name]['avg_answer_length']:.1f} words")

    # Save analysis results
    analysis_path = os.path.join(dataset_dir, "dataset_analysis.json")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\nDataset analysis saved to {analysis_path}")
    return analysis


if __name__ == "__main__":
    # Download dataset
    stats = download_financeqa_dataset()

    # Analyze dataset structure
    analysis = analyze_dataset_structure()

    print("\nFinanceQA dataset download and preparation completed!")
    print("Files created:")
    print("  - data/datasets/financeqa/*.jsonl (dataset splits)")
    print("  - data/datasets/financeqa/*.csv (human-readable format)")
    print("  - data/datasets/financeqa/dataset_info.json (metadata)")
    print("  - data/datasets/financeqa/dataset_analysis.json (analysis)")