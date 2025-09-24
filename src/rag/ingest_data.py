#!/usr/bin/env python3
"""Quick data ingestion script to populate ChromaDB for testing."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from src.rag.data_ingestion import OnDemandIngestion

def ingest_qa_data():
    """Ingest the processed Q&A data into ChromaDB."""
    print("=== Ingesting Q&A Data ===")

    # Initialize FinancialRAG first
    from src.rag.financial_rag import FinancialRAG
    rag = FinancialRAG()

    if not rag.is_available():
        print("❌ FinancialRAG not available")
        return False

    ingestion = OnDemandIngestion(rag)

    # Path to processed Q&A data
    qa_file = "data/datasets/financial_qa/processed_qa_pairs.json"

    if not os.path.exists(qa_file):
        print(f"❌ File not found: {qa_file}")
        return False

    print(f"📁 Ingesting from: {qa_file}")

    success = ingestion.ingest_qa_dataset(qa_file)

    if success:
        print("✅ Q&A data ingested successfully!")

        # Test the RAG after ingestion

        # Check collection count
        collection_info = rag.collection.get()
        doc_count = len(collection_info.get('ids', []))
        print(f"📊 Collection now has {doc_count} documents")

        # Test search
        test_question = "What is working capital?"
        results = rag.search(test_question, n_results=3)
        print(f"🔍 Test search for '{test_question}': {len(results)} results")

        for i, result in enumerate(results[:2]):
            preview = result[:100] + "..." if len(result) > 100 else result
            print(f"   Result {i+1}: {preview}")

    else:
        print("❌ Ingestion failed!")

    return success

if __name__ == "__main__":
    ingest_qa_data()