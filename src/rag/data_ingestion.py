"""Data ingestion utilities for FinancialRAG with on-demand document caching.

This module handles dynamic ingestion of documents that come with questions,
ensuring documents are cached and not re-added when they appear in multiple questions.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .financial_rag import FinancialRAG, generate_10k_document_id, generate_qa_document_id

logger = logging.getLogger(__name__)


class OnDemandIngestion:
    """Handles on-demand ingestion of documents that come with questions."""

    def __init__(self, financial_rag: FinancialRAG):
        """Initialize with FinancialRAG instance.

        Args:
            financial_rag: FinancialRAG instance for document storage
        """
        self.rag = financial_rag
        self.processed_docs = set()  # Track processed document IDs in memory

    def ingest_question_context(
        self,
        question: str,
        context: str,
        context_metadata: Dict[str, Any]
    ) -> bool:
        """Ingest document context that comes with a question.

        Args:
            question: The financial question
            context: Document content (10-K text)
            context_metadata: Metadata containing file_link, company, section, etc.

        Returns:
            True if document was processed (new or existing), False on error
        """
        try:
            # Generate consistent document ID from metadata
            doc_id = self._generate_context_document_id(context, context_metadata)

            # Check if we've already processed this document in this session
            if doc_id in self.processed_docs:
                logger.debug(f"Document {doc_id} already processed in this session")
                return True

            # Prepare metadata for ChromaDB
            document_metadata = {
                "source_type": "10k",
                "company": context_metadata.get("company", "unknown"),
                "section": context_metadata.get("section", "unknown"),
                "filing_url": context_metadata.get("original_context", ""),
                "fiscal_year": context_metadata.get("fiscal_year", ""),
                "doc_length": len(context),
                "associated_question": question[:100]  # Track which question uses this doc
            }

            # Add document to FinancialRAG (will skip if already exists)
            success = self.rag.add_documents(
                documents=[context],
                metadatas=[document_metadata],
                ids=[doc_id],
                skip_existing=True
            )

            if success:
                self.processed_docs.add(doc_id)
                logger.debug(f"Successfully processed document: {doc_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to ingest question context: {e}")
            return False

    def ingest_qa_dataset(self, processed_qa_file: str) -> bool:
        """Ingest processed Q&A dataset into FinancialRAG.

        Args:
            processed_qa_file: Path to processed Q&A JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(processed_qa_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            qa_pairs = data.get('qa_pairs', [])
            if not qa_pairs:
                logger.error("No Q&A pairs found in processed data")
                return False

            # Prepare documents for ingestion
            documents = []
            metadatas = []
            ids = []

            for qa_pair in qa_pairs:
                try:
                    # Combine question and answer for embedding
                    combined_text = f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
                    documents.append(combined_text)

                    # Generate consistent ID
                    doc_id = generate_qa_document_id(
                        dataset_source=qa_pair.get('source', 'kaggle_financial_qa'),
                        dataset_name="financial_qa",
                        original_index=qa_pair['metadata'].get('original_index', len(ids))
                    )
                    ids.append(doc_id)

                    # Prepare metadata
                    metadata = {
                        "source_type": "qa_dataset",
                        "question": qa_pair['question'][:500],  # Truncate for storage
                        "answer": qa_pair['answer'][:1000],
                        "reasoning_context": qa_pair['context'][:5000],
                        "topics": ",".join(qa_pair['metadata'].get('topics', [])),
                        "question_length": qa_pair['metadata'].get('question_length', 0),
                        "answer_length": qa_pair['metadata'].get('answer_length', 0),
                        "dataset_source": qa_pair.get('source', 'kaggle_financial_qa')
                    }
                    metadatas.append(metadata)

                except Exception as e:
                    logger.warning(f"Error preparing Q&A pair for ingestion: {e}")
                    continue

            if not documents:
                logger.error("No Q&A documents prepared for ingestion")
                return False

            # Add to FinancialRAG in batches
            batch_size = 100
            total_added = 0

            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                success = self.rag.add_documents(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids,
                    skip_existing=True
                )

                if success:
                    total_added += len(batch_docs)
                    logger.info(f"Added Q&A batch {i//batch_size + 1}: {len(batch_docs)} documents")

            logger.info(f"Q&A dataset ingestion complete. Total processed: {total_added}")
            return total_added > 0

        except Exception as e:
            logger.error(f"Q&A dataset ingestion failed: {e}")
            return False

    def _generate_context_document_id(self, context: str, metadata: Dict[str, Any]) -> str:
        """Generate consistent document ID for question context.

        Args:
            context: Document content
            metadata: Context metadata

        Returns:
            Consistent document ID
        """
        # Extract metadata components
        file_link = metadata.get("original_context", "unknown_source")
        company = metadata.get("company", "unknown_company")
        section = metadata.get("section", "unknown_section")
        fiscal_year = metadata.get("fiscal_year", "")

        # Create content hash for uniqueness (in case metadata isn't sufficient)
        content_hash = hashlib.md5(context[:1000].encode()).hexdigest()[:8]

        return generate_10k_document_id(
            filing_url=file_link,
            company=company,
            section=section,
            chunk_index=0,  # For now, treat each context as single chunk
            fiscal_year=fiscal_year
        ) + f"_{content_hash}"

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingestion process.

        Returns:
            Dictionary with ingestion statistics
        """
        collection_stats = self.rag.get_collection_stats()

        return {
            "session_processed_docs": len(self.processed_docs),
            "collection_stats": collection_stats,
            "rag_available": self.rag.is_available()
        }


def setup_financial_knowledge_base(
    qa_dataset_file: Optional[str] = None,
    persist_directory: str = "./chroma_db",
    collection_name: str = "financial_knowledge"
) -> OnDemandIngestion:
    """Setup FinancialRAG with optional Q&A dataset pre-loading.

    Args:
        qa_dataset_file: Optional path to processed Q&A dataset
        persist_directory: ChromaDB persistence directory
        collection_name: Collection name

    Returns:
        OnDemandIngestion instance ready for use
    """
    try:
        # Create FinancialRAG instance
        financial_rag = FinancialRAG(persist_directory, collection_name)

        if not financial_rag.is_available():
            logger.error("Failed to initialize FinancialRAG")
            raise Exception("FinancialRAG initialization failed")

        # Create ingestion handler
        ingestion = OnDemandIngestion(financial_rag)

        # Pre-load Q&A dataset if provided
        if qa_dataset_file and Path(qa_dataset_file).exists():
            logger.info(f"Pre-loading Q&A dataset from {qa_dataset_file}")
            success = ingestion.ingest_qa_dataset(qa_dataset_file)
            if success:
                logger.info("Q&A dataset pre-loaded successfully")
            else:
                logger.warning("Q&A dataset pre-loading failed")

        # Get initial stats
        stats = ingestion.get_ingestion_stats()
        logger.info(f"Financial knowledge base ready. Stats: {stats}")

        return ingestion

    except Exception as e:
        logger.error(f"Failed to setup financial knowledge base: {e}")
        raise


# Example usage workflow
def example_question_processing_workflow():
    """Example of how to use on-demand ingestion with questions."""

    # Setup knowledge base
    ingestion = setup_financial_knowledge_base(
        qa_dataset_file="data/datasets/financial_qa/processed_qa_pairs.json"
    )

    # Example: Process questions that come with context
    example_questions = [
        {
            "question": "What is Costco's revenue for 2024?",
            "context": "... [10-K document content] ...",
            "metadata": {
                "company": "Costco",
                "original_context": "https://sec.gov/archives/edgar/data/909832/000090983224000049/cost-20240901.htm",
                "section": "MD&A",
                "fiscal_year": "2024"
            }
        }
    ]

    for item in example_questions:
        # Ingest context on-demand (will cache for future questions)
        success = ingestion.ingest_question_context(
            question=item["question"],
            context=item["context"],
            context_metadata=item["metadata"]
        )

        if success:
            # Now search across all cached documents
            results = ingestion.rag.search(item["question"], n_results=5)
            logger.info(f"Found {len(results)} relevant documents")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    example_question_processing_workflow()