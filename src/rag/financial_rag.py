"""FinancialRAG - Production RAG implementation with unified collection and metadata filtering.
"""

import logging
import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any, Union
from src.agent.prompts import get_context_builder_prompt

logger = logging.getLogger(__name__)


class FinancialRAG:
    """Production RAG implementation for financial domain with metadata filtering."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "financial_knowledge"):
        """Initialize FinancialRAG with unified collection.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the unified collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and get/create collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Try to get existing collection, create if doesn't exist
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Unified financial knowledge base with source metadata"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            self.client = None
            self.collection = None

    def is_available(self) -> bool:
        """Check if ChromaDB collection is available for queries."""
        return self.client is not None and self.collection is not None

    def search(
        self,
        question: Union[str, List[str]],
        n_results: int = 5,
        source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant financial context with optional source filtering.

        Args:
            question: The financial question(s) to search context for - can be single string or list of keywords
            n_results: Number of results to return
            source_filter: Filter by source type(s): "10k", "qa_dataset", or ["10k", "qa_dataset"]

        Returns:
            List of relevant context with metadata and source attribution
        """
        if not self.is_available():
            logger.warning("ChromaDB not available, returning empty results")
            return []

        # Handle both string and list inputs
        if isinstance(question, list):
            if not question or all(not q.strip() for q in question):
                logger.warning("Empty question list provided, returning empty results")
                return []
            # Combine multiple keywords into a single query
            combined_question = ' '.join(q.strip() for q in question if q.strip())
        else:
            if not question or not question.strip():
                logger.warning("Empty question provided, returning empty results")
                return []
            combined_question = question.strip()

        try:
            # Prepare where clause for source filtering
            where_clause = None
            if source_filter:
                if isinstance(source_filter, str):
                    where_clause = {"source_type": source_filter}
                elif isinstance(source_filter, list):
                    where_clause = {"source_type": {"$in": source_filter}}

            # Perform search with optional filtering
            results = self.collection.query(
                query_texts=[combined_question],
                n_results=n_results,
                where=where_clause
            )

            # Format results with metadata
            formatted_results = []
            if results and results.get('documents') and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]

                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    score = 1 - distances[i] if i < len(distances) else 0.0

                    result = {
                        'text': doc,
                        'source_type': metadata.get('source_type', 'unknown'),
                        'metadata': metadata,
                        'score': score
                    }
                    formatted_results.append(result)

            filter_desc = f" (filtered by: {source_filter})" if source_filter else ""
            logger.info(f"Found {len(formatted_results)} results for question{filter_desc}")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def search_10k_only(self, question: Union[str, List[str]], n_results: int = 5) -> List[Dict[str, Any]]:
        """Search only 10-K documents.

        Args:
            question: Search query - can be single string or list of keywords
            n_results: Number of results to return

        Returns:
            List of 10-K results
        """
        return self.search(question, n_results, source_filter="10k")

    def search_qa_only(self, question: Union[str, List[str]], n_results: int = 5) -> List[Dict[str, Any]]:
        """Search only Q&A dataset.

        Args:
            question: Search query - can be single string or list of keywords
            n_results: Number of results to return

        Returns:
            List of Q&A results
        """
        return self.search(question, n_results, source_filter="qa_dataset")

    def search_all_sources(self, question: Union[str, List[str]], n_results: int = 5) -> List[Dict[str, Any]]:
        """Search across all sources without filtering.

        Args:
            question: Search query - can be single string or list of keywords
            n_results: Number of results to return

        Returns:
            List of results from all sources
        """
        return self.search(question, n_results, source_filter=None)

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        skip_existing: bool = True
    ) -> bool:
        """Add documents to the collection with metadata and deduplication.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries (must include 'source_type')
            ids: List of unique document IDs
            skip_existing: If True, skip documents that already exist

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("ChromaDB collection not available")
            return False

        if not (len(documents) == len(metadatas) == len(ids)):
            logger.error("Documents, metadatas, and ids must have same length")
            return False

        # Validate that all metadata includes source_type
        for i, metadata in enumerate(metadatas):
            if 'source_type' not in metadata:
                logger.error(f"Metadata at index {i} missing required 'source_type' field")
                return False

        try:
            # Check for existing documents if skip_existing is True
            if skip_existing:
                documents_to_add, metadatas_to_add, ids_to_add = self._filter_existing_documents(
                    documents, metadatas, ids
                )

                if len(ids_to_add) == 0:
                    logger.info("All documents already exist in collection, skipping")
                    return True

                logger.info(f"Adding {len(ids_to_add)} new documents (skipped {len(ids) - len(ids_to_add)} existing)")
            else:
                documents_to_add, metadatas_to_add, ids_to_add = documents, metadatas, ids

            self.collection.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            logger.info(f"Successfully added {len(documents_to_add)} documents to collection")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def _filter_existing_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> tuple:
        """Filter out documents that already exist in the collection.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs

        Returns:
            Tuple of (filtered_documents, filtered_metadatas, filtered_ids)
        """
        try:
            # Check which IDs already exist
            existing_check = self.collection.get(ids=ids)
            existing_ids = set(existing_check.get('ids', []))

            # Filter out existing documents
            filtered_documents = []
            filtered_metadatas = []
            filtered_ids = []

            for i, doc_id in enumerate(ids):
                if doc_id not in existing_ids:
                    filtered_documents.append(documents[i])
                    filtered_metadatas.append(metadatas[i])
                    filtered_ids.append(doc_id)

            return filtered_documents, filtered_metadatas, filtered_ids

        except Exception as e:
            logger.warning(f"Error checking existing documents: {e}")
            # If check fails, return all documents to be safe
            return documents, metadatas, ids

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection by source type.

        Returns:
            Dictionary with collection statistics
        """
        if not self.is_available():
            return {"available": False}

        try:
            total_count = self.collection.count()

            # Get counts by source type
            stats = {
                "available": True,
                "collection_name": self.collection_name,
                "total_documents": total_count,
                "persist_directory": self.persist_directory,
                "by_source": {}
            }

            # Count documents by source type
            for source_type in ["10k", "qa_dataset"]:
                try:
                    source_results = self.collection.get(
                        where={"source_type": source_type}
                    )
                    count = len(source_results.get('ids', [])) if source_results else 0
                    stats["by_source"][source_type] = count
                except Exception:
                    stats["by_source"][source_type] = 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"available": False, "error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all documents from the collection.

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("ChromaDB collection not available")
            return False

        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data and all_data.get('ids'):
                self.collection.delete(ids=all_data['ids'])
                logger.info(f"Cleared {len(all_data['ids'])} documents from collection")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def clear_source_type(self, source_type: str) -> bool:
        """Clear documents of a specific source type.

        Args:
            source_type: Source type to clear ("10k" or "qa_dataset")

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("ChromaDB collection not available")
            return False

        try:
            # Get documents of this source type
            source_data = self.collection.get(where={"source_type": source_type})

            if source_data and source_data.get('ids'):
                self.collection.delete(ids=source_data['ids'])
                logger.info(f"Cleared {len(source_data['ids'])} documents of type '{source_type}'")
            else:
                logger.info(f"No documents found for source type '{source_type}'")

            return True

        except Exception as e:
            logger.error(f"Failed to clear source type '{source_type}': {e}")
            return False


# Utility functions for common operations

def create_financial_rag(
    persist_directory: str = "./chroma_db",
    collection_name: str = "financial_knowledge"
) -> FinancialRAG:
    """Create and initialize FinancialRAG instance.

    Args:
        persist_directory: ChromaDB persistence directory
        collection_name: Collection name

    Returns:
        FinancialRAG instance
    """
    return FinancialRAG(persist_directory, collection_name)


def generate_document_id(file_link: str, file_name: str, company: str, chunk_index: Optional[int] = None) -> str:
    """Generate consistent document ID using the format: {file_link}_{file_name}_{company}.

    Args:
        file_link: Source file link or URL
        file_name: Name of the source file
        company: Company name
        chunk_index: Optional chunk index for multi-chunk documents

    Returns:
        Formatted document ID
    """
    # Clean components to be safe for IDs
    clean_link = file_link.replace("/", "_").replace(":", "").replace("?", "").replace("&", "")
    clean_file = file_name.replace(".", "_").replace(" ", "_")
    clean_company = company.replace(" ", "_").replace(",", "").replace(".", "")

    base_id = f"{clean_link}_{clean_file}_{clean_company}"

    if chunk_index is not None:
        return f"{base_id}_chunk_{chunk_index}"

    return base_id


def generate_10k_document_id(
    filing_url: str,
    company: str,
    section: str,
    chunk_index: int,
    fiscal_year: str = ""
) -> str:
    """Generate document ID for 10-K documents.

    Args:
        filing_url: SEC filing URL
        company: Company name (e.g., "Costco")
        section: Document section (e.g., "MD&A", "Financial_Statements")
        chunk_index: Chunk number within section
        fiscal_year: Optional fiscal year

    Returns:
        Document ID for 10-K chunk
    """
    file_name = f"10K_{fiscal_year}" if fiscal_year else "10K"
    return generate_document_id(filing_url, file_name, company, chunk_index)


def generate_qa_document_id(
    dataset_source: str,
    dataset_name: str,
    original_index: int
) -> str:
    """Generate document ID for Q&A dataset entries.

    Args:
        dataset_source: Source URL or identifier (e.g., kaggle URL)
        dataset_name: Name of the dataset
        original_index: Original index in the source dataset

    Returns:
        Document ID for Q&A pair
    """
    return generate_document_id(dataset_source, dataset_name, "financial_qa", original_index)


def search_with_analysis_routing(
    rag: FinancialRAG,
    question: str,
    analysis_output: Dict[str, Any],
    n_results: int = 5
) -> Dict[str, Any]:
    """Search using enhanced analysis output with intelligent routing.

    Args:
        rag: FinancialRAG instance
        question: Search query
        analysis_output: Output from enhanced analysis step
        n_results: Number of results to return

    Returns:
        Dictionary with search results and routing metadata
    """
    # Extract routing information from analysis
    recommended_sources = analysis_output.get("recommended_sources", ["both"])
    search_keywords = analysis_output.get("search_keywords", [])
    question_type = analysis_output.get("question_type", "basic")

    # Use search keywords if available, otherwise use original question
    search_query = " ".join(search_keywords) if search_keywords else question

    # Route based on analysis recommendations
    if recommended_sources == ["10k"]:
        results = rag.search_10k_only(search_query, n_results)
        routing_strategy = "10k_only"
    elif recommended_sources == ["qa_dataset"]:
        results = rag.search_qa_only(search_query, n_results)
        routing_strategy = "qa_dataset_only"
    elif recommended_sources == ["both"] or "both" in recommended_sources:
        results = rag.search_all_sources(search_query, n_results)
        routing_strategy = "all_sources"
    else:
        # Multiple specific sources
        valid_sources = [s for s in recommended_sources if s in ["10k", "qa_dataset"]]
        if valid_sources:
            results = rag.search(search_query, n_results, source_filter=valid_sources)
            routing_strategy = f"filtered_{'+'.join(valid_sources)}"
        else:
            results = rag.search_all_sources(search_query, n_results)
            routing_strategy = "fallback_all_sources"

    return {
        "results": results,
        "routing_metadata": {
            "strategy": routing_strategy,
            "recommended_sources": recommended_sources,
            "search_query": search_query,
            "original_question": question,
            "question_type": question_type,
            "total_results": len(results)
        }
    }


def search_with_source_routing(
    rag: FinancialRAG,
    question: str,
    recommended_sources: List[str],
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """Search using analysis-recommended sources (legacy function).

    Args:
        rag: FinancialRAG instance
        question: Search query
        recommended_sources: List of recommended sources from analysis
        n_results: Number of results to return

    Returns:
        Search results from recommended sources
    """
    if not recommended_sources:
        return rag.search_all_sources(question, n_results)

    if len(recommended_sources) == 1:
        source = recommended_sources[0]
        if source == "10k":
            return rag.search_10k_only(question, n_results)
        elif source == "qa_dataset":
            return rag.search_qa_only(question, n_results)
        elif source == "both":
            return rag.search_all_sources(question, n_results)

    # Multiple specific sources
    valid_sources = [s for s in recommended_sources if s in ["10k", "qa_dataset"]]
    if valid_sources:
        return rag.search(question, n_results, source_filter=valid_sources)

    # Fallback to all sources
    return rag.search_all_sources(question, n_results)


def build_llm_context(
    search_results: List[Dict[str, Any]],
    question: str,
    provided_context: str,
    model_manager
) -> str:
    """Build context using LLM instead of hardcoded templates.

    Args:
        search_results: Results from FinancialRAG search
        question: Original question
        provided_context: Original context
        model_manager: Model manager for LLM access

    Returns:
        LLM-organized context string
    """
    if not search_results:
        return "No relevant context found."

    try:

        # Get LLM prompt for context building
        context_prompt = get_context_builder_prompt(question, provided_context, search_results)

        # Generate organized context using LLM
        try:
            response = model_manager.generate_sync(context_prompt)
            if hasattr(response, 'content'):
                organized_context = response.content or "Failed to organize context."
            else:
                organized_context = str(response) or "Failed to organize context."

            return organized_context

        except Exception as e:
            logger.error(f"LLM context building failed: {e}")
            # Simple fallback - just concatenate results with basic attribution
            fallback_parts = []
            for i, result in enumerate(search_results[:5], 1):
                source_type = result.get('source_type', 'unknown')
                text = result.get('text', '')[:500]  # Truncate
                fallback_parts.append(f"{i}. [{source_type.upper()}] {text}")

            return "\n\n".join(fallback_parts)

    except Exception as e:
        logger.error(f"Context building completely failed: {e}")
        return "Context building failed."


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create FinancialRAG instance
    rag = create_financial_rag()

    if rag.is_available():
        # Get statistics
        stats = rag.get_collection_stats()
        print(f"Collection stats: {stats}")

        # Test search
        if stats.get("total_documents", 0) > 0:
            results = rag.search("What is EBITDA?", n_results=3)
            print(f"Search results: {len(results)} found")

            for result in results:
                print(f"- Source: {result['source_type']}, Score: {result['score']:.3f}")
        else:
            print("No documents in collection. Use ingestion scripts to add data.")
    else:
        print("FinancialRAG not available")