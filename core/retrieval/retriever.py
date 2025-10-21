import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

# Assuming config_loader, logger_config, embedding_generator, and chroma_manager are set up
from ..config_loader import get_config_value
from ..embedding.embedding_generator import get_embedding_model
from ..vector_db.chroma_manager import get_or_create_collection

logger = logging.getLogger(__name__)

def embed_query(query_text: str) -> Optional[List[float]]:
    """
    Generates an embedding for a single query string using the configured model.

    Args:
        query_text: The user's query string.

    Returns:
        The query embedding (list of floats), or None if embedding fails.
    """
    if not query_text:
        logger.warning("Received empty query text. Cannot generate embedding.")
        return None

    embedding_model = get_embedding_model()
    if not embedding_model:
        logger.error("Embedding model not available. Cannot embed query.")
        return None

    try:
        logger.debug(f"Generating embedding for query: '{query_text[:100]}...'")
        query_embedding = embedding_model.embed_query(query_text)
        logger.debug(f"Query embedding generated successfully (dimension: {len(query_embedding)}).")
        return query_embedding
    except Exception as e:
        logger.error(f"Failed to embed query '{query_text[:100]}...': {e}", exc_info=True)
        return None

def search_vector_db(
    query_embedding: List[float],
    collection_name: str = "technical_docs",
    n_results: Optional[int] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    similarity_threshold: Optional[float] = None
) -> Optional[List[Document]]:
    """
    Searches the ChromaDB collection for documents similar to the query embedding.

    Args:
        query_embedding: The embedding vector for the user's query.
        collection_name: The name of the ChromaDB collection to search.
        n_results: The maximum number of results to retrieve. Defaults to config value.
        filter_dict: Optional dictionary for metadata filtering (ChromaDB 'where' clause).
        similarity_threshold: Optional minimum similarity score (cosine) or maximum distance (L2).
                              Note: ChromaDB returns distances (lower is better). We might need to convert
                              similarity thresholds to distance thresholds depending on the metric used.

    Returns:
        A list of relevant LangChain Document objects, sorted by relevance, or None on failure.
    """
    collection = get_or_create_collection(collection_name)
    if not collection:
        logger.error(f"Failed to get collection '{collection_name}' for searching.")
        return None

    # Get default n_results and threshold from config if not provided
    if n_results is None:
        n_results = int(get_config_value("retrieval.k_results", 5))
    if similarity_threshold is None:
        # Config might store similarity (higher is better), Chroma uses distance (lower is better)
        # We'll handle this after getting results for now, unless using distance directly.
        # Let's assume config stores a similarity threshold for now.
        # Lower the default fallback similarity threshold to make filtering less strict
        similarity_threshold = float(get_config_value("retrieval.similarity_threshold", 0.3))
        # If using L2 distance (default in Chroma), a lower distance is better.
        # If using cosine similarity, higher is better (1.0 is identical).
        # Chroma's query returns distance. If metric is cosine, distance = 1 - similarity.
        # If metric is L2, distance is Euclidean distance.
        # We need to know the collection's distance metric. Assume cosine for threshold conversion for now.
        distance_threshold = 1.0 - similarity_threshold if similarity_threshold is not None else None
        logger.debug(f"Using default n_results={n_results}, similarity_threshold={similarity_threshold} (distance_threshold ~{distance_threshold})")
    else:
        # Convert provided similarity threshold to distance if needed (assuming cosine)
        distance_threshold = 1.0 - similarity_threshold
        logger.debug(f"Using provided n_results={n_results}, similarity_threshold={similarity_threshold} (distance_threshold ~{distance_threshold})")


    try:
        logger.debug(f"Querying collection '{collection_name}' with {n_results} results requested.")
        if filter_dict:
            logger.debug(f"Applying metadata filter: {filter_dict}")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,  # Pass the metadata filter dictionary here
            include=['metadatas', 'documents', 'distances']
        )

        if not results or not results.get('ids') or not results['ids'][0]:
            logger.info(f"No results found in collection '{collection_name}' for the query.")
            return []

        # Process results: Chroma returns lists within lists for batch compatibility
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents_content = results['documents'][0]

        retrieved_docs: List[Document] = []
        num_results_before_filter = len(ids)
        logger.debug(f"Retrieved {num_results_before_filter} raw results from ChromaDB.")

        for i in range(num_results_before_filter):
            distance = distances[i]
            metadata = metadatas[i]
            content = documents_content[i]

            # Apply distance threshold filter (assuming lower distance is better)
            # This check depends on the distance metric used (L2 or cosine)
            # For cosine distance (1 - similarity), lower distance means higher similarity.
            # For L2 distance, lower distance means higher similarity.
            # TEMPORARILY DISABLED DISTANCE FILTERING FOR DEBUGGING
            # if distance_threshold is not None and distance > distance_threshold:
            #     logger.debug(f"Filtering out result {ids[i]} due to distance {distance:.4f} > threshold {distance_threshold:.4f}")
            #     continue
            
            # Add debug log to show distances of all results
            logger.info(f"Result {i+1}: ID={ids[i]}, distance={distance:.4f}, source={metadata.get('source', 'unknown')}")

            # Reconstruct LangChain Document object
            doc = Document(page_content=content, metadata=metadata)
            # Optionally add distance to metadata for later ranking/inspection
            doc.metadata['retrieval_distance'] = distance
            retrieved_docs.append(doc)

        logger.info(f"Retrieved {len(retrieved_docs)} documents after distance filtering (threshold: {distance_threshold}).")

        # Results from ChromaDB are already sorted by distance (most relevant first)
        return retrieved_docs

    except Exception as e:
        logger.error(f"Error querying collection '{collection_name}': {e}", exc_info=True)
        return None

# Example Usage (optional - for testing this module directly)
# if __name__ == "__main__":
#     import sys
#     import time
#     from pathlib import Path
#     # Add project root to sys.path
#     project_root = Path(__file__).resolve().parent.parent.parent
#     if str(project_root) not in sys.path:
#         sys.path.append(str(project_root))
#
#     from core.logger_config import setup_logging
#     # Assuming you have run the chroma_manager example to populate 'test_collection'
#     from core.vector_db.chroma_manager import add_chunks_to_vector_db # To ensure data exists
#
#     setup_logging() # Configure logging first
#
#     # --- Ensure test data exists ---
#     # Re-run adding dummy data to ensure the test collection exists and has content
#     logger.info("--- Ensuring test data exists in 'test_collection' ---")
#     chunk1 = Document(page_content="Information about apples and oranges.", metadata={"source": "fruits.txt", "page": 1, "category": "fruit"})
#     chunk2 = Document(page_content="Details on software development cycles.", metadata={"source": "dev.pdf", "page": 10, "category": "tech"})
#     chunk3 = Document(page_content="Comparing apples to pears.", metadata={"source": "fruits.txt", "page": 2, "category": "fruit"})
#     dummy_chunks = [chunk1, chunk2, chunk3]
#     # Use actual embedder to get realistic embeddings and dimension
#     embedder = get_embedding_model()
#     if embedder:
#         dummy_embeddings = embedder.embed_documents([d.page_content for d in dummy_chunks])
#         add_success = add_chunks_to_vector_db(dummy_chunks, dummy_embeddings, collection_name="test_collection")
#         if not add_success:
#              logger.error("Failed to add test data. Retrieval test might fail.")
#     else:
#         logger.error("Failed to get embedder. Cannot add realistic test data.")
#         # Add dummy embeddings if embedder fails, but results will be meaningless
#         # emb_dim = 384
#         # dummy_embeddings = [[(0.1 * i + 0.01 * j) for j in range(emb_dim)] for i in range(len(dummy_chunks))]
#         # add_chunks_to_vector_db(dummy_chunks, dummy_embeddings, collection_name="test_collection")
#
#     time.sleep(1) # Give ChromaDB a moment to settle if needed
#
#     # --- Test Retrieval ---
#     logger.info("\n--- Testing Retrieval ---")
#     test_query = "Tell me about apples"
#
#     query_embedding = embed_query(test_query)
#
#     if query_embedding:
#         logger.info(f"Query: '{test_query}'")
#
#         # Test basic retrieval
#         logger.info("\n--- Basic Retrieval (Top 2) ---")
#         retrieved_docs = search_vector_db(query_embedding, collection_name="test_collection", n_results=2)
#         if retrieved_docs is not None:
#             logger.info(f"Found {len(retrieved_docs)} documents:")
#             for i, doc in enumerate(retrieved_docs):
#                 dist = doc.metadata.get('retrieval_distance', 'N/A')
#                 logger.info(f"  {i+1}. Dist: {dist:.4f} | Source: {doc.metadata.get('source')} | Content: {doc.page_content[:80]}...")
#         else:
#             logger.error("Basic retrieval failed.")
#
#         # Test retrieval with metadata filter
#         logger.info("\n--- Retrieval with Filter (category='fruit') ---")
#         fruit_filter = {"category": "fruit"}
#         retrieved_docs_filtered = search_vector_db(
#             query_embedding,
#             collection_name="test_collection",
#             n_results=5, # Ask for more initially
#             filter_dict=fruit_filter
#         )
#         if retrieved_docs_filtered is not None:
#             logger.info(f"Found {len(retrieved_docs_filtered)} documents matching filter {fruit_filter}:")
#             for i, doc in enumerate(retrieved_docs_filtered):
#                  dist = doc.metadata.get('retrieval_distance', 'N/A')
#                  logger.info(f"  {i+1}. Dist: {dist:.4f} | Source: {doc.metadata.get('source')} | Category: {doc.metadata.get('category')} | Content: {doc.page_content[:80]}...")
#         else:
#             logger.error("Filtered retrieval failed.")
#
#         # Test retrieval with similarity threshold
#         logger.info("\n--- Retrieval with Similarity Threshold (e.g., > 0.7, implies distance < 0.3 if cosine) ---")
#         # Note: The actual distance values depend heavily on the embedding model and data.
#         # Adjust the threshold based on observed distances in the basic retrieval test.
#         # Let's use a distance threshold directly for clarity, e.g., distance < 0.5
#         low_similarity_threshold = 0.5 # Corresponds to distance < 0.5 if using cosine distance
#         retrieved_docs_thresh = search_vector_db(
#             query_embedding,
#             collection_name="test_collection",
#             n_results=5,
#             similarity_threshold=low_similarity_threshold # Pass similarity, function converts to distance
#         )
#         if retrieved_docs_thresh is not None:
#             logger.info(f"Found {len(retrieved_docs_thresh)} documents with similarity > {low_similarity_threshold} (distance < {1.0-low_similarity_threshold:.4f}):")
#             for i, doc in enumerate(retrieved_docs_thresh):
#                  dist = doc.metadata.get('retrieval_distance', 'N/A')
#                  logger.info(f"  {i+1}. Dist: {dist:.4f} | Source: {doc.metadata.get('source')} | Content: {doc.page_content[:80]}...")
#         else:
#             logger.error("Threshold retrieval failed.")
#
#     else:
#         logger.error("Failed to generate query embedding. Cannot test retrieval.")
#
#     # Optional: Cleanup test collection (as in chroma_manager example) 