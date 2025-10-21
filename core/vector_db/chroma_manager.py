import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from pathlib import Path
import os

# Assuming config_loader and logger_config are set up correctly
from ..config_loader import get_config_value, BASE_DIR
from ..embedding.embedding_generator import get_embedding_model # To get embedder for collection

logger = logging.getLogger(__name__)

# --- Global variable to cache the ChromaDB client ---
_chroma_client_instance: Optional[chromadb.Client] = None

def get_chroma_client() -> Optional[chromadb.Client]:
    """
    Initializes and returns a persistent ChromaDB client instance.
    Caches the client instance globally.

    Reads the persistence path from configuration.
    """
    global _chroma_client_instance
    if _chroma_client_instance is not None:
        logger.debug("Returning cached ChromaDB client instance.")
        return _chroma_client_instance

    persist_directory = str(BASE_DIR / get_config_value("data_paths.vector_db", "data/processed/chroma_db"))
    logger.info(f"Initializing ChromaDB persistent client with directory: {persist_directory}")
    abs_persist_directory = os.path.abspath(persist_directory)
    logger.info(f"Absolute persistence directory path: {abs_persist_directory}")
    logger.info(f"Directory exists: {os.path.exists(abs_persist_directory)}")
    logger.info(f"Directory is writable: {os.access(abs_persist_directory, os.W_OK) if os.path.exists(abs_persist_directory) else 'N/A - directory does not exist'}")

    try:
        os.makedirs(abs_persist_directory, exist_ok=True)
        logger.info(f"Created or verified directory: {abs_persist_directory}")
        # Test file creation
        test_file_path = os.path.join(abs_persist_directory, "chroma_test_file.txt")
        try:
            with open(test_file_path, "w") as f:
                f.write("Test file to verify ChromaDB can write to this directory.")
            logger.info(f"Successfully created test file at: {test_file_path}")
            os.remove(test_file_path)
            logger.info(f"Test file removed successfully.")
        except Exception as test_file_err:
            logger.error(f"Failed to create test file in persistence directory: {test_file_err}")

        # Use the new PersistentClient API
        _chroma_client_instance = chromadb.PersistentClient(path=abs_persist_directory)
        logger.info(f"ChromaDB PersistentClient object created: {_chroma_client_instance}")
        logger.info("ChromaDB persistent client initialized successfully.")
        return _chroma_client_instance

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB persistent client: {e}", exc_info=True)
        return None

def get_in_memory_chroma_client() -> Optional[chromadb.Client]:
    """
    Creates an in-memory ChromaDB client when persistence fails.
    This client will not persist data between application restarts.
    """
    logger.warning("Creating IN-MEMORY ChromaDB client. Data will NOT persist between restarts!")
    try:
        # Create in-memory client
        client_settings = Settings(
            chroma_db_impl="duckdb+parquet",  # Using DuckDB but with no persist_directory
            anonymized_telemetry=False
        )
        client = chromadb.Client(client_settings)
        logger.info("IN-MEMORY ChromaDB client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize IN-MEMORY ChromaDB client: {e}", exc_info=True)
        return None

def get_or_create_collection(
    collection_name: str = "technical_docs",
    embedding_function: Optional[chromadb.EmbeddingFunction] = None
) -> Optional[Collection]:
    """
    Gets or creates a ChromaDB collection. 
    Falls back to in-memory client if persistent client fails.
    """
    # Try with persistent client first
    client = get_chroma_client()
    
    # If persistent client fails, try in-memory client
    if not client:
        logger.warning("Persistent ChromaDB client not available. Trying in-memory client.")
        client = get_in_memory_chroma_client()
        
    if not client:
        logger.error("All ChromaDB client options failed. Cannot access collections.")
        return None

    try:
        # Get or create the collection
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None  # We provide embeddings manually
        )
        try:
            count = collection.count()
            logger.info(f"Collection '{collection_name}' obtained/created. Initial count: {count}")
        except Exception as count_exc:
            logger.error(f"Could not get count for collection '{collection_name}' immediately after get/create: {count_exc}")
        logger.info(f"Successfully retrieved or created collection: '{collection_name}'")
        return collection

    except Exception as e:
        logger.error(f"Failed to get or create collection '{collection_name}': {e}", exc_info=True)
        return None

def add_chunks_to_vector_db(
    chunks: List[Document],
    embeddings: List[List[float]],
    collection_name: str = "technical_docs",
    batch_size: int = 500  # Conservative batch size well below the limit
) -> bool:
    """
    Adds document chunks and their embeddings to the specified ChromaDB collection.
    Processes in batches to avoid exceeding ChromaDB's batch size limit.

    Args:
        chunks: List of LangChain Document objects (containing text and metadata).
        embeddings: List of corresponding embeddings (list of floats).
        collection_name: Name of the target ChromaDB collection.
        batch_size: Size of batches to process (to avoid exceeding ChromaDB limits)

    Returns:
        True if adding was successful (or partially successful), False otherwise.
    """
    if not chunks or not embeddings:
        logger.warning("Received empty chunks or embeddings list. Nothing to add.")
        return True # Nothing to do, considered success

    if len(chunks) != len(embeddings):
        logger.error(f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}). Cannot add to DB.")
        return False

    collection = get_or_create_collection(collection_name)
    if not collection:
        logger.error(f"Failed to get collection '{collection_name}'. Cannot add documents.")
        return False

    logger.info(f"Adding {len(chunks)} chunks to collection '{collection_name}' in batches of {batch_size}...")

    # Process in batches
    total_added = 0
    all_batches_succeeded = True  # Flag to track overall success
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        # Prepare data for ChromaDB batch insertion
        ids = []
        metadatas = []
        documents_content = []

        for j, chunk in enumerate(batch_chunks):
            chunk_idx = i + j  # Global index
            # Create a unique ID for each chunk
            source = chunk.metadata.get("source", "unknown_source")
            page = chunk.metadata.get("page", 0)
            chunk_seq = chunk.metadata.get("chunk_sequence_number", chunk_idx)
            chunk_id = f"{source}_p{page}_c{chunk_seq}"
            ids.append(chunk_id)

            # Process metadata
            chroma_metadata = {
                k: str(v) if v is not None and not isinstance(v, (str, int, float, bool)) else v
                for k, v in chunk.metadata.items()
            }
            chroma_metadata.setdefault("source", "unknown_source")
            chroma_metadata.setdefault("page", 0)
            chroma_metadata.setdefault("chunk_sequence_number", chunk_idx)

            metadatas.append(chroma_metadata)
            documents_content.append(chunk.page_content)

        try:
            # Add current batch to collection
            collection.add(
                embeddings=batch_embeddings,
                documents=documents_content,
                metadatas=metadatas,
                ids=ids
            )
            total_added += len(batch_chunks)
            logger.info(f"Added batch {(i//batch_size)+1}: {len(batch_chunks)} items. Total progress: {total_added}/{len(chunks)}")
            
        except Exception as e:
            logger.error(f"Failed to add batch starting at index {i} to collection '{collection_name}': {e}")
            all_batches_succeeded = False  # Mark failure
            # Continue with next batch instead of failing completely? Or break? 
            # For now, let's continue but ensure we return False later.
            
    if total_added > 0:
        logger.info(f"Finished adding {total_added} of {len(chunks)} items to collection '{collection_name}'.")
        # Explicit persist is no longer needed and causes errors in newer versions.
        # Persistence is handled automatically if the client was initialized with a persist_directory.
        # try:
        #     logger.info("Attempting explicit client.persist()...")
        #     # Need to get the client instance here
        #     client = get_chroma_client()
        #     if client:
        #         client.persist()
        #         logger.info("Explicit client.persist() completed.")
        #     else:
        #         logger.error("Could not get client instance to call persist().")
        # except Exception as persist_err:
        #     logger.error(f"Error during explicit client.persist(): {persist_err}", exc_info=True)
        return all_batches_succeeded # Return True only if ALL batches succeeded
    else:
        logger.error(f"Failed to add any chunks to collection '{collection_name}'.")
        return False

# --- Optional: Basic Query Function ---
# The main retrieval logic will be more complex and likely reside in core/retrieval
def simple_query_vector_db(
    query_embedding: List[float],
    collection_name: str = "technical_docs",
    n_results: int = 5
) -> Optional[Dict[str, Any]]:
    """Performs a simple similarity search."""
    collection = get_or_create_collection(collection_name)
    if not collection:
        logger.error(f"Failed to get collection '{collection_name}' for querying.")
        return None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances'] # Specify what to return
        )
        return results
    except Exception as e:
        logger.error(f"Error querying collection '{collection_name}': {e}", exc_info=True)
        return None


# Example Usage (optional - for testing this module directly)
if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to sys.path
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # --- Need these imports ---
    from core.logger_config import setup_logging
    # Ensure get_embedding_model is imported correctly relative to this file's location
    from ..embedding.embedding_generator import get_embedding_model 
    # --- End imports ---

    setup_logging() # Configure logging first

    # --- Define Test Data ---
    logger.info("--- Setting up Test Data ---")
    test_collection_name = "test_collection_main"
    # Assuming a common dimension like 384 or 768. Adjust if your model differs significantly.
    # Check your embedding_generator if unsure.
    embed_dim = 384 
    logger.info(f"Using assumed embedding dimension for dummy data: {embed_dim}. Ensure this matches your model.")

    chunk1 = Document(page_content="This is the very first chunk about apples.", metadata={"source": "doc1.pdf", "page": 1, "chunk_sequence_number": 1, "topic": "fruit"})
    chunk2 = Document(page_content="Second piece of text mentions bananas.", metadata={"source": "doc1.pdf", "page": 1, "chunk_sequence_number": 2, "topic": "fruit"})
    chunk3 = Document(page_content="Content about oranges from another document.", metadata={"source": "doc2.txt", "page": 0, "chunk_sequence_number": 1, "topic": "fruit"})
    dummy_chunks = [chunk1, chunk2, chunk3]

    # Generate simple dummy embeddings (replace with actual model if testing accuracy needed)
    # For basic add/query testing, simple vectors are okay, but matching dimension is key.
    dummy_embeddings = [[(0.1 * i + 0.01 * j) % 1.0 for j in range(embed_dim)] for i in range(len(dummy_chunks))]


    # --- Clean up potential previous test run ---
    client = get_chroma_client()
    if client:
        try:
            logger.info(f"Attempting to delete existing test collection '{test_collection_name}' if present...")
            client.delete_collection(test_collection_name)
            logger.info(f"Pre-test cleanup: Collection '{test_collection_name}' deleted or did not exist.")
        except Exception as delete_exc:
            # Chroma throws an exception if the collection doesn't exist, safe to ignore here.
            logger.info(f"Pre-test cleanup: Collection '{test_collection_name}' likely did not exist ({delete_exc}).")
            pass 

    # --- Test Adding ---
    logger.info("\n--- Testing Add Chunks ---")
    success = add_chunks_to_vector_db(dummy_chunks, dummy_embeddings, collection_name=test_collection_name)
    logger.info(f"Add chunks operation successful: {success}")

    # --- Test Querying ---
    if success:
        logger.info("\n--- Testing Query ---")
        embedder = get_embedding_model() # Get the actual embedding model
        if not embedder:
             logger.error("Cannot get embedding model for testing queries. Exiting test.")
        else:
            try:
                # Query related to the first chunk
                query_text_1 = "Tell me about the first chunk"
                logger.info(f"Generating embedding for query: \"{query_text_1}\"")
                query_emb_1 = embedder.embed_query(query_text_1)
                
                # Query related to the second/third chunk
                query_text_2 = "What documents mention oranges or bananas?"
                logger.info(f"Generating embedding for query: \"{query_text_2}\"")
                query_emb_2 = embedder.embed_query(query_text_2)

                queries = [(query_text_1, query_emb_1), (query_text_2, query_emb_2)]

                for q_text, q_emb in queries:
                    logger.info(f"\n--- Running Query: \"{q_text}\" ---")
                    # Ensure the query embedding is a list of floats
                    if isinstance(q_emb, list) and all(isinstance(x, float) for x in q_emb):
                         query_results = simple_query_vector_db(q_emb, collection_name=test_collection_name, n_results=2)
                    else:
                         logger.error(f"Query embedding for '{q_text}' is not a valid list of floats. Skipping query.")
                         query_results = None

                    if query_results:
                        logger.info("Query Results:")
                        # Process and print results (Chroma returns lists even for single query)
                        ids = query_results.get('ids', [[]])[0]
                        distances = query_results.get('distances', [[]])[0]
                        metadatas_res = query_results.get('metadatas', [[]])[0]
                        documents_res = query_results.get('documents', [[]])[0]

                        if not ids:
                            logger.warning("Query returned no results.")
                        else:
                            for i in range(len(ids)):
                                logger.info(f"  Result {i+1}:")
                                logger.info(f"    ID: {ids[i]}")
                                logger.info(f"    Distance: {distances[i]:.4f}")
                                logger.info(f"    Metadata: {metadatas_res[i]}")
                                logger.info(f"    Document: {documents_res[i][:100]}...") # Preview
                    else:
                        logger.error("Query failed or was skipped.")

            except Exception as query_err:
                 logger.error(f"An error occurred during query testing: {query_err}", exc_info=True)

    # --- Test Cleanup (Recommended) ---
    client = get_chroma_client() # Re-get client just in case
    if client:
        try:
            logger.info(f"\n--- Cleaning up test collection '{test_collection_name}' ---")
            client.delete_collection(test_collection_name)
            logger.info(f"Test collection '{test_collection_name}' deleted successfully.")
        except Exception as e:
            # Catch potential errors if collection doesn't exist (e.g., if add failed)
            logger.warning(f"Could not delete test collection '{test_collection_name}': {e}") 