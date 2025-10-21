import logging
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assuming config_loader and logger_config are set up correctly
from ..config_loader import get_config_value

logger = logging.getLogger(__name__)

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: A list of cleaned LangChain Document objects.

    Returns:
        A list of smaller LangChain Document objects (chunks).
    """
    chunk_size = int(get_config_value("processing.chunk_size", 500))
    chunk_overlap = int(get_config_value("processing.chunk_overlap", 50))

    logger.info(f"Starting chunking process for {len(documents)} documents...")
    logger.info(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use standard character length function
        is_separator_regex=False, # Treat separators literally
        # Common separators for technical text, add more if needed
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        # keep_separator=True # Might be useful for some contexts
    )

    all_chunks: List[Document] = []
    total_original_docs = len(documents)

    for i, doc in enumerate(documents):
        if not doc.page_content:
            logger.warning(f"Document {i+1}/{total_original_docs} from source '{doc.metadata.get('source', 'N/A')}' has empty content, skipping chunking.")
            continue

        original_content = doc.page_content
        original_metadata = doc.metadata.copy() # Keep original metadata

        logger.debug(f"Chunking document {i+1}/{total_original_docs} from source '{original_metadata.get('source', 'N/A')}' (page {original_metadata.get('page', 'N/A')})...")

        try:
            # Split the document content
            split_texts = text_splitter.split_text(original_content)

            # Create new Document objects for each chunk
            doc_chunks: List[Document] = []
            for j, chunk_text in enumerate(split_texts):
                chunk_metadata = original_metadata.copy()
                # Add chunk-specific metadata
                chunk_metadata["chunk_sequence_number"] = j + 1
                chunk_metadata["total_chunks_in_doc"] = len(split_texts)
                # Potentially add a unique chunk ID later if needed
                # chunk_metadata["chunk_id"] = f"{original_metadata.get('source', 'unknown')}_p{original_metadata.get('page', '0')}_c{j+1}"

                chunk_doc = Document(page_content=chunk_text, metadata=chunk_metadata)
                doc_chunks.append(chunk_doc)

            logger.debug(f"Document {i+1} split into {len(doc_chunks)} chunks.")
            all_chunks.extend(doc_chunks)

        except Exception as e:
            logger.error(f"Error chunking document {i+1} from source '{original_metadata.get('source', 'N/A')}': {e}", exc_info=True)
            # Decide whether to skip the doc or halt processing

    logger.info(f"Finished chunking process. Generated {len(all_chunks)} chunks from {total_original_docs} original documents.")
    return all_chunks

# Example Usage (optional - for testing this module directly)
# if __name__ == "__main__":
#     import sys
#     from pathlib import Path
#     # Add project root to sys.path to allow imports like core.logger_config
#     project_root = Path(__file__).resolve().parent.parent.parent
#     if str(project_root) not in sys.path:
#         sys.path.append(str(project_root))
#
#     from core.logger_config import setup_logging
#     setup_logging() # Configure logging first
#
#     # Create dummy documents (simulate output from cleaning)
#     long_text = """This is the first paragraph. It contains several sentences and provides introductory context. We need to ensure it gets split correctly.
#
# This is the second paragraph, separated by a double newline. It discusses a different aspect of the topic. Technical manuals often have distinct paragraphs like this. Let's make this one a bit longer to test the chunk size limit effectively. We need more words here to push it over the edge, hopefully triggering a split based on the configured chunk size. Adding more filler text just to increase the length. Still going. Almost there. Okay, this should be enough.
#
# Third paragraph. Maybe a short one. Followed by another. And another. This tests splitting on single newlines if paragraphs are close.
# Final sentence."""
#
#     doc1 = Document(
#         page_content=long_text,
#         metadata={"source": "manual.pdf", "page": 1, "file_path": "data/raw/manual.pdf"}
#     )
#     doc2 = Document(
#         page_content="This is a second document, much shorter. It should likely fit within a single chunk.",
#         metadata={"source": "manual.pdf", "page": 2, "file_path": "data/raw/manual.pdf"}
#     )
#     doc3 = Document(
#         page_content="", # Empty document
#         metadata={"source": "manual.pdf", "page": 3, "file_path": "data/raw/manual.pdf"}
#     )
#
#     docs_to_chunk = [doc1, doc2, doc3]
#
#     logger.info("--- Before Chunking ---")
#     for i, doc in enumerate(docs_to_chunk):
#         logger.info(f"Doc {i+1} Metadata: {doc.metadata}, Length: {len(doc.page_content)}")
#
#     # Assume config chunk_size=150, chunk_overlap=20 for this example test
#     # You might need to temporarily modify get_config_value or set env vars for testing
#     # Or adjust the expected output based on your actual config.yaml settings
#     print("\nNOTE: Example assumes chunk_size=500, overlap=50 from default config.")
#     print("Adjust expectations or config.yaml if needed for testing.\n")
#
#     chunked_documents = chunk_documents(docs_to_chunk)
#
#     logger.info("\n--- After Chunking ---")
#     logger.info(f"Total chunks generated: {len(chunked_documents)}")
#     for i, chunk in enumerate(chunked_documents):
#         logger.info(f"--- Chunk {i+1} ---")
#         logger.info(f"Metadata: {chunk.metadata}")
#         logger.info(f"Content Preview ({len(chunk.page_content)} chars):\n>>>\n{chunk.page_content}\n<<<") 