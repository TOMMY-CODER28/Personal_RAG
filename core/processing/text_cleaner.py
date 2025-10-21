import logging
import re
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def remove_excess_whitespace(text: str) -> str:
    """Removes leading/trailing whitespace, multiple spaces, and excessive newlines."""
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines (3 or more) with two newlines (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    return text

def merge_hyphenated_lines(text: str) -> str:
    """Merges words hyphenated across line breaks."""
    # Matches a word ending with a hyphen, followed by a newline,
    # and then a lowercase letter at the start of the next line.
    # It captures the part before the hyphen (group 1) and the part after the newline (group 2).
    return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

def basic_header_footer_removal(text: str, lines_to_check: int = 3, max_char_length: int = 70) -> str:
    """
    Attempts basic removal of repeating headers/footers.
    This is a simple heuristic and might remove valid content or miss complex headers/footers.
    It checks the first/last few lines and removes them if they are short and appear frequently.
    (Note: More robust methods often require layout analysis or comparing lines across pages).
    """
    lines = text.split('\n')
    if len(lines) <= 2 * lines_to_check: # Don't process very short texts
        return text

    potential_headers = lines[:lines_to_check]
    potential_footers = lines[-lines_to_check:]

    cleaned_lines = lines

    # Simple check: remove short lines at the beginning/end that might be headers/footers
    # This is very basic and might need significant refinement based on document patterns.
    # A more robust approach would involve comparing lines across multiple pages (Documents).
    num_lines_removed_start = 0
    for line in potential_headers:
        # Example heuristic: if line is short and doesn't end with punctuation typical of sentences.
        if len(line) < max_char_length and not line.strip().endswith(('.', '?', '!', ':', ';')):
             # And maybe check if it contains page numbers? re.search(r'\b(page|p\.)\s*\d+\b', line.lower())
            num_lines_removed_start += 1
        else:
            break # Stop if a line looks like real content

    num_lines_removed_end = 0
    for line in reversed(potential_footers):
        if len(line) < max_char_length and not line.strip().endswith(('.', '?', '!', ':', ';')):
             # And maybe check if it contains page numbers? re.search(r'\b(page|p\.)\s*\d+\b', line.lower())
            num_lines_removed_end += 1
        else:
            break

    if num_lines_removed_start > 0:
        logger.debug(f"Potential header removal: Removed {num_lines_removed_start} lines from start.")
        cleaned_lines = cleaned_lines[num_lines_removed_start:]

    if num_lines_removed_end > 0:
        logger.debug(f"Potential footer removal: Removed {num_lines_removed_end} lines from end.")
        # Ensure we don't cause index error if all lines were removed (unlikely)
        end_index = len(cleaned_lines) - num_lines_removed_end
        cleaned_lines = cleaned_lines[:end_index] if end_index > 0 else []


    return '\n'.join(cleaned_lines)


def clean_document_content(doc: Document) -> Document:
    """Applies a series of cleaning steps to the document's page_content."""
    if not isinstance(doc.page_content, str):
        logger.warning(f"Document content is not a string, skipping cleaning. Metadata: {doc.metadata}")
        return doc

    original_length = len(doc.page_content)
    cleaned_content = doc.page_content

    # Apply cleaning functions in order
    # 1. Merge hyphenated lines first, as whitespace removal might interfere otherwise
    cleaned_content = merge_hyphenated_lines(cleaned_content)
    # 2. Basic header/footer removal (apply before excessive whitespace removal)
    #    Note: This is heuristic and might need tuning or disabling.
    # cleaned_content = basic_header_footer_removal(cleaned_content)
    # 3. Remove excess whitespace
    cleaned_content = remove_excess_whitespace(cleaned_content)
    # Add more cleaning steps here as needed (e.g., unicode normalization, removing control chars)

    if len(cleaned_content) != original_length:
        logger.debug(f"Cleaned content for doc from source '{doc.metadata.get('source', 'N/A')}' (page {doc.metadata.get('page', 'N/A')}). Length change: {original_length} -> {len(cleaned_content)}")

    doc.page_content = cleaned_content
    return doc

def clean_documents(documents: List[Document]) -> List[Document]:
    """Applies cleaning steps to a list of documents."""
    logger.info(f"Starting cleaning process for {len(documents)} documents...")
    cleaned_docs = [clean_document_content(doc) for doc in documents]
    # Filter out documents that might have become empty after cleaning
    final_docs = [doc for doc in cleaned_docs if doc.page_content]
    if len(final_docs) < len(documents):
        logger.warning(f"Removed {len(documents) - len(final_docs)} documents that became empty after cleaning.")
    logger.info(f"Finished cleaning process. {len(final_docs)} documents remaining.")
    return final_docs

# Example Usage (optional - for testing this module directly)
# if __name__ == "__main__":
#     from core.logger_config import setup_logging
#     setup_logging()
#
#     # Create dummy documents
#     doc1 = Document(
#         page_content="   This is the first line.\n\nThis line has extra    spaces. \n\n\nThis might be a para-\ngraph that continues.\n\nPage 1\n",
#         metadata={"source": "test.pdf", "page": 1}
#     )
#     doc2 = Document(
#         page_content="\n\n   Another document \t with tabs and \n trailing whitespace.   \n\nFooter Line\n",
#         metadata={"source": "test.pdf", "page": 2}
#     )
#     doc3 = Document(page_content="Header Line\n\nReal content starts here.\n\nFooter Line", metadata={"source": "test.pdf", "page": 3})
#
#     docs_to_clean = [doc1, doc2, doc3]
#
#     logger.info("--- Before Cleaning ---")
#     for i, doc in enumerate(docs_to_clean):
#         logger.info(f"Doc {i+1} Metadata: {doc.metadata}")
#         logger.info(f"Content:\n>>>\n{doc.page_content}\n<<<")
#
#     cleaned_documents = clean_documents(docs_to_clean)
#
#     logger.info("\n--- After Cleaning ---")
#     for i, doc in enumerate(cleaned_documents):
#         logger.info(f"Doc {i+1} Metadata: {doc.metadata}")
#         logger.info(f"Content:\n>>>\n{doc.page_content}\n<<<")
#
#     # Test empty doc
#     empty_doc = Document(page_content="\n   \n", metadata={"source": "empty.txt"})
#     cleaned_empty = clean_documents([empty_doc])
#     logger.info(f"\n--- Cleaning Empty Doc ---")
#     logger.info(f"Resulting docs count: {len(cleaned_empty)}") 