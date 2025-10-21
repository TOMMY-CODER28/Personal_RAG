import logging
from typing import List
from langchain_core.documents import Document

# Assuming config_loader and logger_config are set up
from ..config_loader import get_config_value

logger = logging.getLogger(__name__)

# --- Constants for formatting ---
DEFAULT_CONTEXT_SEPARATOR = "\n\n---\n\n" # Separator between document chunks
DEFAULT_CHUNK_TEMPLATE = """
Source: {source} (Page: {page})
Content:
{content}
"""
# A simpler template if metadata is less reliable or needed
# DEFAULT_CHUNK_TEMPLATE = "{content}"

# Rough estimate for token limit - adjust based on the specific Gemini model used
# Gemini Pro has a large context window (e.g., 32k tokens), but we should still be mindful.
# Let's use a character limit as a proxy for now, as token counting requires a tokenizer.
# Average ~4 chars per token. 30k tokens ~ 120k chars. Let's aim lower initially.
DEFAULT_MAX_CONTEXT_CHARS = int(get_config_value("generation.max_context_chars", 8000)) # Default to 8k chars

def clean_source_reference(metadata: dict) -> str:
    """Clean up source references to be more readable."""
    if not metadata:
        return "Unknown Source"
    
    source = metadata.get('source', '')
    page = metadata.get('page', '')
    
    if not source:
        return "Unknown Source"
    
    # Extract just the filename from the path
    try:
        from pathlib import Path
        filename = Path(source).name
        # Remove any temp directory references
        if 'temp' in filename:
            filename = filename.split('temp')[-1].lstrip('\\/')
        # Clean up common PDF naming patterns
        filename = (filename
                   .replace('dokumen.pub_', '')
                   .replace('-1593277598-9781593277598-j-6513629', '')
                   .replace('.pdf', '')
                   .replace('-', ' ')
                   .title())
    except:
        filename = source
    
    # Add page reference if available
    if page:
        return f"{filename} (Page {page})"
    return filename

def assemble_context(
    retrieved_docs: List[Document],
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    chunk_template: str = DEFAULT_CHUNK_TEMPLATE,
    separator: str = DEFAULT_CONTEXT_SEPARATOR
) -> str:
    """
    Assembles a context string from a list of retrieved documents, respecting a character limit.

    Args:
        retrieved_docs: List of LangChain Document objects, sorted by relevance (most relevant first).
        max_chars: The maximum number of characters allowed for the assembled context.
        chunk_template: A format string for each chunk. Must contain placeholders for
                        metadata keys (e.g., {source}, {page}) and {content}.
        separator: The string used to separate formatted chunks in the final context.

    Returns:
        A single string containing the formatted context, truncated if necessary.
    """
    if not retrieved_docs:
        logger.warning("No documents provided for context assembly.")
        return ""

    assembled_context = ""
    current_char_count = 0
    docs_included_count = 0

    logger.info(f"Assembling context from {len(retrieved_docs)} retrieved documents (max chars: {max_chars}).")

    for i, doc in enumerate(retrieved_docs):
        try:
            # Clean up the source reference
            source_ref = clean_source_reference(doc.metadata)

            # Format the current chunk using the template
            formatted_chunk = chunk_template.format(
                source=source_ref,
                page=doc.metadata.get("page", "N/A"),
                content=doc.page_content,
            )
        except KeyError as e:
            logger.warning(f"Metadata key {e} missing in chunk {i} from source '{source_ref}', skipping this chunk for context assembly.")
            continue
        except Exception as e:
             logger.warning(f"Error formatting chunk {i} from source '{source_ref}': {e}", exc_info=True)
             continue

        # Calculate potential new length
        # Add separator length only if context is not empty
        separator_len = len(separator) if assembled_context else 0
        potential_new_length = current_char_count + separator_len + len(formatted_chunk)

        # Check if adding this chunk exceeds the limit
        if potential_new_length <= max_chars:
            if assembled_context:
                assembled_context += separator
            assembled_context += formatted_chunk
            current_char_count = len(assembled_context) # Recalculate length accurately
            docs_included_count += 1
            logger.debug(f"Added chunk {i+1} (Source: {source_ref}). Current chars: {current_char_count}/{max_chars}")
        else:
            logger.info(f"Reached character limit ({current_char_count}/{max_chars}). Stopping context assembly after {docs_included_count} documents.")
            break # Stop adding chunks once limit is reached

    if not assembled_context:
         logger.warning("Context assembly resulted in an empty string (possibly due to errors or all chunks exceeding limit).")

    return assembled_context

# Example Usage (optional - for testing this module directly)
# if __name__ == "__main__":
#     import sys
#     from pathlib import Path
#     # Add project root to sys.path
#     project_root = Path(__file__).resolve().parent.parent.parent
#     if str(project_root) not in sys.path:
#         sys.path.append(str(project_root))
#
#     from core.logger_config import setup_logging
#     setup_logging()
#
#     # Create dummy retrieved documents
#     doc1 = Document(page_content="This is the most relevant content about topic A.", metadata={"source": "manual_v1.pdf", "page": 10, "retrieval_distance": 0.1})
#     doc2 = Document(page_content="Slightly less relevant info, also about topic A.", metadata={"source": "manual_v1.pdf", "page": 11, "retrieval_distance": 0.2})
#     doc3 = Document(page_content="Content from a different source, maybe related.", metadata={"source": "notes.txt", "page": 1, "retrieval_distance": 0.3})
#     doc4 = Document(page_content="This content is very long and will likely exceed the character limit if added after the others. It contains many words to fill space and test the truncation logic effectively. We keep adding text here to ensure it's substantial enough.", metadata={"source": "appendix.pdf", "page": 5, "retrieval_distance": 0.4})
#     doc5 = Document(page_content="This content should not be included if the limit is reached.", metadata={"source": "manual_v1.pdf", "page": 12, "retrieval_distance": 0.5})
#
#     dummy_retrieved_docs = [doc1, doc2, doc3, doc4, doc5]
#
#     logger.info("--- Testing Context Assembly (Default Limit) ---")
#     # Use a smaller limit for easier testing
#     test_limit = 350
#     context = assemble_context(dummy_retrieved_docs, max_chars=test_limit)
#     logger.info(f"Assembled Context (Limit: {test_limit} chars, Actual: {len(context)} chars):\n>>>\n{context}\n<<<")
#
#     logger.info("\n--- Testing Context Assembly (Larger Limit) ---")
#     test_limit_large = 1000
#     context_large = assemble_context(dummy_retrieved_docs, max_chars=test_limit_large)
#     logger.info(f"Assembled Context (Limit: {test_limit_large} chars, Actual: {len(context_large)} chars):\n>>>\n{context_large}\n<<<")
#
#     logger.info("\n--- Testing Context Assembly (Empty Input) ---")
#     context_empty = assemble_context([])
#     logger.info(f"Assembled Context (Empty Input):\n>>>\n{context_empty}\n<<<")
#
#     logger.info("\n--- Testing Context Assembly (Custom Template) ---")
#     custom_template = "Source Document: {source}\n{content}"
#     context_custom = assemble_context(dummy_retrieved_docs[:2], max_chars=500, chunk_template=custom_template)
#     logger.info(f"Assembled Context (Custom Template, Limit: 500 chars, Actual: {len(context_custom)} chars):\n>>>\n{context_custom}\n<<<") 