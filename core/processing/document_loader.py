import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF

# Using PyPDFLoader from langchain_community as a starting point
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader, TextLoader
from langchain_core.documents import Document # LangChain's Document object

# Get logger instance from the configured logger
logger = logging.getLogger(__name__)

# Define supported extensions and their loaders
# We can expand this dictionary later for EPUB, DOCX, etc.
SUPPORTED_LOADERS: Dict[str, Any] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    # Add more loaders here, e.g.:
    # ".epub": EPubLoader, # Requires installing 'ebooklib' and potentially a custom LangChain loader
    # ".html": UnstructuredHTMLLoader, # Requires 'unstructured' and potentially 'beautifulsoup4'
    # For general unstructured files (docx, pptx, etc.) - requires 'unstructured' library
    # Note: 'unstructured' can have complex dependencies. Start simple.
    # ".docx": UnstructuredFileLoader,
    # ".pptx": UnstructuredFileLoader,
}

def load_pdf_with_pymupdf(file_path: str) -> Optional[List[Document]]:
    """Loads text from a PDF using PyMuPDF, ignoring images."""
    docs = []
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text("text") # Extract only text
            if page_text.strip():
                metadata = {"source": file_path, "page": page_num + 1}
                docs.append(Document(page_content=page_text.strip(), metadata=metadata))
        pdf_document.close()
        return docs
    except Exception as e:
        logger.error(f"Error loading PDF {file_path} with PyMuPDF: {e}", exc_info=True)
        return None

def load_document(file_path: str | Path) -> Optional[List[Document]]:
    """
    Loads a document from the given file path using appropriate LangChain loaders,
    handling potential encoding issues for text files and using PyMuPDF for PDFs.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"File not found: {path}")
        return None

    file_extension = path.suffix.lower()
    logger.info(f"Attempting to load document: {path} (type: {file_extension})")

    loader = None
    documents = None

    try:
        if file_extension == ".pdf":
            # Use PyMuPDF for potentially better handling
            logger.debug(f"Using PyMuPDF loader for {path}")
            documents = load_pdf_with_pymupdf(str(path))
            if documents is None:
                # Optional: Fallback to PyPDFLoader if PyMuPDF fails?
                # logger.warning(f"PyMuPDF failed for {path}, trying PyPDFLoader...")
                # loader = PyPDFLoader(str(path), extract_images=False)
                # documents = loader.load()
                pass # Stick with PyMuPDF failure for now

        elif file_extension == ".txt":
            logger.debug(f"Using TextLoader for {path}")
            # TextLoader needs encoding specified sometimes
            try:
                loader = TextLoader(str(path), encoding='utf-8')
                documents = loader.load()
            except Exception as enc_error:
                logger.warning(f"UTF-8 decoding failed for {path}, trying fallback encoding: {enc_error}")
                try:
                    fallback_encoding = 'latin-1'
                    loader = TextLoader(str(path), encoding=fallback_encoding)
                    documents = loader.load()
                    logger.info(f"Successfully loaded {path} with encoding {fallback_encoding}")
                except Exception as fallback_error:
                    logger.error(f"Failed to load {path} with fallback encoding: {fallback_error}", exc_info=True)
                    return None
        # Add elif blocks for other supported types (e.g., .docx, .html)
        # elif file_extension == ".docx":
            # from langchain_community.document_loaders import Docx2txtLoader
            # loader = Docx2txtLoader(str(path))
            # documents = loader.load()

        else:
            logger.warning(f"Unsupported file type: {file_extension} for {path}. Trying UnstructuredFileLoader as fallback.")
            # Fallback for other types - might need 'unstructured' library
            try:
                from langchain_community.document_loaders import UnstructuredFileLoader
                loader = UnstructuredFileLoader(str(path), mode="elements")
                documents = loader.load()
            except ImportError:
                logger.error("The 'unstructured' library is required for this file type. pip install unstructured")
                return None
            except Exception as unstruct_error:
                logger.error(f"UnstructuredFileLoader failed for {path}: {unstruct_error}", exc_info=True)
                return None

        # Check if loading was successful (documents should be a list)
        if documents is None:
             logger.error(f"Loader failed to produce documents for {path}")
             return None

        # Add sequence numbers if they weren't added by the loader
        # This helps in creating unique IDs later if needed
        for i, doc in enumerate(documents):
            if "chunk_sequence_number" not in doc.metadata:
                 doc.metadata["chunk_sequence_number"] = i

        logger.info(f"Successfully loaded {len(documents)} sections/pages from {path.name}")
        return documents

    except Exception as e:
        logger.error(f"Failed to load or process document {path}: {e}", exc_info=True)
        return None

# Example Usage (optional - for testing this module directly)
# if __name__ == "__main__":
#     from core.logger_config import setup_logging
#     from core.config_loader import get_config_value, BASE_DIR
#
#     setup_logging() # Configure logging first
#
#     # Create a dummy PDF or TXT file in data/raw for testing
#     dummy_file_path = BASE_DIR / get_config_value("data_paths.raw_ebooks", "data/raw") / "dummy_test.txt"
#     dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
#     if not dummy_file_path.exists():
#         with open(dummy_file_path, "w", encoding="utf-8") as f:
#             f.write("This is the first line.\n")
#             f.write("This is the second line, a bit longer.\n")
#             f.write("Page 2 content maybe?\n")
#             f.write("Final line.")
#         logger.info(f"Created dummy file: {dummy_file_path}")
#
#     loaded_docs = load_document(dummy_file_path)
#
#     if loaded_docs:
#         logger.info(f"Loaded {len(loaded_docs)} documents.")
#         for i, doc in enumerate(loaded_docs):
#             logger.info(f"--- Document {i+1} ---")
#             logger.info(f"Content Preview: {doc.page_content[:100]}...") # Show first 100 chars
#             logger.info(f"Metadata: {doc.metadata}")
#     else:
#         logger.warning("Document loading failed.")
#
#     # Test with a non-existent file
#     logger.info("--- Testing non-existent file ---")
#     load_document("non_existent_file.pdf")
#
#     # Test with an unsupported file type (create a dummy .xyz file)
#     logger.info("--- Testing unsupported file type ---")
#     unsupported_file = BASE_DIR / get_config_value("data_paths.raw_ebooks", "data/raw") / "dummy.xyz"
#     if not unsupported_file.exists():
#         with open(unsupported_file, "w") as f: f.write("dummy")
#     load_document(unsupported_file) 