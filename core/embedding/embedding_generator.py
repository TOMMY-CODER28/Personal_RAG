import logging
import os
from typing import List, Optional

# Remove direct Nomic library imports
# import numpy as np
# import nomic

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings # Add OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config_loader import get_config_value, BASE_DIR
from ..config.rag_config_manager import RAGConfigManager # Assuming this exists and works

logger = logging.getLogger(__name__)

_embedding_model_instance: Optional[Embeddings] = None

def get_embedding_model() -> Optional[Embeddings]:
    """
    Loads and returns the configured embedding model instance.
    Handles HuggingFace, Gemini (Google), and Ollama providers.
    Caches the model instance globally.
    """
    global _embedding_model_instance
    if _embedding_model_instance is not None:
        logger.debug("Returning cached embedding model instance.")
        return _embedding_model_instance

    # Use RAGConfigManager to get parameters
    try:
        config_manager = RAGConfigManager()
        embedding_params = config_manager.get_embedding_params() # Get params for current preset
        model_provider = embedding_params.get("provider", "ollama").lower() # Default to ollama
        model_name = embedding_params.get("model", "nomic-embed-text") # Default to nomic
        ollama_base_url = embedding_params.get("ollama_base_url", "http://localhost:11434") # Default Ollama URL
        cache_folder = BASE_DIR / get_config_value("embedding.cache_folder", "data/processed/embedding_cache")
    except Exception as config_e:
        logger.error(f"Failed to load embedding config via RAGConfigManager: {config_e}", exc_info=True)
        # Fallback to older get_config_value method if RAGConfigManager fails or isn't fully integrated
        logger.warning("Falling back to get_config_value for embedding parameters.")
        model_provider = get_config_value("embedding.provider", "ollama").lower()
        model_name = get_config_value("embedding.model_name", "nomic-embed-text")
        ollama_base_url = get_config_value("embedding.ollama_base_url", "http://localhost:11434")
        cache_folder = BASE_DIR / get_config_value("embedding.cache_folder", "data/processed/embedding_cache")

    logger.info(f"Initializing embedding model. Provider: {model_provider}, Name: {model_name}")

    try:
        if model_provider == "ollama":
            logger.info(f"Using Ollama embeddings. Model: {model_name}, Base URL: {ollama_base_url}")
            _embedding_model_instance = OllamaEmbeddings(
                model=model_name,
                base_url=ollama_base_url
            )
            # Test connection during initialization
            try:
                 _embedding_model_instance.embed_query("Test query")
                 logger.info("Successfully connected to Ollama and tested embedding.")
            except Exception as ollama_conn_err:
                 logger.error(f"Failed to connect to Ollama or embed test query: {ollama_conn_err}", exc_info=True)
                 logger.error("Ensure Ollama server is running, reachable at {ollama_base_url}, and the model '{model_name}' is pulled.")
                 _embedding_model_instance = None # Reset instance on connection failure
                 return None

        elif model_provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found.")
                return None
            _embedding_model_instance = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            logger.info(f"Initialized GoogleGenerativeAIEmbeddings with model: {model_name}")

        elif model_provider == "huggingface":
            cache_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache folder for HuggingFace models: {cache_folder}")
            _embedding_model_instance = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=str(cache_folder),
            )
            logger.info(f"Initialized HuggingFaceEmbeddings with model: {model_name}")

        else:
            logger.error(f"Unsupported embedding provider: {model_provider}")
            return None

        return _embedding_model_instance

    except ImportError as e:
        logger.error(f"Failed to import embedding model dependencies for {model_provider}: {e}")
        if model_provider == 'ollama':
             logger.error("Ensure 'langchain-community' is installed: pip install langchain-community")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {model_name} from {model_provider}: {e}", exc_info=True)
        return None

def generate_embeddings(texts: List[str], batch_size: Optional[int] = None) -> Optional[List[List[float]]]:
    """
    Generate embeddings for a list of texts using the configured embedding model.
    Handles batching via the LangChain embedding model's interface.

    Args:
        texts: List of text strings to embed.
        batch_size: (Currently unused, LangChain models handle internal batching) Optional batch size.

    Returns:
        List of embeddings or None if generation fails.
    """
    if not texts:
        logger.warning("generate_embeddings called with empty text list.")
        return []

    embedding_model = get_embedding_model()
    if not embedding_model:
        logger.error("Failed to get embedding model instance. Cannot generate embeddings.")
        return None

    logger.info(f"Generating embeddings for {len(texts)} texts using {type(embedding_model).__name__}...")

    try:
        # LangChain's embed_documents method handles batching efficiently for most models
        embeddings = embedding_model.embed_documents(texts)

        if not embeddings or len(embeddings) != len(texts):
            logger.error(f"Embedding generation failed or returned incorrect number of embeddings. Expected {len(texts)}, got {len(embeddings) if embeddings else 0}.")
            return None

        logger.info(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings

    except Exception as e:
        # Log specific error from the embedding model if possible
        logger.error(f"Error during embedding generation with {type(embedding_model).__name__}: {e}", exc_info=True)
        return None

# Keep the example usage commented out or remove it
# if __name__ == "__main__":
#    ... 