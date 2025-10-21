import streamlit as st
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import core modules
from core.logger_config import setup_logging
from core.config_loader import get_config_value, BASE_DIR
from core.processing.document_loader import load_document
from core.processing.text_cleaner import clean_documents
from core.processing.chunker import chunk_documents
from core.embedding.embedding_generator import generate_embeddings
from core.vector_db.chroma_manager import add_chunks_to_vector_db, get_or_create_collection, get_chroma_client
from core.retrieval.retriever import embed_query, search_vector_db
from core.retrieval.context_assembler import assemble_context
from core.generation.llm_generator import generate_response, initialize_gemini, classify_query_type_with_llm
from core.config.rag_config_manager import RAGConfigManager

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Technical RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility: Clean response history ---
def clean_response_history():
    """Ensure response_history is a list of dicts with required keys. Remove/fix malformed entries."""
    required_keys = {"question", "response", "timestamp"}
    cleaned = []
    removed_count = 0
    rh = st.session_state.get("response_history", [])
    if not isinstance(rh, list):
        logger.warning("response_history was not a list; resetting.")
        st.session_state.response_history = []
        return
    for entry in rh:
        if isinstance(entry, dict) and required_keys.issubset(entry.keys()):
            cleaned.append(entry)
        else:
            removed_count += 1
            logger.warning(f"Malformed response_history entry removed: {entry}")
    if removed_count > 0:
        logger.info(f"Cleaned {removed_count} malformed entries from response_history.")
    st.session_state.response_history = cleaned

# --- Initialization ---
def initialize_app():
    """Initialize app state and check for necessary configurations."""
    if "initialized" not in st.session_state:
        # Check for API key
        if not os.getenv("GEMINI_API_KEY"):
            st.warning("⚠️ GEMINI_API_KEY not found in environment variables. Make sure your .env file is properly set up.")
        
        # Debug RAGConfigManager loading
        try:
            config_manager = RAGConfigManager()
            logger.info(f"RAGConfigManager initialized. Current preset: {config_manager.current_preset}")
            logger.info(f"Available presets: {config_manager.get_available_presets()}")
        except Exception as e:
            logger.error(f"Error initializing RAGConfigManager: {e}", exc_info=True)
        
        # Initialize Gemini API
        gemini_initialized = initialize_gemini()
        if not gemini_initialized:
            st.error("❌ Failed to initialize Gemini API. Please check your API key and try again.")
        
        # Test ChromaDB connection
        collection = get_or_create_collection()
        if not collection:
            st.warning("⚠️ Could not connect to vector database. Document processing might fail.")
            
        # Set initialization flag and default values
        st.session_state.initialized = True
        st.session_state.processing_queue = []
        st.session_state.processed_files = set()
        st.session_state.collection_name = "technical_docs"
        
        # Check if documents already exist in the database
        # This ensures the UI is aware of existing documents after refresh
        try:
            if collection:
                doc_count = collection.count()
                if doc_count > 0:
                    logger.info(f"Found {doc_count} existing documents in collection. Enabling question interface.")
                    # Add a placeholder document to processed_files to enable the UI
                    st.session_state.processed_files.add("existing_documents")
                    
                    # Get a sample to extract actual document names if possible
                    try:
                        peek_results = collection.peek(5)
                        if peek_results and 'metadatas' in peek_results and peek_results['metadatas']:
                            for metadata in peek_results['metadatas']:
                                if metadata and 'source' in metadata:
                                    source = metadata.get('source', '')
                                    if source:
                                        st.session_state.processed_files.add(source)
                    except Exception as e:
                        logger.warning(f"Could not extract document names from database: {e}")
        except Exception as e:
            logger.error(f"Error checking for existing documents: {e}")
    
    # Always ensure these session variables exist
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = "technical_docs"
    # Clean up response_history if present
    clean_response_history()

# --- Document Processing Functions ---
def process_uploaded_file(uploaded_file, collection_name: str = "technical_docs"):
    """Process a single uploaded file and add to the vector database."""
    # Create temp directory if it doesn't exist
    temp_dir = BASE_DIR / "data" / "raw" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file temporarily
    temp_file_path = temp_dir / uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"Processing {uploaded_file.name}... This may take a while for large documents.")
    
    # 1. Load
    with st.spinner("Loading document..."):
        raw_docs = load_document(temp_file_path)
        if not raw_docs:
            st.error(f"Failed to load document: {uploaded_file.name}")
            return False
        st.success(f"Loaded {len(raw_docs)} pages/sections from {uploaded_file.name}")
    
    # 2. Clean
    with st.spinner("Cleaning document..."):
        cleaned_docs = clean_documents(raw_docs)
        if not cleaned_docs:
            st.warning(f"No documents remaining after cleaning for {uploaded_file.name}")
            return False
        st.success(f"Cleaned {len(cleaned_docs)} pages/sections")
    
    # 3. Chunk
    with st.spinner("Chunking document..."):
        chunks = chunk_documents(cleaned_docs)
        if not chunks:
            st.warning(f"No chunks generated from document: {uploaded_file.name}")
            return False
        st.success(f"Generated {len(chunks)} chunks")
    
    # 4. Generate embeddings
    with st.spinner("Generating embeddings... (this may take a while)"):
        embeddings = generate_embeddings(chunks)
        if not embeddings:
            st.error(f"Failed to generate embeddings for chunks from {uploaded_file.name}")
            return False
        st.success(f"Generated {len(embeddings)} embeddings")
    
    # 5. Add to vector DB
    with st.spinner("Adding to vector database..."):
        success = add_chunks_to_vector_db(chunks, embeddings, collection_name=collection_name)
        if not success:
            st.error(f"Failed to add chunks to vector database for {uploaded_file.name}")
            return False
        st.success(f"Added {len(chunks)} chunks to vector database")
    
    # Make sure session state is updated properly
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    # Add to processed files and explicitly save to session state
    st.session_state.processed_files.add(uploaded_file.name)
    
    # Cleanup temp file
    try:
        os.remove(temp_file_path)
    except Exception as e:
        logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")
    
    return True

def query_system(question: str, collection_name: str = "technical_docs"):
    """Process a user query and return a response using the RAG pipeline."""
    if not question:
        return None
    
    # 1. Embed the query
    with st.spinner("Processing query..."):
        query_embedding = embed_query(question)
        if not query_embedding:
            st.error("Failed to generate embedding for the query.")
            return "Sorry, I encountered an error processing your query."
    
    # 2. Search the vector DB
    with st.spinner("Searching knowledge base..."):
        retrieved_docs = search_vector_db(
            query_embedding, 
            collection_name=collection_name,
            n_results=int(get_config_value("retrieval.k_results", 5))
        )
        if retrieved_docs is None:
            st.error("Vector search failed.")
            return "Sorry, I encountered an error searching the knowledge base."
        
        if not retrieved_docs:
            st.warning("No relevant documents found.")
            return "I couldn't find any relevant information in the technical documentation to answer your question."
    
    # 3. Assemble context
    with st.spinner("Assembling context..."):
        context_string = assemble_context(retrieved_docs)
        if not context_string:
            st.warning("Context assembly resulted in empty string.")
            return "I found some potentially relevant information but couldn't process it properly."
    
    # 4. Generate response
    with st.spinner("Generating response..."):
        response = generate_response(question, context_string)
        if not response:
            st.error("Failed to generate response.")
            return "I found relevant information but encountered an error generating a response."
    
    return response

# --- UI Components ---
def render_sidebar():
    """Render the sidebar UI."""
    st.sidebar.title("Technical RAG System")
    st.sidebar.markdown("---")
    
    # Document Upload Section
    st.sidebar.header("📄 Document Processing")
    
    collection_name = st.sidebar.text_input(
        "Collection Name",
        value=st.session_state.get("collection_name", "technical_docs"),
        help="Name of the vector database collection to use."
    )
    st.session_state.collection_name = collection_name
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload Technical Documents",
        accept_multiple_files=True,
        type=["pdf", "txt"], # Add more formats as supported
        help="Upload technical manuals, documentation, or other text files."
    )
    
    if uploaded_files:
        if st.sidebar.button("Process Documents"):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                if file_name in st.session_state.processed_files:
                    st.sidebar.info(f"{file_name} already processed.")
                    continue
                
                # Process file
                success = process_uploaded_file(uploaded_file, collection_name)
                if success:
                    st.session_state.processed_files.add(file_name)
    
    # About Section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown(
        """
        This is a Retrieval-Augmented Generation (RAG) system for technical documentation.
        
        Upload technical documents, then ask questions to get accurate answers based on the content.
        
        Built with:
        - Google Gemini API
        - ChromaDB
        - LangChain
        - Streamlit
        """
    )

def render_main_content():
    """Render the main content area."""
    st.title("📚 Technical Documentation Assistant")
    st.markdown(
        """
        Ask questions about your technical documentation and get accurate, sourced answers.
        
        Upload your technical manuals and documentation using the sidebar, then ask questions below.
        """
    )
    
    # Query Section
    st.header("🔍 Ask a Question")
    
    # Check if any documents have been processed
    if not st.session_state.get("processed_files"):
        st.warning("No documents have been processed yet. Please upload and process some documents first.")
    
    question = st.text_area(
        "Enter your question about the technical documentation:",
        height=100,
        max_chars=1000,
        help="Be specific with your question for the best results."
    )
    
    if st.button("Submit Question", disabled=not st.session_state.get("processed_files")):
        if not question:
            st.warning("Please enter a question.")
        else:
            collection_name = st.session_state.collection_name
            response = query_system(question, collection_name)
            
            # Create or update response in session state
            if "response_history" not in st.session_state:
                st.session_state.response_history = []
            
            st.session_state.response_history.append({
                "question": question,
                "response": response,
                "timestamp": time.time()
            })
    
    # Display response history (most recent first)
    if "response_history" in st.session_state and st.session_state.response_history:
        st.header("📝 Responses")
        malformed_count = 0
        for i, item in enumerate(reversed(st.session_state.response_history)):
            if not (isinstance(item, dict) and "question" in item and "response" in item and "timestamp" in item):
                malformed_count += 1
                logger.warning(f"Malformed response_history entry at display: {item}")
                continue
            with st.expander(f"Q: {item['question'][:100]}...", expanded=(i == 0)):
                st.markdown(item["response"])
                st.caption(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['timestamp']))}")
        if malformed_count > 0:
            st.warning(f"{malformed_count} malformed response(s) were skipped in the display. See logs for details.")

def render_parameter_tuning():
    """Render the parameter tuning section of the admin dashboard."""
    try:
        st.subheader("⚙️ Parameter Tuning")
        
        # Get config manager
        config_manager = RAGConfigManager()
        current_preset = config_manager.current_preset
        
        # Debug current state
        logger.info(f"Parameter tuning: Current preset is '{current_preset}'. Available presets: {config_manager.get_available_presets()}")
        
        # Preset selector
        available_presets = config_manager.get_available_presets()
        if not available_presets:
            st.error("No configuration presets found. Creating default preset.")
            # Force creation of default preset
            config_manager._create_default_config()
            config_manager._save_config()
            available_presets = config_manager.get_available_presets()
            
        preset_index = 0
        if current_preset in available_presets:
            preset_index = available_presets.index(current_preset)
            
        preset = st.selectbox(
            "Select Configuration Preset",
            options=available_presets,
            index=preset_index,
            help="Choose a preset configuration for the RAG system."
        )
        
        if preset != current_preset:
            logger.info(f"Changing preset from '{current_preset}' to '{preset}'")
            if config_manager.set_current_preset(preset):
                st.success(f"Switched to {preset} preset")
                # Refresh the config manager instance to load updated values
                config_manager = RAGConfigManager()
            else:
                st.error(f"Failed to switch to preset '{preset}'")
        
        # Display current settings for debugging
        with st.expander("Debug Information"):
            st.code(f"""
Current preset: {preset}
Available presets: {available_presets}
Retrieval params: {config_manager.get_retrieval_params(preset)}
Chunking params: {config_manager.get_chunking_params(preset)}
Context params: {config_manager.get_context_params(preset)}
Generation params: {config_manager.get_generation_params(preset)}
            """)
            
        # Create a button to add a new preset
        col1, col2 = st.columns([3, 1])
        with col1:
            new_preset_name = st.text_input("Create New Preset", placeholder="Enter new preset name")
        with col2:
            if st.button("Create Preset", key="create_preset_btn") and new_preset_name:
                if new_preset_name in config_manager.get_available_presets():
                    st.warning(f"Preset '{new_preset_name}' already exists.")
                else:
                    # Create new preset based on current one
                    params = {
                        "model": config_manager.get_model_params(preset),
                        "embedding": config_manager.get_embedding_params(preset),
                        "retrieval": config_manager.get_retrieval_params(preset),
                        "chunking": config_manager.get_chunking_params(preset),
                        "context": config_manager.get_context_params(preset),
                        "generation": config_manager.get_generation_params(preset)
                    }
                    if config_manager.update_preset(new_preset_name, params):
                        st.success(f"Created new preset '{new_preset_name}'")
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to create preset '{new_preset_name}'")
        
        # Create tabs for different parameter categories
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Model", "Embedding", "Retrieval", "Chunking", "Context", "Generation"
        ])
        
        # Fetch all parameters for the selected preset
        model_params = config_manager.get_model_params(preset)
        embedding_params = config_manager.get_embedding_params(preset)
        retrieval_params = config_manager.get_retrieval_params(preset)
        chunking_params = config_manager.get_chunking_params(preset)
        context_params = config_manager.get_context_params(preset)
        generation_params = config_manager.get_generation_params(preset)
        
        changes_made = False
        
        with tab1:
            st.markdown("### Model Parameters")
            model_name = st.text_input(
                "Model Name",
                value=model_params.get("name", "models/gemini-2.5-pro-exp-03-25"),
                help="Name of the LLM model to use."
            )
            
            model_provider = st.selectbox(
                "Provider",
                options=["google", "openai", "anthropic", "local", "other"],
                index=["google", "openai", "anthropic", "local", "other"].index(
                    model_params.get("provider", "google")
                ) if model_params.get("provider", "google") in ["google", "openai", "anthropic", "local", "other"] else 0,
                help="Provider of the model."
            )
        
        with tab2:
            st.markdown("### Embedding Parameters")
            embedding_model = st.text_input(
                "Embedding Model",
                value=embedding_params.get("model", "nomic-embed-text"),
                help="Name of the embedding model to use."
            )
            
            embedding_provider = st.selectbox(
                "Provider",
                options=["ollama", "nomic", "google", "openai", "huggingface"],
                index=["ollama", "nomic", "google", "openai", "huggingface"].index(
                    embedding_params.get("provider", "ollama")
                ) if embedding_params.get("provider", "ollama") in ["ollama", "nomic", "google", "openai", "huggingface"] else 0,
                help="Provider of the embedding model."
            )
            
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value=embedding_params.get("ollama_base_url", "http://localhost:11434"),
                help="Base URL for Ollama server (only if using Ollama provider)."
            )
        
        with tab3:
            st.markdown("### Retrieval Parameters")
            similarity = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(retrieval_params.get("similarity_threshold", 0.5)),
                step=0.1,
                help="Lower values return more results but may be less relevant."
            )
            
            k_results = st.slider(
                "Number of Results",
                min_value=1,
                max_value=50,
                value=int(retrieval_params.get("k_results", 10)),
                step=1,
                help="Number of documents to retrieve for context."
            )
            
            distance_metric = st.selectbox(
                "Distance Metric",
                options=["cosine", "euclidean", "dot_product"],
                index=["cosine", "euclidean", "dot_product"].index(
                    retrieval_params.get("distance_metric", "cosine")
                ) if retrieval_params.get("distance_metric", "cosine") in ["cosine", "euclidean", "dot_product"] else 0,
                help="Method to calculate similarity between embeddings."
            )
        
        with tab4:
            st.markdown("### Chunking Parameters")
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=5000,
                value=int(chunking_params.get("chunk_size", 1000)),
                step=100,
                help="Size of text chunks for processing."
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=1000,
                value=int(chunking_params.get("chunk_overlap", 200)),
                step=50,
                help="Number of characters to overlap between chunks."
            )
            
            split_by = st.selectbox(
                "Split By",
                options=["character", "sentence", "paragraph"],
                index=["character", "sentence", "paragraph"].index(
                    chunking_params.get("split_by", "sentence")
                ),
                help="Method to split text into chunks."
            )
        
        with tab5:
            st.markdown("### Context Parameters")
            max_chars = st.slider(
                "Max Context Length",
                min_value=1000,
                max_value=20000,
                value=int(context_params.get("max_chars", 8000)),
                step=1000,
                help="Maximum number of characters in assembled context."
            )
            
            include_sources = st.checkbox(
                "Include Sources",
                value=context_params.get("include_sources", True),
                help="Include source information in context."
            )
            
            include_relevance = st.checkbox(
                "Include Relevance Scores",
                value=context_params.get("include_relevance", False),
                help="Include similarity scores in context."
            )
        
        with tab6:
            st.markdown("### Generation Parameters")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(generation_params.get("temperature", 0.2)),
                step=0.1,
                help="Higher values make output more random, lower values more deterministic."
            )
            
            max_tokens = st.slider(
                "Max Output Tokens",
                min_value=512,
                max_value=8192,
                value=int(generation_params.get("max_output_tokens", 4096)),
                step=512,
                help="Maximum length of generated response."
            )
            
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=float(generation_params.get("top_p", 0.95)),
                step=0.05,
                help="Nucleus sampling parameter."
            )
            
            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=100,
                value=int(generation_params.get("top_k", 40)),
                step=1,
                help="Number of highest probability tokens to consider."
            )
        
        # Save button
        if st.button("Save Changes", type="primary"):
            new_params = {
                "model": {
                    "name": model_name,
                    "provider": model_provider
                },
                "embedding": {
                    "model": embedding_model,
                    "provider": embedding_provider,
                    "ollama_base_url": ollama_base_url if embedding_provider == "ollama" else None
                },
                "retrieval": {
                    "similarity_threshold": similarity,
                    "k_results": k_results,
                    "distance_metric": distance_metric
                },
                "chunking": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "split_by": split_by
                },
                "context": {
                    "max_chars": max_chars,
                    "include_sources": include_sources,
                    "include_relevance": include_relevance
                },
                "generation": {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k
                }
            }
            
            # Clean up embedding params if not Ollama
            if embedding_provider != "ollama":
                new_params["embedding"].pop("ollama_base_url", None)
            
            logger.info(f"Saving updated parameters for preset '{preset}'")
            if config_manager.update_preset(preset, new_params):
                st.success(f"Parameters for preset '{preset}' saved successfully!")
                changes_made = True
            else:
                st.error(f"Failed to save parameters for preset '{preset}'")
            
        # Add a delete button for presets (except for built-in ones)
        if preset not in ["balanced", "precise", "comprehensive"]:
            if st.button("Delete Preset", type="secondary"):
                # Switch to balanced preset first if deleting current preset
                if preset == current_preset:
                    config_manager.set_current_preset("balanced" if "balanced" in config_manager.get_available_presets() else config_manager.get_available_presets()[0])
                    
                # Remove the preset from config
                if "presets" in config_manager.config and preset in config_manager.config["presets"]:
                    del config_manager.config["presets"][preset]
                    if config_manager._save_config():
                        st.success(f"Deleted preset '{preset}'")
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to delete preset '{preset}'")
        
        # If changes were made, show a button to reload the app
        if changes_made:
            if st.button("Reload App to Apply Changes"):
                st.experimental_rerun()
    except Exception as e:
        logger.error(f"Error in parameter tuning: {e}", exc_info=True)
        st.error(f"Error loading parameter tuning interface: {str(e)}")
        st.code(f"Exception details: {e}", language="python")

def render_database_management():
    """Render the database management section of the admin dashboard."""
    # Get ChromaDB client
    client = get_chroma_client()
    if not client:
        st.error("Could not connect to ChromaDB. Admin dashboard unavailable.")
        return
    
    # Get all collection names
    try:
        collection_names = client.list_collections()
        if not collection_names:
            st.warning("No collections found in the database.")
            return
    except Exception as e:
        st.error(f"Error retrieving collections: {e}")
        return
    
    # Collection selector
    selected_collection = st.selectbox(
        "Select Collection", 
        collection_names,
        index=collection_names.index(st.session_state.collection_name) if st.session_state.collection_name in collection_names else 0
    )
    
    # Update the session state with the selected collection
    st.session_state.collection_name = selected_collection
    
    # Get selected collection
    collection = client.get_collection(selected_collection)
    
    # Display collection statistics
    st.subheader("Collection Statistics")
    try:
        count = collection.count()
        st.metric("Documents in Collection", count)
        
        if count > 0:
            # Use peek with a small limit to avoid large data retrieval
            peek_results = collection.peek(limit=5) 
            
            # Display collection metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                # Check if 'ids' key exists and its list is not empty
                doc_count_display = len(peek_results['ids'][0]) if peek_results and 'ids' in peek_results and isinstance(peek_results.get('ids'), list) and peek_results['ids'] and len(peek_results['ids'][0]) > 0 else 0
                st.metric("Document Count (Sample)", doc_count_display)
            with col2:
                # Check if 'embeddings' key exists, its list is not empty, and the first embedding is not empty
                emb_dim = "N/A"
                if peek_results and 'embeddings' in peek_results and isinstance(peek_results.get('embeddings'), list) and peek_results['embeddings']:
                     # Chroma might return [[emb1, emb2,...]] structure
                     embeddings_list = peek_results['embeddings'][0]
                     if isinstance(embeddings_list, list) and len(embeddings_list) > 0 and embeddings_list[0] is not None:
                          try:
                              # Get dimension from the first actual embedding vector
                              emb_dim = len(embeddings_list[0])
                          except TypeError:
                               emb_dim = "Error"
                st.metric("Embedding Dimension", emb_dim)
            with col3:
                 # Check if 'metadatas' key exists, its list is not empty, and the first metadata dict is not empty
                 metadata_fields_count = 0
                 if peek_results and 'metadatas' in peek_results and isinstance(peek_results.get('metadatas'), list) and peek_results['metadatas']:
                     # Chroma might return [[meta1, meta2,...]]
                     metadatas_list = peek_results['metadatas'][0]
                     if isinstance(metadatas_list, list) and len(metadatas_list) > 0 and isinstance(metadatas_list[0], dict):
                          metadata_fields_count = len(metadatas_list[0].keys())
                 st.metric("Metadata Fields", metadata_fields_count)
            
            # Collection management
            st.subheader("Collection Management")
            
            # Delete collection button with confirmation
            with st.expander("Danger Zone"):
                st.warning("Deleting a collection will permanently remove all its documents and cannot be undone.")
                delete_col1, delete_col2 = st.columns([3, 1])
                with delete_col1:
                    delete_confirmation = st.text_input(
                        f"Type '{selected_collection}' to confirm deletion",
                        key=f"delete_{selected_collection}"
                    )
                with delete_col2:
                    if st.button("Delete Collection", type="primary"):
                        if delete_confirmation == selected_collection:
                            try:
                                client.delete_collection(selected_collection)
                                st.success(f"Collection '{selected_collection}' deleted successfully!")
                                st.session_state.pop("collection_name", None)
                                if "processed_files" in st.session_state:
                                    st.session_state.processed_files = set()
                                time.sleep(2)
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error deleting collection: {e}")
                        else:
                            st.error("Confirmation text doesn't match collection name. Collection not deleted.")
    except Exception as e:
        logger.error(f"Error retrieving collection data in admin dashboard: {e}", exc_info=True)
        st.error(f"Error displaying collection data: {e}")

def render_admin_dashboard():
    """Render the admin dashboard for vector DB management."""
    
    st.header("🔧 Admin Dashboard")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Database Management", "Parameter Tuning"])
    
    with tab1:
        # Existing database management code...
        render_database_management()
    
    with tab2:
        # New parameter tuning section
        render_parameter_tuning()

# --- Main App Function ---
def main():
    """Main application entry point."""
    # Initialize
    initialize_app()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📚 Ask Questions", "🔧 Admin Dashboard"])
    
    with tab1:
        render_main_content()
    
    with tab2:
        render_admin_dashboard()
    
    # Sidebar is always visible
    render_sidebar()

if __name__ == "__main__":
    main() 