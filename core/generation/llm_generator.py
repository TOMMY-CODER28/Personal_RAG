import logging
import os
import time
from typing import Dict, Optional, Any
import google.generativeai as genai
from ..config.rag_config_manager import RAGConfigManager

# Assuming config_loader and logger_config are set up
from ..config_loader import get_config_value

logger = logging.getLogger(__name__)

# --- Prompt Templates ---
# System prompt / instruction prompt for technical document RAG
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided technical documentation. 
Respond with accurate, factual information extracted from the documentation context provided below.

- Base your answer solely on the provided context, not on prior knowledge.
- If the context doesn't contain enough information to provide a complete answer, clearly state what's missing.
- If the context contains code examples, explain them clearly and maintain proper formatting.
- Always cite the source documents when providing information (e.g., "According to manual_v1.pdf, page 10...").
- If multiple sources have conflicting information, acknowledge this and explain the differences.
- Maintain a professional, clear, and concise tone appropriate for technical documentation.
- Format your response using Markdown when appropriate (for headings, code blocks, lists, etc.).
"""

# Default template for combining query and context
DEFAULT_RAG_TEMPLATE = """### User Question:
{query}

### Technical Documentation Context:
{context}

### Answer (with citations from the documentation where possible):
"""

def initialize_gemini() -> bool:
    """Initialize the Gemini API with the API key."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return False
        genai.configure(api_key=api_key)
        return True
    except Exception:
        return False

def get_generation_model(model_name: Optional[str] = None):
    """
    Gets the specified Gemini generation model or the default from config.
    
    Args:
        model_name: Optional model name to override config value.
        
    Returns:
        The Gemini model instance or None if initialization fails.
    """
    # Initialize API first (if not already done)
    if not initialize_gemini():
        return None
    
    # Get model name (explicitly provided or from config)
    if not model_name:
        # Override the config value completely - use a known available model
        model_name = "models/gemini-2.5-pro-exp-03-25"  # Updated to use gemini-2.0-pro-exp model
    
    try:
        # Log available models first for debugging
        try:
            available_models = genai.list_models()
            logger.info(f"Available Gemini models: {[m.name for m in available_models]}")
        except Exception as list_err:
            logger.warning(f"Failed to list available models: {list_err}")
        
        # Initialize the model - ensure it has the correct prefix format
        if not model_name.startswith("models/"):
            model_name = "models/" + model_name
            
        logger.info(f"Attempting to use Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)
        logger.info(f"Successfully initialized Gemini model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to get Gemini model '{model_name}': {e}", exc_info=True)
        return None

def generate_response(question: str, context: str) -> Optional[str]:
    """Generate a response using the Gemini model."""
    try:
        config_manager = RAGConfigManager()
        model_params = config_manager.get_model_params()
        generation_params = config_manager.get_generation_params()

        # Get model
        model = genai.GenerativeModel(
            model_name=model_params.get("name", "models/gemini-2.5-pro-exp-03-25")
        )

        # Prepare prompt
        prompt = f"""You are a seasoned technical expert and educator. When answering questions based on user‑provided context, adhere to the following rules:

## 1. Context Assessment
- **Acknowledge Gaps**  
  If the context is missing, incomplete, or ambiguous, state that clearly (e.g., “The context does not include X; here’s how we can proceed…”).
- **Augment with Expertise**  
  Use your own up‑to‑date technical knowledge to fill gaps, ensuring answers remain accurate and actionable.

## 2. Depth & Structure
- **Comprehensive Coverage**  
  Address all facets of the question: theory, practical steps, caveats, and real‑world considerations.
- **Explain “Why” & “How”**  
  Don’t just list commands or code—explain the rationale, trade‑offs, and best practices.
- **Layered Detail**  
  - **Overview**: A brief summary of the solution.  
  - **Step‑by‑Step**: Concrete instructions, subdivided into logical phases.  
  - **Advanced Notes**: Optional deep‑dive, edge cases, or mitigation strategies.

## 3. Illustrations & Examples
- **Code Blocks**: Show syntax‑highlighted snippets for scripts, shell commands, or pseudocode.
- **Diagrams & Tables**: When they clarify complex relationships or workflows, include simple ASCII or Markdown tables.
- **Sample Outputs**: Illustrate expected results, error messages, or debugging sessions.

## 4. Citation & Attribution
- **Synthesize, Don’t Quote**  
  Paraphrase context material in your own words, then explain its relevance.
- **Formal References**  
  Cite source name and page/section (e.g., *The Art of Software Security Assessment*, p. 200).
- **External Authority**  
  When drawing on external standards (e.g., RFCs, OWASP), link or reference them clearly.

## 5. Tone & Style
- **Professional & Conversational**  
  Use concise, clear language but keep it engaging and approachable.
- **Quick Humor**  
  Sprinkle light, clever humor where it eases understanding—never at the expense of clarity.
- **Encouraging & Supportive**  
  Acknowledge learning curves (“This can be tricky at first, but here’s a tip…”).

## 6. Clarity & Readability
- **Markdown Formatting**  
  - Headings (`##`, `###`) for structure.  
  - Bullet/numbered lists for processes.  
  - **Bold** for emphasis.
- **Explicit Callouts**  
  Use notes, warnings, or tips sections to flag pitfalls or best practices.

Your goal: **deliver expert‑level guidance** that’s **thorough**, **well‑documented**, and **a pleasure to read**.

## 7. Response Format
- **Markdown Formatting**  
  - Headings (`##`, `###`) for structure.  
  - Bullet/numbered lists for processes.  
  - **Bold** for emphasis.  

Context:
{context}

Question:
{question}

Answer:"""

        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=generation_params.get("temperature", 0.2),
                top_p=generation_params.get("top_p", 0.95),
                top_k=generation_params.get("top_k", 40),
                max_output_tokens=generation_params.get("max_output_tokens", 4096),
            )
        )

        return response.text if response else None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def classify_query_type_with_llm(question: str) -> Optional[str]:
    """Classify the question type using Gemini LLM: definition, procedure, code, or general."""
    try:
        config_manager = RAGConfigManager()
        model_params = config_manager.get_model_params()
        generation_params = config_manager.get_generation_params()

        model = genai.GenerativeModel(
            model_name=model_params.get("name", "models/Gemini-2.0-Flash-Experimental")
        )

        prompt = (
            "Classify the following question as one of the following types:\n"
            "- definition\n- procedure\n- code\n- general\n\n"
            "Only return the type, nothing else.\n\n"
            f"Question: {question}"
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,  # Deterministic output
                top_p=1.0,
                top_k=1,
                max_output_tokens=10,
            )
        )
        if response and response.text:
            return response.text.strip().lower()
        return None
    except Exception as e:
        logger.error(f"Error classifying query type with LLM: {e}")
        return None

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
#     # Test initialization
#     logger.info("--- Testing Gemini Initialization ---")
#     init_result = initialize_gemini()
#     if init_result:
#         logger.info("Gemini API initialized successfully for testing.")
#     else:
#         logger.error("Failed to initialize Gemini API. Please check your API key in .env file.")
#         sys.exit(1)  # Stop test if initialization fails
# 
#     # Test response generation
#     test_query = "How do I configure the flux capacitor?"
#     
#     # Create a dummy context mimicking what context_assembler would provide
#     test_context = """
# Source: flux_capacitor_manual.pdf (Page: 42)
# Content:
# The flux capacitor requires 1.21 gigawatts of electrical power to operate correctly. To configure it:
# 1. Turn the main dial to the "Flux" position.
# 2. Set the temporal displacement knob to the desired date.
# 3. Ensure the plutonium chamber is properly loaded.
# WARNING: Improper configuration may result in temporal displacement anomalies.
# 
# Source: delorean_maintenance.pdf (Page: 88)
# Content:
# The flux capacitor connects to the vehicle's electrical system via the primary inverter. 
# Regular maintenance should include checking the capacitor's dilithium crystal alignment.
# """
# 
#     logger.info("\n--- Testing Response Generation ---")
#     logger.info(f"Query: '{test_query}'")
#     logger.info(f"Context (preview): '{test_context[:100]}...'")
#     
#     # Test with default parameters
#     response = generate_response(test_query, test_context)
#     
#     if response:
#         logger.info("\nGenerated Response:")
#         logger.info(f">>>\n{response}\n<<<")
#     else:
#         logger.error("Failed to generate response.")
#     
#     # Optional: Test with different parameters
#     logger.info("\n--- Testing with Different Parameters ---")
#     custom_params = {
#         "temperature": 0.8,  # Higher creativity
#         "max_output_tokens": 200  # Shorter response
#     }
#     response_custom = generate_response(
#         "Summarize the flux capacitor configuration steps briefly.",
#         test_context,
#         generation_params=custom_params
#     )
#     
#     if response_custom:
#         logger.info("\nGenerated Response (Custom Parameters):")
#         logger.info(f">>>\n{response_custom}\n<<<")
#     else:
#         logger.error("Failed to generate response with custom parameters.") 