import logging
import logging.handlers
from pathlib import Path
from .config_loader import get_config_value, BASE_DIR

def setup_logging():
    """Configures the application's logging."""

    log_level_str = get_config_value("logging.log_level", "INFO").upper()
    log_format = get_config_value("logging.log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file_path_str = get_config_value("logging.log_file", "logs/app.log")
    max_bytes = int(get_config_value("logging.max_bytes", 10485760)) # Default 10MB
    backup_count = int(get_config_value("logging.backup_count", 5))

    log_level = getattr(logging, log_level_str, logging.INFO)

    # Ensure the logs directory exists
    log_file_path = BASE_DIR / log_file_path_str
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level) # Console logs at the configured level

    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level) # File logs at the configured level

    # Get the root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level) # Set root logger level

    # Remove existing handlers (if any, e.g., from libraries) to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add the handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Optionally silence overly verbose library loggers
    logging.getLogger("httpx").setLevel(logging.WARNING) # Example for httpx used by some libs
    logging.getLogger("chromadb").setLevel(logging.WARNING) # ChromaDB can be verbose

    logging.info("Logging configured successfully.")
    logging.info(f"Log level set to: {log_level_str}")
    logging.info(f"Log file path: {log_file_path}")

# Call setup_logging() early in your application's entry point
# For example, in your main app script or __init__.py of a core module.

# Example usage (optional)
# if __name__ == "__main__":
#     setup_logging()
#     logging.debug("This is a debug message.")
#     logging.info("This is an info message.")
#     logging.warning("This is a warning message.")
#     logging.error("This is an error message.")
#     logging.critical("This is a critical message.")

logger = logging.getLogger(__name__)

# Example usage
logger.info("Starting document processing.")
try:
    # ... some operation ...
    logger.debug("Intermediate step successful.")
except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True) # exc_info=True logs traceback 