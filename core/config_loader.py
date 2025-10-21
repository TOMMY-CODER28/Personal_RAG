import yaml
from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Dict, Any
import logging

# Load environment variables from .env file
load_dotenv()

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

logger = logging.getLogger(__name__)

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Loads the YAML configuration file."""
    if not config_path.exists():
        print(f"Warning: Configuration file not found at {config_path}. Using default settings.")
        return {} # Return empty dict or default settings dict
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                return {} # Handle empty config file
            return config
    except yaml.YAMLError as e:
        print(f"Error loading configuration file {config_path}: {e}")
        # Decide how to handle error: raise exception, return default, exit?
        raise # Re-raising the exception for now
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        raise

def get_config_value(key: str, default=None):
    """Retrieves a specific value from the loaded configuration."""
    config = load_config()
    # Use os.getenv to allow overriding config with environment variables
    # Environment variables often take precedence
    env_value = os.getenv(key.upper())
    if env_value is not None:
        # Attempt to cast env var to appropriate type if needed,
        # otherwise return as string
        return env_value

    # Navigate nested keys if necessary (e.g., "database.host")
    keys = key.split('.')
    value = config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        # Key not found in config or config structure issue
        if default is not None:
            return default
        else:
            # Or raise an error if the key is mandatory
            print(f"Warning: Configuration key '{key}' not found and no default provided.")
            return None

def reload_config() -> Dict[str, Any]:
    """
    Reload the configuration from the YAML file.
    Use this when configurations have been updated at runtime.
    """
    global _config
    # Since we don't actually have persistent config caching at module level,
    # this function just does a fresh load from CONFIG_PATH
    _config = load_config(CONFIG_PATH)
    logger.info("Configuration reloaded from file")
    return _config

# Example usage (optional, can be removed or put under if __name__ == "__main__":)
# if __name__ == "__main__":
#     config = load_config()
#     print("Loaded config:", config)
#     db_host = get_config_value("database.host", "localhost")
#     print("Database host:", db_host)
#     api_key_from_env = get_config_value("GEMINI_API_KEY") # Should load from .env via os.getenv 