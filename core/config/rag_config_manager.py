import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging

class RAGConfigManager:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(Path(__file__).parent.parent.parent / "config" / "rag_config.yaml")
        self.config = {}
        self._load_config()
        self.current_preset = self.config.get("current_preset", "balanced")
        if self.current_preset not in self.get_available_presets():
            self.current_preset = "balanced" if "balanced" in self.get_available_presets() else self.get_available_presets()[0]

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = self._create_default_config()
            self._save_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "presets": {
                "balanced": {
                    "model": {
                        "name": "models/gemini-2.5-pro-exp-03-25",
                        "provider": "google",
                        "api_version": "v1"
                    },
                    "embedding": {
                        "model": "nomic-embed-text",
                        "provider": "nomic",
                        "dimension": 768,
                        "batch_size": 32
                    },
                    "retrieval": {
                        "similarity_threshold": 0.5,
                        "k_results": 10,
                        "distance_metric": "cosine"
                    },
                    "chunking": {
                        "chunk_size": 1000,
                        "chunk_overlap": 200,
                        "split_by": "sentence"
                    },
                    "context": {
                        "max_chars": 8000,
                        "include_sources": True,
                        "include_relevance": False
                    },
                    "generation": {
                        "temperature": 0.2,
                        "max_output_tokens": 4096,
                        "top_p": 0.95,
                        "top_k": 40
                    }
                }
            },
            "current_preset": "balanced"
        }

    def _save_config(self) -> bool:
        """Save configuration to YAML file."""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f)
            return True
        except Exception:
            return False

    def get_available_presets(self) -> List[str]:
        """Get list of available configuration presets."""
        return list(self.config.get("presets", {}).keys())

    def get_retrieval_params(self, preset: str = None) -> Dict[str, Any]:
        """Get retrieval parameters for specified preset."""
        preset = preset or self.current_preset
        return self.config.get("presets", {}).get(preset, {}).get("retrieval", {})

    def get_chunking_params(self, preset: str = None) -> Dict[str, Any]:
        """Get chunking parameters for specified preset."""
        preset = preset or self.current_preset
        return self.config.get("presets", {}).get(preset, {}).get("chunking", {})

    def get_context_params(self, preset: str = None) -> Dict[str, Any]:
        """Get context parameters for specified preset."""
        preset = preset or self.current_preset
        return self.config.get("presets", {}).get(preset, {}).get("context", {})

    def get_generation_params(self, preset: str = None) -> Dict[str, Any]:
        """Get generation parameters for specified preset."""
        preset = preset or self.current_preset
        return self.config.get("presets", {}).get(preset, {}).get("generation", {})

    def     get_model_params(self, preset: str = None) -> Dict[str, Any]:
        """Get model parameters for specified preset."""
        preset = preset or self.current_preset
        return self.config.get("presets", {}).get(preset, {}).get("model", {})

    def get_embedding_params(self, preset: str = None) -> Dict[str, Any]:
        """Get embedding parameters for specified preset."""
        preset = preset or self.current_preset
        return self.config.get("presets", {}).get(preset, {}).get("embedding", {})

    def update_preset(self, preset: str, params: Dict[str, Any]) -> bool:
        """Update parameters for a preset."""
        logger = logging.getLogger(__name__)
        
        try:
            # Ensure the preset exists
            if preset not in self.config.get("presets", {}):
                self.config.setdefault("presets", {})[preset] = {}
                logger.info(f"Created new preset '{preset}'")
            
            # Update each parameter category if provided
            preset_config = self.config["presets"][preset]
            
            # Handle each nested section separately to maintain structure
            for section, section_params in params.items():
                if section not in preset_config:
                    preset_config[section] = {}
                
                # Update parameters in this section
                preset_config[section].update(section_params)
                logger.info(f"Updated {section} parameters for preset '{preset}'")
            
            # Save the updated config
            success = self._save_config()
            if success:
                logger.info(f"Successfully saved updated configuration for preset '{preset}'")
            else:
                logger.error(f"Failed to save configuration for preset '{preset}'")
            return success
        
        except Exception as e:
            logger.error(f"Error updating preset '{preset}': {e}", exc_info=True)
            return False

    def set_current_preset(self, preset: str) -> bool:
        """Set the current preset."""
        if preset in self.config.get("presets", {}):
            self.current_preset = preset
            self.config["current_preset"] = preset
            return self._save_config()
        return False 