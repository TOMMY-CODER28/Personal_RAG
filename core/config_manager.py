import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class RAGConfigManager:
    """Manages RAG system configuration and presets."""
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.current_preset = self.config.get("default_preset", "balanced")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def get_preset(self, preset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific preset."""
        preset_name = preset_name or self.current_preset
        return self.config.get("presets", {}).get(preset_name, {})
    
    def get_retrieval_params(self, preset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get retrieval parameters for a preset."""
        return self.get_preset(preset_name).get("retrieval", {})
    
    def get_chunking_params(self, preset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get chunking parameters for a preset."""
        return self.get_preset(preset_name).get("chunking", {})
    
    def get_context_params(self, preset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get context parameters for a preset."""
        return self.get_preset(preset_name).get("context", {})
    
    def get_generation_params(self, preset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get generation parameters for a preset."""
        return self.get_preset(preset_name).get("generation", {})
    
    def get_system_params(self) -> Dict[str, Any]:
        """Get system-wide parameters."""
        return self.config.get("system", {})
    
    def update_preset(self, preset_name: str, params: Dict[str, Any]) -> bool:
        """Update parameters for a specific preset."""
        try:
            if preset_name not in self.config["presets"]:
                self.config["presets"][preset_name] = {}
            
            for category, values in params.items():
                if category not in self.config["presets"][preset_name]:
                    self.config["presets"][preset_name][category] = {}
                self.config["presets"][preset_name][category].update(values)
            
            self._save_config()
            return True
        except Exception as e:
            logger.error(f"Error updating preset: {e}")
            return False
    
    def _save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def get_available_presets(self) -> List[str]:
        """Get list of available presets."""
        return list(self.config.get("presets", {}).keys())
    
    def set_current_preset(self, preset_name: str) -> bool:
        """Set the current preset to use."""
        if preset_name in self.get_available_presets():
            self.current_preset = preset_name
            return True
        return False 