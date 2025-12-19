"""
Configuration manager
"""

import yaml
import json
from typing import Dict, Any
import os


class ConfigManager:
    """Configuration manager for loading and accessing configuration"""
    
    @staticmethod
    def load(file_path: str) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            file_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return yaml.safe_load(f) or {}
            elif file_path.endswith('.json'):
                return json.load(f)
            else:
                # Try YAML first
                try:
                    return yaml.safe_load(f) or {}
                except:
                    return json.load(f)
    
    @staticmethod
    def get_nested(config: Dict, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., 'detection_layer.detectors.statistical')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

