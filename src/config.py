"""
Configuration loader for ChessDrum.
Loads and manages settings from config.json.
"""
import json
import os
from typing import Any, Dict

DEFAULT_CONFIG = {
    "project": {
        "name": "ChessDrum",
        "version": "0.2.0"
    },
    "audio": {
        "enabled": True,
        "mode": "samples",
        "kit": "classic",
        "sample_rate": 44100,
        "buffer_size": 512,
        "samples": {
            "kick": None,
            "snare": None,
            "hihat": None,
            "clap": None
        }
    },
    "midi": {
        "enabled": False,
        "port_name": "ChessDrum",
        "notes": {
            "kick": 36,
            "snare": 38,
            "clap": 39,
            "hihat": 42
        }
    },
    "filter": {
        "enabled": True,
        "resonance": 2.5,
        "min_freq": 150,
        "max_freq": 8000,
        "type": "resonant"
    },
    "sequencer": {
        "default_bpm": 120,
        "min_bpm": 30,
        "max_bpm": 300,
        "steps": 16
    },
    "camera": {
        "enabled": False,
        "device_id": 0,
        "width": 640,
        "height": 480,
        "brightness": 0,
        "contrast": 1.0,
        "manual_corners": None,
        "bpm_min_distance": 80,
        "bpm_max_distance": 880
    },
    "ui": {
        "cell_size": 60,
        "window_title": "ChessDrum 🎵",
        "theme": "dark"
    }
}


class Config:
    """Configuration manager for ChessDrum."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from config.json or use defaults."""
        config_path = self._find_config_file()
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self._config = json.load(f)
                print(f"✓ Config loaded from {config_path}")
            except Exception as e:
                print(f"⚠ Error loading config: {e}, using defaults")
                self._config = DEFAULT_CONFIG.copy()
        else:
            self._config = DEFAULT_CONFIG.copy()
            print("✓ Using default configuration")
    
    def _find_config_file(self) -> str:
        """Find config.json in various locations."""
        # Check these paths in order
        paths = [
            os.path.join(os.getcwd(), 'config.json'),
            os.path.join(os.path.dirname(__file__), '..', 'config.json'),
            os.path.join(os.path.dirname(__file__), 'config.json'),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        return None
    
    def get(self, *keys, default=None) -> Any:
        """
        Get a config value by path.
        
        Example: config.get('audio', 'enabled') returns True
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys_and_value):
        """
        Set a config value by path.
        
        Example: config.set('audio', 'enabled', False)
        """
        if len(keys_and_value) < 2:
            return
        
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
        
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save current config to file."""
        if path is None:
            path = self._find_config_file() or 'config.json'
        
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=4)
        print(f"✓ Config saved to {path}")
    
    @property
    def audio(self) -> Dict:
        return self._config.get('audio', {})
    
    @property
    def midi(self) -> Dict:
        return self._config.get('midi', {})
    
    @property
    def filter(self) -> Dict:
        return self._config.get('filter', {})
    
    @property
    def sequencer(self) -> Dict:
        return self._config.get('sequencer', {})
    
    @property
    def camera(self) -> Dict:
        return self._config.get('camera', {})
    
    @property
    def ui(self) -> Dict:
        return self._config.get('ui', {})


# Global config instance
config = Config()
