import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union

def load_env_vars(env_path: Union[str, Path] = ".env"):
    """Reads .env file and sets environment variables if present."""
    p = Path(env_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file and loads .env variables."""
    load_env_vars(".env")
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file.resolve()}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config

class Config:
    """Wrapper class for convenient dot-notation access to configuration options."""
    def __init__(self, config_dict: Dict[str, Any]):
        self._dict = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return self._dict

    def __getitem__(self, item: str):
        return self._dict[item]

def get_config(config_path: Union[str, Path] = "config.yaml") -> Config:
    """Returns a Config object instance."""
    raw_config = load_config(config_path)
    return Config(raw_config)
