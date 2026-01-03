import yaml
from pydantic import BaseModel
from typing import Dict


class AppSettings(BaseModel):
    title: str
    version: str

class ChromaSettings(BaseModel):
    path: str
    collections: Dict[str, str]

class ModelSettings(BaseModel):
    bert: str
    qwen: str

class Config(BaseModel):
    app: AppSettings
    chroma: ChromaSettings
    models: ModelSettings


def load_config(config_path: str = "config.yaml") -> Config:
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        return Config(**raw_config)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found at: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")


settings = load_config()