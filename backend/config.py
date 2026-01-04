import yaml
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, Optional

# Load variables from .env file into os.environ
load_dotenv()


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
    semantic_scholar_api_key: Optional[str] = Field(default=None)


def load_config(config_path: str = "config.yaml") -> Config:
    try:
        # 1. Load from YAML
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # 2. Load Secrets from Environment (now populated by .env)
        raw_config["semantic_scholar_api_key"] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

        return Config(**raw_config)
    except FileNotFoundError:
        print(f"⚠️ Config file not found at {config_path}")
        raise
    except Exception as e:
        raise RuntimeError(f"❌ Error loading config: {e}")


settings = load_config()