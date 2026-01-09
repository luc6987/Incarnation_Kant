import os
from pathlib import Path
from typing import Optional

# Define Project Root
PROJECT_ROOT = Path(__file__).parent.parent

from pydantic_settings import BaseSettings
from pydantic import Field
import yaml

class AppConfig(BaseSettings):
    title: str = "Digital Kant"
    version: str = "2.0.0"
    language: str = "zh"

class LLMConfig(BaseSettings):
    model_name: str = "gemma-3-27b-it"
    temperature: float = 0.7
    max_output_tokens: int = 2096
    api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")

class DatabaseConfig(BaseSettings):
    persist_directory: str = "./data/chromadb"
    collection_name: str = "kant_corpus_local"
    embedding_model: str = "BAAI/bge-m3"
    top_k: int = 5
    search_type: str = "similarity"
    mmr_lambda: float = 0.3
    fetch_k: int = 50

class HydeConfig(BaseSettings):
    enabled: bool = True
    model_name: str = "gemma-3-27b-it"
    temperature: float = 0.0

class RearticulationConfig(BaseSettings):
    model_name: str = "gemma-3-27b-it"
    temperature: float = 0.3

class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    hyde: HydeConfig = HydeConfig()
    rearticulation: RearticulationConfig = RearticulationConfig()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

def load_config(config_path: str = "config/settings.yaml") -> Settings:
    """Load configuration from YAML file and environment variables."""
    # Ensure config path exists
    path = Path(config_path)
    if not path.exists():
        # Fallback if running from a different directory
        alt_path = Path("..") / config_path
        if alt_path.exists():
            path = alt_path
        else:
            # If no yaml, return defaults (which will try to read from env)
            return Settings()
            
    with open(path, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)
        
    # Manually merge YAML into settings if needed, or rely on pydantic loading
    # Ideally, we map yaml structure to pydantic. 
    # For simplicity here, we'll initialize Settings with values from YAML 
    # allowing env vars to override them if we were using a more complex loader.
    # But BaseSettings + yaml is tricky without a custom source.
    # Let's simple construct the dict.
    
    return Settings(**yaml_config)

# Global settings instance
settings = load_config()
