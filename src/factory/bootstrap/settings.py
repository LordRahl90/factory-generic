from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FACTORY_", extra="ignore")

    default_model: str = "ollama_chat/qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
