# config.py
from pydantic_settings import BaseSettings   # ← updated import
from Enums import ProviderName, RetrievalMethod, QueryType

class Settings(BaseSettings):
    # external APIs
    FIRE_CRAWL_API: str
    TOGETHER_API_KEY: str
    AZURE_API_KEY: str

    # file paths
    DB_PATH: str
    NEW_DB_PATH: str
    OUTPUT_DIR: str

    # model & provider configs
    PROVIDER_NAME: ProviderName
    LLM_MODEL_NAME: str
    EMBEDDING_MODEL_NAME: str

    # retrieval defaults
    RETRIEVE_METHOD: RetrievalMethod = RetrievalMethod.SIMILARITY_SEARCH
    K: int = 5
    FETCH_K: int = 7

    class Config:
        env_file = ".env"
        case_sensitive = False

# singleton you can import everywhere
settings = Settings()