from typing import Dict, Any
import logging


from LLMProvider.LLMProvider import LLMProvider

from PromptManager.PromptManager import PromptManager

from Muffakir.Enums import ProviderName

from Muffakir.Search import Search

class MuffakirSearch:

    
    DEFAULT_CONFIG = {

        "llm_api_key": None,
        "fire_crawl_api" : None, 
        "llm_provider": "together", 
        "llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "llm_temperature": 0.0,
        "llm_max_tokens": 1000,
        "max_depth" : 1,
        "time_limit": 30,
        "max_urls": 5,

    }
    
    PROVIDER_MAPPING = {
        "together": ProviderName.TOGETHER,
        "openai": ProviderName.OPENAI,
        "open_router": ProviderName.OPENROUTER,
        "groq": ProviderName.GROQ,
    }
    


    def __init__(self, config: Dict[str, Any]):

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.config = self._merge_config(config)
        
        self._validate_config()
        
        self._initialize_components()
        
        self._setup_pipeline()
        
        self.logger.info("🎉 MuffakirSearch initialized successfully!")

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with default values."""
        config = self.DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_params = ["fire_crawl_api", "llm_api_key", "llm_provider"]
        
        for param in required_params:
            if not self.config.get(param):
                raise ValueError(f"❌ Required parameter '{param}' is missing from configuration")
        

        # Validate provider
        if self.config["llm_provider"] not in self.PROVIDER_MAPPING:
            available_providers = ", ".join(self.PROVIDER_MAPPING.keys())
            raise ValueError(f"❌ Unsupported LLM provider: {self.config['llm_provider']}. Available providers: {available_providers}")
        

        self.logger.info("✅ Configuration validated successfully")

    def _initialize_components(self):
        """Initialize all core components."""
        self.logger.info("🔧 Initializing core components...")
        
        # Initialize LLM Provider
        self.llm_provider = LLMProvider(
            api_key=self.config["llm_api_key"],
            provider=self.PROVIDER_MAPPING[self.config["llm_provider"]],
            model=self.config["llm_model"],
            temperature=self.config.get("llm_temperature", 0.0),
            max_tokens=self.config.get("llm_max_tokens", 1000),
        )
        self.logger.info(f"✅ LLM Provider initialized: {self.config['llm_provider']}")

        # Initialize Prompt Manager
        self.prompt_manager = PromptManager()
        self.logger.info("✅ Prompt Manager initialized")
        



    def _setup_pipeline(self):
        """Setup the complete RAG pipeline."""
        self.logger.info("🔄 Setting up Search pipeline...")
        

        # Initialize RAG Pipeline Manager
        self.search_pipline = Search(
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
            fire_crawl_api=self.config["fire_crawl_api"],
            max_depth=self.config.get("max_depth", 3),
            time_limit=self.config.get("time_limit", 60),
            max_urls=self.config.get("max_urls", 10),
        )
        
    



    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    



    def search(self, query: str) -> Dict[str, Any]:
        """Perform a search using the configured LLM provider."""
        self.logger.info(f"🔍 Performing search for query: {query}")
        
        try:
            results = self.search_pipline.search_web(query)
            self.logger.info("✅ Search completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            raise e