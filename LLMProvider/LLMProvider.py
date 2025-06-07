
# LLMProvider.py
from typing import Any
import logging
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from Muffakir.Enums import ProviderName

class LLMProvider:

    def __init__(
        self,
        api_key: str,
        provider: ProviderName,
        model: str,
        temperature: float = 0.5,
        max_tokens: int = 300,
    ):
        if not api_key:
            raise ValueError("An API key must be provided.")

        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.llm = self.initialize_llm()

    def initialize_llm(self) -> Any:

        common = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)

        if self.provider == ProviderName.GROQ:
            return ChatGroq(api_key=self.api_key, **common)

        if self.provider == ProviderName.TOGETHER:
            return ChatOpenAI(
                openai_api_key=self.api_key,
                openai_api_base="https://api.together.xyz/v1",
                **common,
            )

        if self.provider == ProviderName.OPENROUTER:
            return ChatOpenAI(
                openai_api_key=self.api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                **common,
            )
        
        if self.provider == ProviderName.OPENAI:
            return ChatOpenAI(
                openai_api_key=self.api_key,
                **common,
            )

        raise ValueError(f"Unsupported provider: {self.provider}")

    def get_llm(self) -> Any:
        """
        Get the underlying LLM client.
        """
        return self.llm

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Proxy method to invoke the LLM.
        """
        try:
            return self.llm(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
