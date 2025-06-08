from typing import List, Dict, Any
import logging

from firecrawl import FirecrawlApp
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import PromptManager
from Generation.AnswerGenerator import AnswerGenerator

class Search:
    def __init__(
        self,
        fire_crawl_api: str,
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        max_depth: int = 2,
        time_limit: int = 30,
        max_urls: int = 5,
    ):
        self.logger = logging.getLogger(__name__)
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.max_urls = max_urls

        self.firecrawl = FirecrawlApp(api_key=fire_crawl_api)

        self.generator = AnswerGenerator(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager
        )

    def deep_search(self, original_query: str) -> Dict[str, Any]:
        optimized = original_query

        try:
            results = self.firecrawl.deep_research(
                query=optimized,
                max_depth=self.max_depth,
                time_limit=self.time_limit,
                max_urls=self.max_urls,
            )
            return results
        except Exception as e:
            self.logger.error(f"DeepSearch failed: {e}", exc_info=True)
            raise

    def get_sources(self, deep_search_result: Dict[str, Any]) -> List[Dict[str, str]]:
        raw = deep_search_result.get("data", {}).get("sources", [])
        sources: List[Dict[str, str]] = []
        for s in raw:
            title = s.get("title", "").strip()
            url = s.get("url", "").strip()
            if title and url:
                sources.append({"title": title, "url": url})
        self.logger.debug(f"Parsed {len(sources)} sources.")
        return sources

    def search_web(self, original_query: str) -> Dict[str, Any]:
        raw = self.deep_search(original_query)
        analysis = raw.get("data", {}).get("finalAnalysis", "")
        sources = self.get_sources(raw)

        try:
            answer = self.generator.generate_answer(original_query, analysis)
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}", exc_info=True)
            answer = "❌ فشل في توليد الإجابة."

        return {
            "answer": answer,
            "sources": sources
        }