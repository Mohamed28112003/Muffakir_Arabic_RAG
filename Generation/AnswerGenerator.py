from typing import List, Union
import logging

from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import PromptManager
from langchain.schema import Document

class AnswerGenerator:
    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
        self.generation_prompt = self.prompt_manager.get_prompt("generation")

    def generate_answer(
        self,
        query: str,
        documents: Union[List[Document], str]
    ) -> str:
        if isinstance(documents, str):
            context = documents
        else:
            try:
                context = "\n\n".join(doc.page_content for doc in documents)
            except Exception as e:
                self.logger.error(f"Failed to join page_content: {e}", exc_info=True)
                context = str(documents)

        try:
            prompt = self.generation_prompt.format(
                context=context,
                question=query
            )
        except Exception as e:
            self.logger.error(f"Failed to format generation prompt: {e}", exc_info=True)
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        try:
            llm = self.llm_provider.get_llm()
            if hasattr(llm, "invoke"):
                response = llm.invoke(prompt)
            else:
                response = llm(prompt)

            if hasattr(response, "content"):
                return response.content
            return str(response)

        except Exception as e:
            self.logger.error(f"Error generating answer from LLM: {e}", exc_info=True)
            return "❌ فشل في توليد الإجابة."