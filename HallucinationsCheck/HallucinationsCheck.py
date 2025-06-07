import re
from LLMProvider.LLMProvider import *
from PromptManager.PromptManager import *

class HallucinationsCheck:
 
    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager):
        self.llm_provider = llm_provider
        self.hallucination_check_prompt = prompt_manager.get_prompt("hallucination_check_prompt")

    def clean_text(self, text: str) -> str:

        # If you need to allow additional punctuation, add them inside the brackets.
        return re.sub(r'[^\u0600-\u06FF\s]', '', text)

    def check_answer(self, answer: str) -> str:
        try:
            llm = self.llm_provider.get_llm()
            prompt = self.hallucination_check_prompt.format(answer=answer)
            response = llm.invoke(prompt)

            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            cleaned_response = self.clean_text(response_text)
            return cleaned_response
        except Exception as e:
            print(f"Error HallucinationsCheck: {e}")

