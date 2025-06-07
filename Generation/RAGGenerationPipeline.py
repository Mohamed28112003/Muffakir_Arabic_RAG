from typing import Dict,Any
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import PromptManager
from Generation.AnswerGenerator import AnswerGenerator
from HallucinationsCheck.HallucinationsCheck import HallucinationsCheck
from QueryTransformer.QueryTransformer import QueryTransformer
from typing import Optional
from Reranker.Reranker import Reranker

from typing import List, Optional
from langchain.schema import Document

class RAGGenerationPipeline:
    
    def __init__(
        self,
        pipeline_manager,
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        query_transformer: Optional[QueryTransformer],
        hallucination: Optional[HallucinationsCheck],
        reranker: Optional[Reranker] = None,
        k: int = 5
    ):
        self.pipeline_manager = pipeline_manager
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.query_transformer = query_transformer
        self.hallucination = hallucination
        
        self.generator = AnswerGenerator(llm_provider, prompt_manager)

        self.k = k
        self.reranker = reranker
        
   
        
    def _process_vector_db_query(self, query: str) -> Dict[str, Any]:
        from Generation.DocumentRetriever import DocumentRetriever
        if self.query_transformer:
            query = self._query_transform(query)

        retriever = DocumentRetriever(self.pipeline_manager )
        retrieval_result = retriever.retrieve_documents(query, self.k)
        formatted_documents = retriever.format_documents(retrieval_result)

        if self.reranker:
            formatted_documents = self._rerank_documents(query, formatted_documents)
        
        answer = self.generator.generate_answer(query, formatted_documents)
        if self.hallucination:
            answer = self._hallucination_check(answer)
        

        
        
        return {
            "answer": answer,
            "retrieved_documents": [doc.page_content for doc in formatted_documents],
            "source_metadata": [doc.metadata for doc in formatted_documents],
        }
        
    
    
  
    def _query_transform(self, query: str) -> str:
        """Transform the query using the query transformer"""
        if self.query_transformer:
            return self.query_transformer.transform_query(query)
    
    def _rerank_documents(self, query:str,formatted_documents: List[Document]) -> List[Document]:
        """Rerank the retrieved documents if a reranker is provided"""
        if self.reranker:
            return self.reranker.rerank(query, formatted_documents)
    
    def _hallucination_check(self, answer: str) -> str:
        """Check for hallucinations in the generated answer"""
        if self.hallucination:
            return self.hallucination.check_answer(answer)

        
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Main entry point to generate a response"""

        return self._process_vector_db_query(query)
