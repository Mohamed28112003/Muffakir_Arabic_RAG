from typing import List, Dict, Any, Optional
import logging
from langchain.schema import Document

from Muffakir.Enums import RetrievalMethod
from LLMProvider.LLMProvider import LLMProvider
from QueryTransformer.QueryTransformer import QueryTransformer
from PromptManager.PromptManager import PromptManager
from VectorDB.ChromaDBManager import ChromaDBManager
from RAGPipeline.RetrieveMethods import RetrieveMethods
from HallucinationsCheck.HallucinationsCheck import HallucinationsCheck
from Generation.RAGGenerationPipeline import RAGGenerationPipeline
from Reranker.Reranker import Reranker
class RAGPipelineManager:
    """
    Manages the full RAG pipeline: retrieval, reranking, hallucination checking, generation.
    """
    def __init__(
        self,
        db_path: str,
        collection_name: str = 'Book',
        model_name: str = 'mohamed2811/Muffakir_Embedding',
        llm_provider: Optional[LLMProvider] = None,
        query_transformer: Optional[QueryTransformer] = None,
        prompt_manager: Optional[PromptManager] = None,
        
        hallucination: Optional[HallucinationsCheck] = None,
        reranker: Optional[Reranker] = None,  # Placeholder for future reranker integration
        k: int = 2,
        fetch_k: int = 7,
        retrieve_method: RetrievalMethod = RetrievalMethod.SIMILARITY_SEARCH,
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Core components
        self.db_manager = ChromaDBManager(
            path=db_path,
            collection_name=collection_name,
            model_name=model_name,
        )
        self.llm_provider = llm_provider
        self.query_transformer = query_transformer
        self.prompt_manager = prompt_manager
        
        self.hallucination = hallucination
        self.reranker = reranker  

        # RAG parameters
        self.k = k
        self.fetch_k = fetch_k
        self.retrieve_method = retrieve_method

        # Subsystems
        self.retriever = RetrieveMethods(self.db_manager.vector_store)
        self.generation_pipeline = RAGGenerationPipeline(
            pipeline_manager=self,
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
    
            query_transformer=self.query_transformer,
            hallucination = self.hallucination,
            reranker=self.reranker,
        )

    def store_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector database.
        """
        self.db_manager.add_documents(documents)
        self.logger.info(f"Stored {len(documents)} documents successfully.")

    def query_similar_documents(
        self,
        query: str,
        k: Optional[int] = None,
        method: Optional[RetrievalMethod] = None
    ) -> List[Document]:
        """
        Retrieve similar documents based on the selected retrieval strategy.

        :param query: the user’s query
        :param k: override the top-k count (defaults to self.k)
        :param method: override the retrieval method (defaults to self.retrieve_method)
        """
        k = k or self.k
        method = method or self.retrieve_method


        self.logger.info(f"Retrieving documents using {method.value} (k={k}) for query: {query}")

        if method == RetrievalMethod.MAX_MARGINAL_RELEVANCE:
            return self.retriever.max_marginal_relevance_search(query, k, self.fetch_k)

        if method == RetrievalMethod.SIMILARITY_SEARCH:
            return self.retriever.similarity_search(query, k)

        if method == RetrievalMethod.HYBRID:
            return self.retriever.HybridRAG(query, k)

        if method == RetrievalMethod.CONTEXTUAL:
            return self.retriever.ContextualRAG(llm_provider=self.llm_provider, query=query)

        raise ValueError(f"Unsupported retrieval method: {method}")

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate a final answer from retrieved documents and the generation pipeline.
        """
        return self.generation_pipeline.generate_response(query)