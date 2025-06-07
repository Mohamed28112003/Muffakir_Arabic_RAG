from typing import  List ,Optional
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from LLMProvider.LLMProvider import *

class RetrieveMethods:
    def __init__(self, vector_store: Chroma):

        self.vector_store = vector_store
        self._all_documents = self.vector_store.get()["documents"]


    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        return self.vector_store.similarity_search(query, k)

    def max_marginal_relevance_search(self, query: str, k: int = 2, fetch_k: int = 12) -> List[Document]:
        return self.vector_store.max_marginal_relevance_search(query, k, fetch_k)
    
    def HybridRAG(self, query: str, k: int = 2) -> List[Document]:
        vector_retriever = self.vector_store.as_retriever()

        bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content=doc) for doc in self._all_documents]
        )
        bm25_retriever.k = k

        hybrid_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])
        results = hybrid_retriever.get_relevant_documents(query)

        seen = set()
        unique_results = []
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_results.append(doc)
        
        return unique_results[:k]
    
    def ContextualRAG(self, query: str, k: int = 2,llm_provider: Optional[LLMProvider] = None) -> List[Document]:
        compressor = LLMChainExtractor.from_llm(llm_provider.get_llm())

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vector_store.as_retriever()
        )

        return compression_retriever.get_relevant_documents(query)
    
    # def AgenticRAG(self,query:str ,top_k: int = 5)-> List[Document]:
    #     db_path = "D:\Graduation Project\Local\DB_FINAL"

    #     agent = AgenticRag(db_path=db_path,k=top_k)
    #     response = agent.run_query(query)
    #     return response


        







