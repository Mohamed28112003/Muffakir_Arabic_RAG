from Embedding.EmbeddingProvider import   *
from langchain_community.vectorstores import Chroma

from langchain.schema import Document
from typing import List

class ChromaDBManager:
    def __init__(self, path: str, collection_name: str = 'Book',
                 model_name: str = "mohamed2811/Muffakir_Embedding"):
        """
        Initialize ChromaDBManager with LangChain's Chroma vector store.
        """
        self.embedding_provider = EmbeddingProvider(model_name=model_name)

        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=path,
            embedding_function=self.embedding_provider

        )

    def add_documents(self, documents: List[Document]):
        """Add documents to the Chroma collection"""
        ids = [f"chunk_{i}" for i in range(len(documents))]

        self.vector_store.add_documents(
            documents=documents,
            ids=ids
        )
        self.vector_store.persist()  # Ensure persistence
        print(f"Stored {len(documents)} documents in the collection.")

    def similarity_search(self, query: str, k: int = 2) -> List[Document]:

        return self.vector_store.similarity_search(query, k)
    
    def max_marginal_relevance_search(self, query: str, k: int = 2 ,fetch_k:int=12) -> List[Document]:
        return self.vector_store.max_marginal_relevance_search(query, k,fetch_k)


    def get_collection_count(self) -> int:
        return self.vector_store._collection.count()



