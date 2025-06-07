from typing import List, Dict, Any
from langchain.schema import Document
from typing import TYPE_CHECKING

from RAGPipeline.RAGPipelineManager import RAGPipelineManager

class DocumentRetriever:
    def __init__(self, pipeline_manager: "RAGPipelineManager"):
        """
        :param pipeline_manager: the RAGPipelineManager instance
        """
        self.pipeline_manager = pipeline_manager

    def retrieve_documents(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top-k similar documents and extracts source & page_content.
        Returns a list of dicts with keys:
          - "source": the document’s source metadata
          - "page_content": the text
        """
        # Now binds k to the new k parameter, not to the enum
        results = self.pipeline_manager.query_similar_documents(query, k)

        return [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content
            }
            for doc in results
        ]

    def format_documents(self, retrieval_result: List[Dict[str, Any]]) -> List[Document]:
        """
        Formats the retrieved data into LangChain Document objects.
        """
        return [
            Document(page_content=item["page_content"], metadata={"source": item["source"]})
            for item in retrieval_result
        ]
