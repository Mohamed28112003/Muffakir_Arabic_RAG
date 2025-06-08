from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path

from TextProcessor.ChunkingAndProcessing import ChunkingAndProcessing
from LLMProvider.LLMProvider import LLMProvider
from Embedding.EmbeddingProvider import EmbeddingProvider
from PromptManager.PromptManager import PromptManager
from QueryTransformer.QueryTransformer import QueryTransformer
from HallucinationsCheck.HallucinationsCheck import HallucinationsCheck
from RAGPipeline.RAGPipelineManager import RAGPipelineManager
from Muffakir.Enums import ProviderName, RetrievalMethod
from langchain.schema import Document
from Reranker.Reranker import Reranker

class MuffakirRAG:
    """
    الفهرس الذكي - مكتبة الاسترجاع التوليدي المعزز للنصوص العربية
    Arabic RAG (Retrieval-Augmented Generation) Library
    
    A comprehensive Arabic RAG library that provides intelligent document processing,
    retrieval, and question-answering capabilities.
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        # Required parameters
        "data_dir": None,
        "api_key": None,
        
        # LLM Configuration
        "llm_provider": None,  # together, openai, groq, anthropic
        "llm_model": None,
        "llm_temperature": 0.0,
        "llm_max_tokens": 1000,
        
        # Embedding Configuration
        "embedding_model": "mohamed2811/Muffakir_Embedding",
        "embedding_batch_size": 16,
        
        # Document Processing
        "chunk_size": 600,
        "chunk_overlap": 200,
        "chunking_method": "recursive",  # recursive, character, token, spacy
        
        # Vector Database
        "db_path": "./muffakir_db",
        "collection_name": "ArabicBooks",
        
        # Retrieval Configuration
        "retrieval_method": "max_marginal_relevance",  # similarity_search, max_marginal_relevance, hybrid, contextual
        "k": 5,
        "fetch_k": 15,
        
        # Query Processing
        "query_transformer": False,
        "hallucination_check": True,
        
        # OCR Configuration (optional)
        "use_ocr": False,
        "azure_endpoint": None,
        "azure_api_key": None,

        # Reranking
        "reranking": False,  # Enable or disable reranking
        "reranking_method": "semantic_similarity",  # Options: semantic_similarity, bm25, cross_encoder
    }
    
    # Provider name mapping
    PROVIDER_MAPPING = {
        "together": ProviderName.TOGETHER,
        "openai": ProviderName.OPENAI,
        "open_router": ProviderName.OPENROUTER,
        "groq": ProviderName.GROQ,
    }
    
    # Retrieval method mapping
    RETRIEVAL_MAPPING = {
        "similarity_search": RetrievalMethod.SIMILARITY_SEARCH,
        "max_marginal_relevance": RetrievalMethod.MAX_MARGINAL_RELEVANCE,
        "hybrid": RetrievalMethod.HYBRID,
        "contextual": RetrievalMethod.CONTEXTUAL,
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MuffakirRAG with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all settings
            
        Example:
            config = {
                "data_dir": "content/files/books",
                "llm_provider": "together",
                "api_key": "your_api_key_here",
                "embedding_model": "mohamed2811/Muffakir_Embedding",
                "k": 5,
                "query_transformer": True
            }
            rag = MuffakirRAG(config)
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Merge user config with defaults
        self.config = self._merge_config(config)
        
        # Validate required parameters
        self._validate_config()
        
        # Initialize components
        self._initialize_components()
        
        # Process documents and setup RAG pipeline
        self._setup_pipeline()
        
        self.logger.info("🎉 MuffakirRAG initialized successfully! مرحباً بك في الفهرس الذكي")

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with default values."""
        config = self.DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_params = ["data_dir", "api_key", "llm_provider","llm_model"]
        
        for param in required_params:
            if not self.config.get(param):
                raise ValueError(f"❌ Required parameter '{param}' is missing from configuration")
        
        # Validate data directory exists
        data_path = Path(self.config["data_dir"])
        if not data_path.exists():
            raise FileNotFoundError(f"❌ Data directory does not exist: {self.config['data_dir']}")
        
        # Validate provider
        if self.config["llm_provider"] not in self.PROVIDER_MAPPING:
            available_providers = ", ".join(self.PROVIDER_MAPPING.keys())
            raise ValueError(f"❌ Unsupported LLM provider: {self.config['llm_provider']}. Available providers: {available_providers}")
        
        # Validate retrieval method
        if self.config["retrieval_method"] not in self.RETRIEVAL_MAPPING:
            available_methods = ", ".join(self.RETRIEVAL_MAPPING.keys())
            raise ValueError(f"❌ Unsupported retrieval method: {self.config['retrieval_method']}. Available methods: {available_methods}")
        
        self.logger.info("✅ Configuration validated successfully")

    def _initialize_components(self):
        """Initialize all core components."""
        self.logger.info("🔧 Initializing core components...")
        
        # Initialize LLM Provider
        self.llm_provider = LLMProvider(
            api_key=self.config["api_key"],
            provider=self.PROVIDER_MAPPING[self.config["llm_provider"]],
            model=self.config["llm_model"],
            temperature=self.config.get("llm_temperature", 0.0),
            max_tokens=self.config.get("llm_max_tokens", 1000),
        )
        self.logger.info(f"✅ LLM Provider initialized: {self.config['llm_provider']}")
        
        # Initialize Embedding Provider
        self.embedding_provider = EmbeddingProvider(
            model_name=self.config["embedding_model"],
            batch_size=self.config.get("embedding_batch_size", 16),
        )
        self.logger.info(f"✅ Embedding Provider initialized: {self.config['embedding_model']}")
        
        # Initialize Prompt Manager
        self.prompt_manager = PromptManager()
        self.logger.info("✅ Prompt Manager initialized")
        

        # Initialize optional components
        self.query_transformer = None
        if self.config["query_transformer"]:
            self.query_transformer = QueryTransformer(
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
              
            )
            self.logger.info("✅ Query Transformer enabled")
        
        self.hallucination_checker = None
        if self.config.get("hallucination_check", True):
            self.hallucination_checker = HallucinationsCheck(
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
            )
            self.logger.info("✅ Hallucination Checker enabled")
        
        self.reranker = None
        if self.config.get("reranking", False):
            self.reranker = Reranker(
                embedding_provider=self.embedding_provider,
                reranking_method=self.config["reranking_method"]
            )
            self.logger.info(f"✅ Reranker enabled with method: {self.config['reranking_method']}")

    def _setup_pipeline(self):
        """Setup the complete RAG pipeline."""
        self.logger.info("🔄 Setting up RAG pipeline...")
        
        # Initialize document processor
        self.document_processor = ChunkingAndProcessing(
            directory_path=self.config["data_dir"],
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            azure_endpoint=self.config.get("azure_endpoint"),
            azure_api_key=self.config.get("azure_api_key")
        )
        
        # Process documents
        self.logger.info("📚 Processing documents...")
        self.documents = self.document_processor.process_all(
            chunking_method=self.config["chunking_method"],
            use_ocr=self.config.get("use_ocr", False)
        )
        self.logger.info(f"✅ Processed {len(self.documents)} document chunks")
        
        # Initialize RAG Pipeline Manager
        self.rag_manager = RAGPipelineManager(
            db_path=self.config["db_path"],
            collection_name=self.config["collection_name"],
            model_name=self.config["embedding_model"],
            llm_provider=self.llm_provider,
            query_transformer=self.query_transformer,
            prompt_manager=self.prompt_manager,
         
            hallucination=self.hallucination_checker,
            reranker=self.reranker,
            k=self.config["k"],
            fetch_k=self.config["fetch_k"],
            retrieve_method=self.RETRIEVAL_MAPPING[self.config["retrieval_method"]],
        )
        
        # Store documents in vector database
        self.logger.info("💾 Storing documents in vector database...")
        self.rag_manager.store_documents(self.documents)
        self.logger.info("✅ Documents stored successfully")

    def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Ask a question and get an intelligent answer.
        
        Args:
            question (str): The question in Arabic or English
            **kwargs: Additional parameters to override defaults
                - k (int): Number of documents to retrieve
                - retrieval_method (str): Retrieval method to use
                - temperature (float): LLM temperature for this query
        
        Returns:
            Dict[str, Any]: Response containing answer and metadata
            
        Example:
            response = rag.ask("ما هو تعريف الحق في القانون المدني؟")
            print(response["answer"])
        """
        if not question or not question.strip():
            return {
                "answer": "❌ يرجى تقديم سؤال صحيح",
                "error": "Empty question provided",
                "sources": []
            }
        
        try:
            self.logger.info(f"🤔 Processing question: {question[:100]}...")
            
            # Store original values for restoration
            original_k = None
            original_method = None
            
            # Override parameters if provided
            if 'k' in kwargs:
                original_k = self.rag_manager.k
                self.rag_manager.k = kwargs['k']
            
            if 'retrieval_method' in kwargs:
                original_method = self.rag_manager.retrieve_method
                if kwargs['retrieval_method'] in self.RETRIEVAL_MAPPING:
                    self.rag_manager.retrieve_method = self.RETRIEVAL_MAPPING[kwargs['retrieval_method']]
                else:
                    self.logger.warning(f"⚠️ Unknown retrieval method: {kwargs['retrieval_method']}")
            
            # Generate answer
            response = self.rag_manager.generate_answer(question)
            
            # Restore original parameters
            if original_k is not None:
                self.rag_manager.k = original_k
            if original_method is not None:
                self.rag_manager.retrieve_method = original_method
            
            self.logger.info("✅ Answer generated successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error generating answer: {str(e)}")
            return {
                "answer": "❌ عذراً، حدث خطأ أثناء معالجة سؤالك",
                "error": str(e),
                "sources": []
            }

    def get_similar_documents(
        self, 
        query: str, 
        k: Optional[int] = None,
        method: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve similar documents for a given query.
        
        Args:
            query (str): Search query
            k (int, optional): Number of documents to retrieve
            method (str, optional): Retrieval method to use
            
        Returns:
            List[Document]: Similar documents
        """
        if not query or not query.strip():
            self.logger.warning("⚠️ Empty query provided for document retrieval")
            return []
            
        try:
            k = k or self.config["k"]
            method_enum = self.RETRIEVAL_MAPPING.get(
                method or self.config["retrieval_method"]
            )
            
            return self.rag_manager.query_similar_documents(
                query=query,
                k=k,
                method=method_enum
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving documents: {str(e)}")
            return []

    def add_documents(self, documents: Union[List[Document], List[str]]) -> bool:
        """
        Add new documents to the knowledge base.
        
        Args:
            documents: List of Document objects or file paths
            
        Returns:
            bool: Success status
        """
        if not documents:
            self.logger.warning("⚠️ No documents provided")
            return False
            
        try:
            if isinstance(documents[0], str):
                # Process file paths
                processed_docs = []
                for file_path in documents:
                    if not Path(file_path).exists():
                        self.logger.warning(f"⚠️ File not found: {file_path}")
                        continue
                        
                    temp_processor = ChunkingAndProcessing(
                        directory_path=str(Path(file_path).parent),
                        chunk_size=self.config["chunk_size"],
                        chunk_overlap=self.config["chunk_overlap"]
                    )
                    docs = temp_processor.process_all(
                        chunking_method=self.config["chunking_method"]
                    )
                    processed_docs.extend(docs)
                documents = processed_docs
            
            if not documents:
                self.logger.warning("⚠️ No valid documents to add")
                return False
                
            self.rag_manager.store_documents(documents)
            self.logger.info(f"✅ Added {len(documents)} new documents")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding documents: {str(e)}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    

    def __repr__(self) -> str:
        doc_count = len(self.documents) if hasattr(self, 'documents') else 0
        return f"MuffakirRAG(provider={self.config['llm_provider']}, docs={doc_count}, k={self.config['k']})"


