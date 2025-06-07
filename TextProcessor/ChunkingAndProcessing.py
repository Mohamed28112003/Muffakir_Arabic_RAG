from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)
from langchain.schema import Document
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Optional, Literal
import re
import os
import logging

class ChunkingAndProcessing:
    """
    A comprehensive class for loading, processing, chunking, and OCR of documents
    with support for Arabic text processing and multiple chunking strategies.
    """
    
    def __init__(
        self, 
        directory_path: str,
        chunk_size: int = 600,
        chunk_overlap: int = 200,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None
    ):
        """
        Initialize the ChunkingAndProcessing class.
        
        Args:
            directory_path (str): Path to directory containing documents
            chunk_size (int): Size of text chunks (default: 600)
            chunk_overlap (int): Overlap between chunks (default: 200)
            azure_endpoint (str, optional): Azure Document Intelligence endpoint
            azure_api_key (str, optional): Azure Document Intelligence API key
        """
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Azure Document Intelligence setup
        self.azure_endpoint = azure_endpoint 
        self.azure_api_key = azure_api_key 
        
        if self.azure_endpoint and self.azure_api_key:
            self.document_analysis_client = DocumentAnalysisClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(self.azure_api_key)
            )
        else:
            self.document_analysis_client = None
            
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_documents(self) -> List[Document]:
        """
        Load all text and PDF documents from the specified directory.
        
        Returns:
            List[Document]: List of loaded documents
        """
        try:
            loader = DirectoryLoader(
                self.directory_path,
                glob=["*.txt", "*.pdf"],
                show_progress=True
            )
            documents = loader.load()
            self.logger.info(f"Loaded {len(documents)} documents from {self.directory_path}")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")
            raise

    def get_text_splitter(
        self, 
        method: Literal["recursive", "character", "token", "spacy"] = "recursive"
    ):
        """
        Get the appropriate text splitter based on the specified method.
        
        Args:
            method (str): Chunking method to use
            
        Returns:
            Text splitter instance
        """
        splitter_map = {
            "recursive": RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            ),
            "character": CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            ),
            "token": TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),

        }
        
        if method not in splitter_map:
            self.logger.warning(f"Unknown method '{method}', using 'recursive' instead")
            method = "recursive"
            
        return splitter_map[method]

    def chunk_documents(
        self, 
        documents: List[Document], 
        method: Literal["recursive", "character", "token", "spacy"] = "recursive"
    ) -> List[Document]:
        """
        Split documents into chunks using the specified method.
        
        Args:
            documents (List[Document]): Documents to chunk
            method (str): Chunking method to use
            
        Returns:
            List[Document]: List of chunked documents
        """
        try:
            text_splitter = self.get_text_splitter(method)
            chunked_docs = text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(chunked_docs)} chunks using {method} method")
            return chunked_docs
        except Exception as e:
            self.logger.error(f"Error chunking documents: {str(e)}")
            raise

    @staticmethod
    def clean_arabic_text(text: str) -> str:
        """
        Clean Arabic text by removing unwanted characters and formatting.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove page numbers and formatting

        text = re.sub(r'- \d+ -', '', text)
        
        # Keep Arabic characters, numbers, and basic punctuation
        text = re.sub(r'[^؀-ۿ0-9\s\.\,\!\?\:\;\-\(\)]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def clean_and_process_chunks(self, chunked_documents: List[Document]) -> List[Document]:
        """
        Clean text content and update metadata for chunked documents.
        
        Args:
            chunked_documents (List[Document]): Chunked documents to process
            
        Returns:
            List[Document]: Processed documents with cleaned text and updated metadata
        """
        processed_docs = []
        
        for i, doc in enumerate(chunked_documents):
            # Clean the text content
            cleaned_content = self.clean_arabic_text(doc.page_content)
            
            # Skip empty chunks
            if not cleaned_content.strip():
                continue
            
            # Update metadata
            new_metadata = doc.metadata.copy() if doc.metadata else {}
            new_metadata['chunk_id'] = i + 1
            new_metadata['chunk_size'] = len(cleaned_content)
            
            # Add source information if available
            if 'source' in new_metadata:
                source_path = new_metadata['source']
                filename = os.path.basename(source_path)
                filename_without_ext = os.path.splitext(filename)[0]
                new_metadata['source_file'] = filename_without_ext
            
            # Create processed document
            processed_doc = Document(
                page_content=cleaned_content,
                metadata=new_metadata
            )
            processed_docs.append(processed_doc)
        
        self.logger.info(f"Processed {len(processed_docs)} chunks after cleaning")
        return processed_docs

    def ocr_document(self, file_path: str, model_id: str = "prebuilt-layout") -> str:
        """
        Perform OCR on a document using Azure Document Intelligence.
        
        Args:
            file_path (str): Path to the document file
            model_id (str): Azure model ID to use (default: "prebuilt-layout")
            
        Returns:
            str: Extracted text from the document
        """
        if not self.document_analysis_client:
            raise ValueError("Azure Document Intelligence client not configured")
        
        try:
            with open(file_path, "rb") as document:
                poller = self.document_analysis_client.begin_analyze_document(
                    model_id=model_id,
                    document=document
                )
            
            result = poller.result()
            
            # Extract text from all pages
            extracted_text = ""
            for page in result.pages:
                for line in page.lines:
                    extracted_text += line.content + "\n"
            
            self.logger.info(f"OCR completed for {file_path}. Total pages: {len(result.pages)}")
            return extracted_text
            
        except Exception as e:
            self.logger.error(f"Error performing OCR on {file_path}: {str(e)}")
            raise

    def ocr_directory(self, output_dir: Optional[str] = None) -> List[Document]:
        """
        Perform OCR on all supported files in the directory.
        
        Args:
            output_dir (str, optional): Directory to save OCR results
            
        Returns:
            List[Document]: Documents created from OCR results
        """
        if not self.document_analysis_client:
            raise ValueError("Azure Document Intelligence client not configured")
        
        ocr_documents = []
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff',".txt"]
        
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in supported_extensions:
                try:
                    extracted_text = self.ocr_document(file_path)
                    
                    # Save OCR result if output directory specified
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(
                            output_dir, 
                            f"{os.path.splitext(filename)[0]}_ocr.txt"
                        )
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(extracted_text)
                    
                    # Create document
                    doc = Document(
                        page_content=extracted_text,
                        metadata={
                            'source': file_path,
                            'ocr_processed': True,
                            'original_filename': filename
                        }
                    )
                    ocr_documents.append(doc)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {filename}: {str(e)}")
                    continue
        
        self.logger.info(f"OCR processed {len(ocr_documents)} documents")
        return ocr_documents

    def process_all(
        self, 
        chunking_method: Literal["recursive", "character", "token", "spacy"] = "recursive",
        use_ocr: bool = False,
        ocr_output_dir: Optional[str] = None
    ) -> List[Document]:
        """
        Complete processing pipeline: load, OCR (optional), chunk, and clean documents.
        
        Args:
            chunking_method (str): Method to use for chunking
            use_ocr (bool): Whether to perform OCR on documents
            ocr_output_dir (str, optional): Directory to save OCR results
            
        Returns:
            List[Document]: Final processed and chunked documents
        """
        try:
            # Load documents
            if use_ocr:
                documents = self.ocr_directory(ocr_output_dir)
            else:
                documents = self.load_documents()
            
            if not documents:
                self.logger.warning("No documents found to process")
                return []
            
            # Chunk documents
            chunked_docs = self.chunk_documents(documents, chunking_method)
            
            # Clean and process chunks
            final_docs = self.clean_and_process_chunks(chunked_docs)
            
            self.logger.info(f"Processing complete. Final document count: {len(final_docs)}")
            return final_docs
            
        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {str(e)}")
            raise

# Example usage:
    # # Initialize processor
    # processor = ChunkingAndProcessing(
    #     directory_path="dir_test",
    #     chunk_size=600,
    #     chunk_overlap=200,
    #     azure_api_key=os.getenv("AZURE_API_KEY"),
    #     azure_endpoint= "https://documentsfree.cognitiveservices.azure.com/"
    # )
    # # Process documents with OCR
    # processed_docs_ocr = processor.process_all(
    #     chunking_method="character",
    #     use_ocr=True,
    #     ocr_output_dir="./ocr_results",
    # )

    # processor = ChunkingAndProcessing(
    #     directory_path="./documents",
    #     chunk_size=600,
    #     chunk_overlap=200
    # )
    
    # # Process documents with different methods
    # processed_docs = processor.process_all(
    #     chunking_method="recursive",
    #     use_ocr=False
    # )
    
    # print(f"Processed {len(processed_docs)} document chunks")