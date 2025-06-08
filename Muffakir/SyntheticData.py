from PromptManager.PromptManager import PromptManager
from LLMProvider.LLMProvider import LLMProvider
from TextProcessor.ChunkingAndProcessing import ChunkingAndProcessing
from typing import List, Optional, Dict, Any
from langchain.schema import Document
import pandas as pd
import os
import re
import logging
from datetime import datetime

class SyntheticData:
    """
    A class for generating synthetic Q&A data from document chunks using LLM.
    Creates a DataFrame with question, answer, and context columns for each chunk.
    """
    
    def __init__(self, 
                 llm_provider: LLMProvider, 
                 prompt_manager: PromptManager,
                 data_dir: str,
                 chunk_size: int = 600,
                 chunk_overlap: int = 200,
                 chunking_method: str = "recursive",
                 use_ocr: bool = False,
                 azure_endpoint: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 output_dir: str = "./synthetic_data_output"):
        """
        Initialize the SyntheticData class.
        
        Args:
            llm_provider: LLM provider instance
            prompt_manager: Prompt manager instance
            data_dir: Directory containing input documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunking_method: Method for chunking ("recursive", "character", "token")
            use_ocr: Whether to use OCR for document processing
            azure_endpoint: Azure endpoint for OCR (if use_ocr=True)
            azure_api_key: Azure API key for OCR (if use_ocr=True)
            output_dir: Directory to save output files
        """
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.data_dir = data_dir
        self.chunking_method = chunking_method
        self.use_ocr = use_ocr
        self.output_dir = output_dir
        
        # Initialize chunking processor
        self.chunking_processor = ChunkingAndProcessing(
            directory_path=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize DataFrame to store results
        self.qa_dataframe = pd.DataFrame(columns=['question', 'answer', 'context', 'chunk_id', 'source_file'])
    
    def process_and_chunk_data(self) -> List[Document]:
        """Process and chunk the input text data."""
        try:
            self.logger.info(f"Processing documents from: {self.data_dir}")
            processed_chunks = self.chunking_processor.process_all(
                chunking_method=self.chunking_method,
                use_ocr=self.use_ocr,
                ocr_output_dir=os.path.join(self.output_dir, "ocr_results") if self.use_ocr else None
            )
            self.logger.info(f"Successfully processed {len(processed_chunks)} chunks")
            return processed_chunks
        except Exception as e:
            self.logger.error(f"Error processing and chunking data: {e}")
            return []
    
    def parse_qa_response(self, response: str) -> Dict[str, str]:
        """
        Parse the LLM response to extract question and answer.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Dictionary with 'question' and 'answer' keys
        """
        try:
            # Clean the response
            response = response.strip()
            
            # Extract question using regex
            question_match = re.search(r'السؤال:\s*([^\n]+)', response)
            question = question_match.group(1).strip() if question_match else ""
            
            # Extract answer using regex
            answer_match = re.search(r'الإجابة:\s*([^\n]+(?:\n[^\n]+)*)', response)
            answer = answer_match.group(1).strip() if answer_match else ""
            
            # Fallback: try alternative patterns
            if not question or not answer:
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    if 'السؤال' in line and ':' in line:
                        question = line.split(':', 1)[1].strip()
                    elif 'الإجابة' in line and ':' in line:
                        answer = line.split(':', 1)[1].strip()
                        # Include subsequent lines as part of answer if they don't contain new labels
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() and not any(keyword in lines[j] for keyword in ['السؤال', 'الإجابة']):
                                answer += ' ' + lines[j].strip()
                            else:
                                break
            
            return {
                'question': question,
                'answer': answer
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Q&A response: {e}")
            return {'question': "", 'answer': ""}
    
    def generate_synthetic_data(self, context: str) -> Dict[str, str]:
        """Generate synthetic Q&A data based on the provided context."""
        try:
            prompt = self.prompt_manager.get_prompt("QA").format(context=context)
            
            llm = self.llm_provider.get_llm()
            response = llm.invoke(prompt)
            
            # Extract content from response
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response to extract Q&A
            qa_data = self.parse_qa_response(response_content)
            
            return qa_data
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data: {e}")
            return {'question': "", 'answer': ""}
    
    def add_to_dataframe(self, question: str, answer: str, context: str, 
                        chunk_id: int, source_file: str):
        """Add a new Q&A pair to the DataFrame."""
        new_row = pd.DataFrame({
            'question': [question],
            'answer': [answer],
            'context': [context],
            'chunk_id': [chunk_id],
            'source_file': [source_file]
        })
        
        self.qa_dataframe = pd.concat([self.qa_dataframe, new_row], ignore_index=True)
    
    def save_dataframe(self, filename: str = "synthetic_qa_data.csv", mode: str = 'w'):
        """Save the DataFrame to a single CSV file (append or write)."""
        try:
            csv_path = os.path.join(self.output_dir, filename)
            file_exists = os.path.exists(csv_path)

            self.qa_dataframe.to_csv(
                csv_path,
                mode=mode,
                index=False,
                header=not file_exists if mode == 'a' else True,
                encoding='utf-8'
            )

            self.logger.info(f"Data saved to: {csv_path}")
            # Clear DataFrame after saving if in append mode
            if mode == 'a':
                self.qa_dataframe = self.qa_dataframe.iloc[0:0]
            return csv_path
        except Exception as e:
            self.logger.error(f"❌ Error saving CSV file: {e}")
            return ""

    
    def generate_qa_pipeline(self, save_frequency: int = 10) -> pd.DataFrame:
        """
        Complete pipeline to generate Q&A data from documents.
        
        Args:
            save_frequency: How often to save the DataFrame (after every N chunks)
            
        Returns:
            pandas.DataFrame: DataFrame containing all generated Q&A pairs
        """
        try:
            # Step 1: Process and chunk documents
            self.logger.info("Starting Q&A generation pipeline...")
            chunks = self.process_and_chunk_data()
            
            if not chunks:
                self.logger.warning("No chunks found to process")
                return self.qa_dataframe
            
            # Step 2: Generate Q&A for each chunk
            successful_generations = 0
            failed_generations = 0
            
            for i, chunk in enumerate(chunks, 1):
                self.logger.info(f"Processing chunk {i}/{len(chunks)}")
                
                try:
                    # Generate Q&A
                    qa_data = self.generate_synthetic_data(chunk.page_content)
                    
                    # Check if Q&A was generated successfully
                    if qa_data['question'] and qa_data['answer']:
                        # Add to DataFrame
                        self.add_to_dataframe(
                            question=qa_data['question'],
                            answer=qa_data['answer'],
                            context=chunk.page_content,
                            chunk_id=chunk.metadata.get('chunk_id', i),
                            source_file=chunk.metadata.get('source_file', 'unknown')
                        )
                        successful_generations += 1
                        
                        # Save periodically
                        if successful_generations % save_frequency == 0:
                            self.save_dataframe(f"synthetic_qa_data_checkpoint_{successful_generations}")
                            self.logger.info(f"Checkpoint saved after {successful_generations} successful generations")
                    else:
                        failed_generations += 1
                        self.logger.warning(f"Failed to generate Q&A for chunk {i}")
                        
                except Exception as e:
                    failed_generations += 1
                    self.logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            # Final save
            csv_path, excel_path = self.save_dataframe()
            
            # Log summary
            self.logger.info(f"Pipeline completed!")
            self.logger.info(f"Total chunks processed: {len(chunks)}")
            self.logger.info(f"Successful Q&A generations: {successful_generations}")
            self.logger.info(f"Failed generations: {failed_generations}")
            self.logger.info(f"Final dataset shape: {self.qa_dataframe.shape}")
            
            return self.qa_dataframe
            
        except Exception as e:
            self.logger.error(f"Error in Q&A generation pipeline: {e}")
            # Save whatever we have so far
            if not self.qa_dataframe.empty:
                self.save_dataframe("synthetic_qa_data_error_recovery")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated Q&A dataset."""
        if self.qa_dataframe.empty:
            return {"message": "No data generated yet"}
        
        stats = {
            "total_qa_pairs": len(self.qa_dataframe),
            "unique_sources": self.qa_dataframe['source_file'].nunique(),
            "avg_question_length": self.qa_dataframe['question'].str.len().mean(),
            "avg_answer_length": self.qa_dataframe['answer'].str.len().mean(),
            "avg_context_length": self.qa_dataframe['context'].str.len().mean(),
            "source_distribution": self.qa_dataframe['source_file'].value_counts().to_dict()
        }
        
        return stats


