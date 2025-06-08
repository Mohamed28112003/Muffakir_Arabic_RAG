from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from TextProcessor.ChunkingAndProcessing import ChunkingAndProcessing
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import PromptManager
from Muffakir.Enums import ProviderName
from Muffakir.SyntheticData import SyntheticData


class MuffakirSyntheticData:

    DEFAULT_CONFIG = {
        "data_dir": None,
        "api_key": None,
        "llm_provider": "together",
        "llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "llm_temperature": 0.3,
        "llm_max_tokens": 2000,
        "chunk_size": 600,
        "chunk_overlap": 200,
        "chunking_method": "recursive",
        "use_ocr": False,
        "azure_endpoint": None,
        "azure_api_key": None,
        "output_dir": "./muffakir_synthetic_data",
        "save_frequency": 10,
        "output_format": ["csv", "excel"],
        "max_retries": 3,
        "skip_empty_chunks": True,
        "min_chunk_length": 50,
        "min_question_length": 10,
        "min_answer_length": 15,
        "validate_qa_pairs": True,
    }

    PROVIDER_MAPPING = {
        "together": ProviderName.TOGETHER,
        "openai": ProviderName.OPENAI,
        "open_router": ProviderName.OPENROUTER,
        "groq": ProviderName.GROQ,
    }

    SUPPORTED_FORMATS = ["csv", "excel", "json"]

    def __init__(self, config: Dict[str, Any]):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.config = self._merge_config(config)
        self._validate_config()
        self._initialize_components()
        self._setup_synthetic_generator()
        self.logger.info("🎉 MuffakirSyntheticData initialized successfully! مرحباً بك في مولد البيانات الاصطناعية الذكي")

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        config = self.DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config

    def _validate_config(self):
        required_params = ["data_dir", "api_key", "llm_provider"]
        for param in required_params:
            if not self.config.get(param):
                raise ValueError(f"❌ Required parameter '{param}' is missing from configuration")
        data_path = Path(self.config["data_dir"])
        if not data_path.exists():
            raise FileNotFoundError(f"❌ Data directory does not exist: {self.config['data_dir']}")
        if self.config["llm_provider"] not in self.PROVIDER_MAPPING:
            available_providers = ", ".join(self.PROVIDER_MAPPING.keys())
            raise ValueError(f"❌ Unsupported LLM provider: {self.config['llm_provider']}. Available providers: {available_providers}")
        invalid_formats = [fmt for fmt in self.config["output_format"] if fmt not in self.SUPPORTED_FORMATS]
        if invalid_formats:
            available_formats = ", ".join(self.SUPPORTED_FORMATS)
            raise ValueError(f"❌ Unsupported output format(s): {invalid_formats}. Available formats: {available_formats}")
        if self.config["chunk_size"] <= 0:
            raise ValueError("❌ chunk_size must be greater than 0")
        if self.config["save_frequency"] <= 0:
            raise ValueError("❌ save_frequency must be greater than 0")
        self.logger.info("✅ Configuration validated successfully")

    def _initialize_components(self):
        self.logger.info("🔧 Initializing core components...")
        self.llm_provider = LLMProvider(
            api_key=self.config["api_key"],
            provider=self.PROVIDER_MAPPING[self.config["llm_provider"]],
            model=self.config["llm_model"],
            temperature=self.config.get("llm_temperature", 0.3),
            max_tokens=self.config.get("llm_max_tokens", 2000),
        )
        self.logger.info(f"✅ LLM Provider initialized: {self.config['llm_provider']}")
        self.prompt_manager = PromptManager()
        self.logger.info("✅ Prompt Manager initialized")

    def _setup_synthetic_generator(self):
        self.logger.info("🔄 Setting up synthetic data generator...")
        self.synthetic_generator = SyntheticData(
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
            data_dir=self.config["data_dir"],
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            chunking_method=self.config["chunking_method"],
            use_ocr=self.config.get("use_ocr", False),
            azure_endpoint=self.config.get("azure_endpoint"),
            azure_api_key=self.config.get("azure_api_key"),
            output_dir=self.config["output_dir"]
        )
        self.logger.info("✅ Synthetic data generator setup completed")

    def generate_dataset(self, custom_prompt: Optional[str] = None, max_chunks: Optional[int] = None) -> pd.DataFrame:
        try:
            self.logger.info("🚀 Starting synthetic Q&A dataset generation...")
            if custom_prompt:
                self.prompt_manager.add_prompt("QA", custom_prompt)
                self.logger.info("✅ Custom prompt applied")
            dataset = self._generate_with_quality_control(max_chunks)
            self.synthetic_generator.save_dataframe("synthetic_qa_data.csv", mode='a')

            
            
            return dataset
        except Exception as e:
            self.logger.error(f"❌ Error in dataset generation: {str(e)}")
            raise

    def _generate_with_quality_control(self, max_chunks: Optional[int] = None) -> pd.DataFrame:
        chunks = self.synthetic_generator.process_and_chunk_data()
        if not chunks:
            self.logger.warning("⚠️ No chunks found to process")
            return pd.DataFrame()
        if max_chunks and max_chunks < len(chunks):
            chunks = chunks[:max_chunks]
            self.logger.info(f"📋 Processing limited to {max_chunks} chunks")
        if self.config["skip_empty_chunks"]:
            original_count = len(chunks)
            chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) >= self.config["min_chunk_length"]]
            filtered_count = original_count - len(chunks)
            if filtered_count > 0:
                self.logger.info(f"🔍 Filtered out {filtered_count} short chunks")
        successful_generations = 0
        failed_generations = 0
        total_retries = 0
        for i, chunk in enumerate(chunks, 1):
            self.logger.info(f"📝 Processing chunk {i}/{len(chunks)}")
            success = False
            retries = 0
            while not success and retries < self.config["max_retries"]:
                try:
                    qa_data = self.synthetic_generator.generate_synthetic_data(chunk.page_content)
                    self.synthetic_generator.add_to_dataframe(
                        question=qa_data['question'],
                        answer=qa_data['answer'],
                        context=chunk.page_content,
                        chunk_id=chunk.metadata.get('chunk_id', i),
                        source_file=chunk.metadata.get('source_file', 'unknown')
                    )
                    successful_generations += 1
                    success = True
                    if successful_generations % self.config["save_frequency"] == 0:
                        self._save_checkpoint(successful_generations)
                except Exception as e:
                    retries += 1
                    total_retries += 1
                    if retries < self.config["max_retries"]:
                        self.logger.warning(f"⚠️ Error processing chunk {i}, retrying... ({retries}/{self.config['max_retries']}): {str(e)}")
            if not success:
                failed_generations += 1
                self.logger.warning(f"❌ Failed to generate Q&A for chunk {i} after {self.config['max_retries']} retries")
        self.logger.info(f"📊 Generation completed:")
        self.logger.info(f"   • Total chunks: {len(chunks)}")
        self.logger.info(f"   • Successful generations: {successful_generations}")
        self.logger.info(f"   • Failed generations: {failed_generations}")
        self.logger.info(f"   • Total retries: {total_retries}")
        return self.synthetic_generator.qa_dataframe

    def _save_checkpoint(self, count: int):
        try:
            self.synthetic_generator.save_dataframe("synthetic_qa_data.csv", mode='a')
            self.logger.info(f"💾 Checkpoint appended: {count} Q&A pairs")
        except Exception as e:
            self.logger.error(f"❌ Failed to append checkpoint: {str(e)}")
