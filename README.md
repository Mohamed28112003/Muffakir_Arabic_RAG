# Muffakir Arabic RAG Library

## Advanced Arabic Retrieval-Augmented Generation (RAG) Library

**Muffakir RAG** is a powerful Python library designed specifically for building sophisticated Retrieval-Augmented Generation (RAG) systems tailored to the Arabic language. It supports advanced document processing, semantic search, and intelligent answer generation powered by multiple LLM providers.

This `README.md` provides comprehensive documentation and usage guides for the Muffakir Arabic RAG library, covering its core components: `MuffakirRAG`, `MuffakirSearch`, and `MuffakirSyntheticData`.




## Features

*   🌟 **Arabic Language Focus**: Optimized for accurate processing of Arabic texts
*   🤖 **Multi-Provider Support**: Seamless integration with Together AI, OpenAI, Groq, and Open Router
*   📚 **Advanced Document Processing**: Handles PDF (including OCR for scanned documents), DOCX, TXT, and images with OCR support
*   🔍 **Smart Retrieval**: Multiple retrieval methods including hybrid and contextual search, with built-in reranking
*   ⚡ **Simple API**: Intuitive interface for quick integration and usage
*   🛡️ **Hallucination Check**: Validates answers to reduce hallucinations
*   🔄 **Query Transformer**: Automatically optimizes user queries for better retrieval
*   🔄 **Reranker**: Enhances retrieval results through semantic similarity reranking




## Installation

You can install the Muffakir library using pip:

```bash
pip install Muffakir
```

For development, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/Mohamed28112003/Muffakir_Arabic_RAG.git
cd Muffakir_Arabic_RAG
pip install -e .
```




## Usage

### MuffakirRAG

`MuffakirRAG` is the core component for building Retrieval-Augmented Generation systems. Below are examples of how to initialize and use `MuffakirRAG`.

#### Initialization

```python
from Muffakir import MuffakirRAG

config = {
    "data_dir": "path/to/your/documents",
    "llm_provider": "your_provider",
    "api_key": "your_api_key_here",
    "embedding_model": "mohamed2811/Muffakir_Embedding",
    "llm_model":"your_model"
    "k": 5,
    "query_transformer": True,
    "hallucination_check": True,
    "reranking": True
}

rag = MuffakirRAG(config)
```

#### Asking Questions

```python
response = rag.ask("ما هو الذكاء الاصطناعي ؟")
print(response["answer"])
```

#### Customizing Parameters for Asking

You can customize parameters like `k`, `retrieval_method`, and `temperature` when asking questions:

```python
response = rag.ask(
    "اشرح القانون الجنائي",
    k=10,
    retrieval_method="hybrid",
    temperature=0.3
)
```

#### Getting Similar Documents

```python
similar_docs = rag.get_similar_documents(
    query="القانون الجنائي",
    k=3,
    method="similarity_search"
)

for doc in similar_docs:
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Content: {doc.page_content[:200]}...")
```

#### Adding New Documents

```python
new_docs = ["path/to/new_doc1.pdf", "path/to/new_doc2.docx"]
success = rag.add_documents(new_docs)

if success:
    print("Documents added successfully!")
```




### MuffakirSearch

`MuffakirSearch` provides capabilities for web search and information retrieval, leveraging an LLM for enhanced results. It integrates with `Firecrawl` for web crawling.

#### Initialization

```python
from Muffakir import MuffakirSearch

config = {
    "api_key": "your_llm_api_key",
    "fire_crawl_api": "your_firecrawl_api_key",
    "llm_provider": "your_provider",
    "llm_model": "your_model",
    "llm_temperature": 0.0,
    "llm_max_tokens": 1000,
    "max_depth": 1,
    "time_limit": 30,
    "max_urls": 5,
}

search_agent = MuffakirSearch(config)
```

#### Performing a Search

To perform a search, use the `search` method:

```python
results = search_agent.search("Latest news on AI advancements in Arabic")
print(results)
```




### MuffakirSyntheticData

`MuffakirSyntheticData` is designed to generate synthetic Question-Answering (Q&A) datasets from your documents, which can be invaluable for training and evaluating RAG systems.

#### Initialization

```python
from Muffakir import MuffakirSyntheticData

config = {
    "data_dir": "path/to/your/documents",
    "api_key": "your_llm_api_key",
    "llm_provider": "your_provider",
    "llm_model":"your_model"
    "llm_temperature": 0.3,
    "llm_max_tokens": 2000,
    "chunk_size": 600,
    "chunk_overlap": 200,
    "chunking_method": "recursive",
    "use_ocr": False,
    "output_dir": "./muffakir_synthetic_data",
    "save_frequency": 10,

}

synthetic_data_generator = MuffakirSyntheticData(config)

```

#### Generating a Dataset

Use the `generate_dataset` method to create your synthetic Q&A dataset. You can optionally provide a `custom_prompt` for Q&A generation and limit the `max_chunks` to process.

```python
dataset = synthetic_data_generator.generate_dataset(
    
    max_chunks=100
)
print(dataset.head())
```




## Configuration Parameters

The `Muffakir` library components are highly configurable. Below is a comprehensive list of parameters you can set in the configuration dictionary when initializing `MuffakirRAG`, `MuffakirSearch`, or `MuffakirSyntheticData`.

| Parameter | Description | Default | Required |
| --- | --- | --- | --- |
| `data_dir` | Path to documents folder for RAG or synthetic data generation. | `None` | Yes (for RAG and Synthetic Data) |
| `api_key` | API key for the chosen LLM provider. | `None` | Yes |
| `fire_crawl_api` | API key for Firecrawl, used by `MuffakirSearch` for web crawling. | `None` | Yes (for MuffakirSearch) |
| `llm_provider` | Language model provider to use (e.g., `"together"`, `"openai"`, `"groq"`, `"open_router"`). | `"together"` | Yes |
| `llm_model` | Specific LLM model to use from the chosen provider. | `None` | Yes |
| `llm_temperature` | Controls the randomness of the LLM's output. Higher values mean more random. | `0.0` (for RAG/Search), `0.3` (for Synthetic Data) | No |
| `llm_max_tokens` | Maximum number of tokens to generate in the LLM's response. | `1000` (for RAG/Search), `2000` (for Synthetic Data) | No |
| `embedding_model` | Embedding model to use for document vectorization. | `"mohamed2811/Muffakir_Embedding"` | No |
| `k` | Number of top relevant documents to retrieve. | `5` | No |
| `query_transformer` | Boolean to enable/disable query transformation for better retrieval. | `True` | No |
| `hallucination_check` | Boolean to enable/disable hallucination checking for generated answers. | `True` | No |
| `reranking` | Boolean to enable/disable reranking of retrieved documents. | `True` | No |
| `reranking_method` | Method used for reranking (e.g., `"semantic_similarity"`). | `"semantic_similarity"` | No |
| `chunk_size` | Size of text chunks for document processing. | `600` | No |
| `chunk_overlap` | Overlap between text chunks. | `200` | No |
| `chunking_method` | Method used for text chunking (e.g., `"recursive"`). | `"recursive"` | No |
| `use_ocr` | Boolean to enable/disable OCR for scanned documents and images. | `False` | No |
| `azure_endpoint` | Azure Computer Vision endpoint for OCR. Required if `use_ocr` is `True` and processing images. | `None` | No |
| `azure_api_key` | Azure Computer Vision API key for OCR. Required if `use_ocr` is `True` and processing images. | `None` | No |
| `output_dir` | Directory to save generated synthetic data. | `./muffakir_synthetic_data` | No (for Synthetic Data) |
| `save_frequency` | How often to save checkpoints during synthetic data generation (in number of Q&A pairs). | `10` | No (for Synthetic Data) |
| `output_format` | List of formats to save synthetic data (e.g., `["csv", "excel", "json"]`). | `["csv", "excel"]` | No (for Synthetic Data) |
| `max_retries` | Maximum retries for generating Q&A for a chunk. | `3` | No (for Synthetic Data) |
| `skip_empty_chunks` | Boolean to skip chunks that are too short. | `True` | No (for Synthetic Data) |
| `min_chunk_length` | Minimum length of a chunk to be processed for synthetic data generation. | `50` | No (for Synthetic Data) |
| `min_question_length` | Minimum length of a generated question. | `10` | No (for Synthetic Data) |
| `min_answer_length` | Minimum length of a generated answer. | `15` | No (for Synthetic Data) |
| `validate_qa_pairs` | Boolean to validate generated Q&A pairs. | `True` | No (for Synthetic Data) |
| `max_depth` | Maximum depth for web crawling in `MuffakirSearch`. | `1` | No (for MuffakirSearch) |
| `time_limit` | Time limit for web crawling in `MuffakirSearch` (in seconds). | `30` | No (for MuffakirSearch) |
| `max_urls` | Maximum number of URLs to crawl in `MuffakirSearch`. | `5` | No (for MuffakirSearch) |




## Advanced Usage

### Supported LLM Providers

The `Muffakir` library supports integration with several LLM providers. You can specify your desired provider in the configuration using the `llm_provider` parameter. The available providers and their corresponding `Enum Name` are:

| Provider | Enum Name | Description |
| --- | --- | --- |
| Together AI | `TOGETHER` | Together AI LLM provider |
| OpenAI | `OPENAI` | OpenAI GPT models |
| Groq | `GROQ` | Groq's AI platform |
| Open Router | `OPENROUTER` | Open Router API |

### Supported Document Types

The `Muffakir` library's advanced document processing capabilities support a variety of document types, including:

*   **PDF**: Including OCR for scanned documents
*   **DOCX**: Microsoft Word files
*   **TXT**: Plain text files
*   **Images**: Processed with Azure Computer Vision OCR




## Contributing

Contributions are welcome! If you'd like to contribute to the `Muffakir` library, please follow these steps:

1.  Fork the repository
2.  Create a new feature branch (`git checkout -b feature/YourFeatureName`)
3.  Add your improvements and tests
4.  Commit your changes (`git commit -m 'Add new feature'`) 
5.  Push to the branch (`git push origin feature/YourFeatureName`)
6.  Submit a pull request for review




## License

This project is licensed under the MIT License.


## Contact Me

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Email**: [mohamedtawfik28112003@gmail.com](mailto:mohamedtawfik28112003@gmail.com)
- **LinkedIn**: [www.linkedin.com/in/mohamedkhaled2811](https://www.linkedin.com/in/mohamedkhaled2811)
