# 🧠 Muffakir RAG

## Advanced Arabic Retrieval-Augmented Generation (RAG) Library

**Muffakir RAG** is a powerful Python library designed specifically for building sophisticated Retrieval-Augmented Generation (RAG) systems tailored to the Arabic language. It supports advanced document processing, semantic search, and intelligent answer generation powered by multiple LLM providers.

---

## ✨ Key Features

- 🌟 **Arabic Language Focus**: Optimized for accurate processing of Arabic texts  
- 🤖 **Multi-Provider Support**: Seamless integration with Together AI, OpenAI, Groq, and Open Router  
- 📚 **Advanced Document Processing**: Handles PDF (including OCR for scanned documents), DOCX, TXT, and images with OCR support  
- 🔍 **Smart Retrieval**: Multiple retrieval methods including hybrid and contextual search, with built-in reranking  
- ⚡ **Simple API**: Intuitive interface for quick integration and usage  
- 🛡️ **Hallucination Check**: Validates answers to reduce hallucinations  
- 🔄 **Query Transformer**: Automatically optimizes user queries for better retrieval  
- 🔄 **Reranker**: Enhances retrieval results through semantic similarity reranking  

---

## 🚀 Installation

```bash
pip install Muffakir
````

For development:

```bash
git clone https://github.com/yourusername/muffakir-rag.git
cd muffakir-rag
pip install -e .
```

---

## 📖 Quick Start

```python
from Muffakir import MuffakirRAG

config = {
    "data_dir": "path/to/your/documents",
    "llm_provider": "together",
    "api_key": "your_api_key_here",
    "embedding_model": "mohamed2811/Muffakir_Embedding",
    "k": 5,
    "query_transformer": True,
    "hallucination_check": True,
    "reranking": True
}

rag = MuffakirRAG(config)

response = rag.ask("What is the definition of artificial intelligence?")
print(response["answer"])
```

---

## 🔧 Configuration Parameters

### Core

| Parameter         | Description              | Default                            | Required |
| ----------------- | ------------------------ | ---------------------------------- | -------- |
| `data_dir`        | Path to documents folder | None                               | Yes      |
| `api_key`         | API key for LLM provider | None                               | Yes      |
| `llm_provider`    | Language model provider  | `"together"`                       | Yes      |
| `embedding_model` | Embedding model to use   | `"mohamed2811/Muffakir_Embedding"` | No       |

### Advanced Options

```python
config = {
    # Basic
    "data_dir": "documents/",
    "api_key": "your_api_key",
    "llm_provider": "together",

    # LLM
    "llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "llm_temperature": 0.0,
    "llm_max_tokens": 1000,

    # Text Processing
    "chunk_size": 600,
    "chunk_overlap": 200,
    "chunking_method": "recursive",

    # Retrieval
    "retrieval_method": "max_marginal_relevance_search",
    "k": 5,
    "fetch_k": 15,

    # Features
    "query_transformer": True,
    "hallucination_check": True,
    "reranking": True,
    "reranking_method": "semantic_similarity"
}
```

---

## 🎯 Usage Examples

### Search Similar Documents

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

### Add New Documents

```python
new_docs = ["path/to/new_doc1.pdf", "path/to/new_doc2.docx"]
success = rag.add_documents(new_docs)

if success:
    print("Documents added successfully!")
```

### Customize Parameters for Asking

```python
response = rag.ask(
    "Explain neural networks.",
    k=10,
    retrieval_method="hybrid",
    temperature=0.3
)
```

---

## 🏗️ Project Structure

```
muffakir-rag/
├── Muffakir/              # Core module
├── TextProcessor/         # Document processing and chunking
├── LLMProvider/           # Language model interfaces
├── Embedding/             # Embedding models and management
├── Generation/            # Answer generation pipeline
├── VectorDB/              # Vector database management
├── Reranker/              # Semantic reranking
├── RAGPipeline/           # End-to-end RAG pipeline control
├── PromptManager/         # Prompt templates and management
└── QueryTransformer/      # Query optimization and transformation
```

---

## 🔌 Supported Providers

| Provider    | Enum Name    | Description              |
| ----------- | ------------ | ------------------------ |
| Together AI | `TOGETHER`   | Together AI LLM provider |
| OpenAI      | `OPENAI`     | OpenAI GPT models        |
| Groq        | `GROQ`       | Groq's AI platform       |
| Open Router | `OPENROUTER` | Open Router API          |

---

## 📄 Supported File Types

* **PDF**: Including OCR for scanned documents
* **DOCX**: Microsoft Word files
* **TXT**: Plain text files
* **Images**: Processed with Azure Computer Vision OCR

---

## 📊 Performance

* ⚡ Fast processing of large-scale Arabic documents
* 🎯 Optimized for high accuracy on Arabic text
* 💾 Efficient memory and resource usage
* 🔄 Scalable to thousands of documents

---

## 🤝 Contribution

Contributions are welcome! Please:

1. Fork the repository
2. Create a new feature branch
3. Add your improvements and tests
4. Submit a pull request for review

---


---

<div align="center">

**Built with ❤️ for the Arabic community**

</div>
```
