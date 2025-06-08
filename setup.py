from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="Muffakir",
    version="0.1.4",
    author="Mohamed",
    description="Arabic Retrieval-Augmented Generation Library",
    long_description=long_description,
    long_description_content_type="text/markdown",  # لو README.md
    packages=find_packages(),
    install_requires=[
        "azure-ai-formrecognizer==3.3.3",
        "chromadb==0.6.3",
        "fastembed==0.5.1",
        "langchain==0.3.17",
        "langchain-chroma==0.2.1",
        "langchain-community==0.3.16",
        "langchain-openai==0.3.3",
        "langchain-groq==0.2.4",
        "nltk==3.9.1",
        "rank-bm25==0.2.2",
        "sentence-transformers==3.4.1",
        "tiktoken==0.8.0",
        "transformers==4.48.2",
        "unstructured[pdf]==0.16.17",
        "pydantic==2.11.3",
        "python-dotenv==1.0.1",
        "pydantic-settings==2.7.1",
        "spacy==3.8.7"
    ],
    python_requires='>=3.8',
)
