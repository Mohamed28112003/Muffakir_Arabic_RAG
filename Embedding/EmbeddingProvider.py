from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
import hashlib
import json
import os
import asyncio
from functools import lru_cache
from langchain.embeddings.base import Embeddings  # Import the base Embeddings class

class EmbeddingProvider(Embeddings):  # Inherit from LangChain's Embeddings
    def __init__(self, model_name: str = 'mohamed2811/Muffakir_Embedding', 
                 cache_dir: str = '.embedding_cache',
                 batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a unique cache key for the input text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _check_cache(self, text: str) -> Optional[List[float]]:
        """Check if embedding exists in cache"""
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Cache read error: {e}")
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]):
        """Save embedding to cache"""
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(embedding, f)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    @lru_cache(maxsize=100)  
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text with caching"""
     
        cached = self._check_cache(text)
        if cached:
            return cached
        
        embedding = self.model.encode(text).tolist()
        self._save_to_cache(text, embedding)
        return embedding
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with batching and caching"""
        results = []
        batch = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = self._check_cache(text)
            if cached:
                results.append(cached)
            else:
                results.append(None)  
                batch.append(text)
                uncached_indices.append(i)
        
        if batch:
            for i in range(0, len(batch), self.batch_size):
                sub_batch = batch[i:i+self.batch_size]
                sub_indices = uncached_indices[i:i+self.batch_size]
                
                embeddings = self.model.encode(sub_batch).tolist()
                
                for j, (text, embedding) in enumerate(zip(sub_batch, embeddings)):
                    self._save_to_cache(text, embedding)
                    results[sub_indices[j]] = embedding
        
        return results
    
    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, texts)
    
    # Required LangChain Embeddings interface methods
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the underlying model"""
        return self.embed(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the underlying model"""
        return self.embed_single(text)