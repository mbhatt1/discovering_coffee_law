"""
Embedding utilities for interacting with OpenAI's embedding API.
"""

import numpy as np
from openai import OpenAI
from typing import Optional
import time
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result from an embedding request."""
    embedding: np.ndarray
    model: str
    usage_tokens: int


class EmbeddingClient:
    """
    Client for generating embeddings using OpenAI's API.

    Handles rate limiting, caching, and batch processing.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "text-embedding-3-small",
        cache_enabled: bool = True,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize the embedding client.

        Args:
            client: OpenAI client (creates new one if not provided)
            model: Embedding model to use
            cache_enabled: Whether to cache embeddings
            rate_limit_delay: Minimum delay between API calls
        """
        self.client = client or OpenAI()
        self.model = model
        self.cache_enabled = cache_enabled
        self.rate_limit_delay = rate_limit_delay
        self._cache: dict[str, np.ndarray] = {}
        self._last_call_time = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Ensure minimum delay between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_call_time = time.time()

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"{self.model}:{hash(text)}"

    def get_embedding(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use cached result if available

        Returns:
            EmbeddingResult with embedding and metadata
        """
        # Check cache
        if self.cache_enabled and use_cache:
            key = self._cache_key(text)
            if key in self._cache:
                return EmbeddingResult(
                    embedding=self._cache[key],
                    model=self.model,
                    usage_tokens=0  # Cached
                )

        # Rate limiting
        self._wait_for_rate_limit()

        # API call
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        embedding = np.array(response.data[0].embedding)

        # Cache result
        if self.cache_enabled:
            self._cache[self._cache_key(text)] = embedding

        return EmbeddingResult(
            embedding=embedding,
            model=self.model,
            usage_tokens=response.usage.total_tokens
        )

    def get_embeddings_batch(
        self,
        texts: list[str],
        use_cache: bool = True
    ) -> list[EmbeddingResult]:
        """
        Get embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached results

        Returns:
            List of EmbeddingResults
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            if self.cache_enabled and use_cache:
                key = self._cache_key(text)
                if key in self._cache:
                    results.append((i, EmbeddingResult(
                        embedding=self._cache[key],
                        model=self.model,
                        usage_tokens=0
                    )))
                    continue
            uncached_texts.append(text)
            uncached_indices.append(i)

        # Batch API call for uncached texts
        if uncached_texts:
            self._wait_for_rate_limit()

            # OpenAI supports batching up to ~2000 texts
            batch_size = 100
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]

                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )

                for j, emb_data in enumerate(response.data):
                    embedding = np.array(emb_data.embedding)
                    text = batch_texts[j]
                    idx = batch_indices[j]

                    # Cache
                    if self.cache_enabled:
                        self._cache[self._cache_key(text)] = embedding

                    results.append((idx, EmbeddingResult(
                        embedding=embedding,
                        model=self.model,
                        usage_tokens=response.usage.total_tokens // len(batch_texts)
                    )))

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between embeddings of two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity in range [-1, 1]
        """
        emb1 = self.get_embedding(text1).embedding
        emb2 = self.get_embedding(text2).embedding

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)
