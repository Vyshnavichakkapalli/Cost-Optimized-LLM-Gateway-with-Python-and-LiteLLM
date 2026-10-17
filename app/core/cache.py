import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import redis.asyncio as aioredis
import litellm
from litellm import aembedding
from app.config import CachingConfig, GatewayConfig, get_gateway_config, get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheLookupResult:
    is_hit: bool
    response: Optional[str] = None
    model_used: Optional[str] = None
    similarity: float = 0.0
    embedding_cost_usd: float = 0.0
    embedding_vector: Optional[List[float]] = None
    cached_prompt: Optional[str] = None


class SemanticCacheManager:
    """
    Manages semantic vector caching using Redis RediSearch.
    Falls back gracefully to an in-memory vector store if Redis is unreachable.
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or get_gateway_config()
        self.settings = get_settings()
        self.redis: Optional[aioredis.Redis] = None
        self.index_name = self.config.caching.index_name
        self.is_connected = False

        # In-memory fallback cache store: list of dicts {prompt, response, model_used, vector}
        self._in_memory_cache: List[Dict[str, Any]] = []

    async def connect(self) -> bool:
        """Connects to Redis and verifies RediSearch index exists."""
        if not self.config.caching.enabled:
            logger.info("Semantic caching is disabled in configuration.")
            return False

        try:
            self.redis = aioredis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD or None,
                decode_responses=False,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            # Test ping
            await self.redis.ping()
            self.is_connected = True
            logger.info(f"Connected to Redis at {self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}")
            await self._ensure_index()
            return True
        except Exception as e:
            logger.warning(
                f"Could not connect to Redis ({e}). Semantic caching will operate in in-memory fallback mode."
            )
            self.is_connected = False
            return False

    async def disconnect(self):
        """Closes Redis connection."""
        if self.redis:
            try:
                await self.redis.aclose()
            except Exception:
                pass
            self.is_connected = False

    async def _ensure_index(self):
        """Ensures the RediSearch vector index exists in Redis."""
        if not self.redis or not self.is_connected:
            return

        try:
            # Check if index exists using FT.INFO
            await self.redis.execute_command("FT.INFO", self.index_name)
            logger.info(f"RediSearch index '{self.index_name}' already exists.")
        except Exception as e:
            # Create index if it does not exist
            err_msg = str(e).lower()
            if "unknown index" in err_msg or "no such index" in err_msg:
                dim = self.config.caching.embedding_dimension
                logger.info(f"Creating RediSearch index '{self.index_name}' with dim={dim}...")
                try:
                    await self.redis.execute_command(
                        "FT.CREATE", self.index_name,
                        "ON", "HASH",
                        "PREFIX", "1", "prompt:",
                        "SCHEMA",
                        "prompt", "TEXT",
                        "response", "TEXT",
                        "model_used", "TEXT",
                        "prompt_vector", "VECTOR", "FLAT", "6",
                        "TYPE", "FLOAT32",
                        "DIM", str(dim),
                        "DISTANCE_METRIC", "COSINE"
                    )
                    logger.info(f"Successfully created RediSearch index '{self.index_name}'.")
                except Exception as create_err:
                    logger.error(f"Failed to create RediSearch index: {create_err}")
            else:
                logger.warning(f"Error checking RediSearch index: {e}")

    async def generate_embedding(self, text: str) -> Tuple[List[float], float]:
        """
        Generates prompt embedding using LiteLLM aembedding.
        Returns:
            (embedding_vector, embedding_cost_usd)
        """
        model = self.config.models.embedding
        try:
            response = await aembedding(model=model, input=[text])
            vector = response.data[0]["embedding"]

            # Calculate embedding cost
            cost = 0.0
            try:
                litellm_cost = litellm.completion_cost(completion_response=response)
                if litellm_cost is not None and litellm_cost >= 0.0:
                    cost = float(litellm_cost)
            except Exception:
                pass

            if cost == 0.0:
                # text-embedding-3-small standard pricing ~$0.00002 / 1k tokens
                tokens = getattr(response, "usage", {}).get("total_tokens", len(text) // 4)
                cost = (tokens / 1000.0) * 0.00002

            return vector, round(cost, 7)
        except Exception as e:
            logger.warning(f"Failed to generate embedding with {model} ({e}). Generating fallback vector.")
            # Deterministic pseudo-vector for fallback/offline mode
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            dim = self.config.caching.embedding_dimension
            vec = rng.randn(dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist(), 0.0

    async def search(self, prompt: str) -> CacheLookupResult:
        """
        Checks cache for semantically similar prompt.
        If cosine similarity >= similarity_threshold, returns cache hit.
        """
        if not self.config.caching.enabled:
            return CacheLookupResult(is_hit=False)

        vector, embedding_cost = await self.generate_embedding(prompt)
        threshold = self.config.caching.similarity_threshold

        # 1. Try Redis RediSearch
        if self.is_connected and self.redis:
            try:
                vector_bytes = np.array(vector, dtype=np.float32).tobytes()
                # RediSearch KNN search query
                query = "*=>[KNN 1 @prompt_vector $vec AS score]"
                results = await self.redis.execute_command(
                    "FT.SEARCH", self.index_name,
                    query,
                    "PARAMS", "2", "vec", vector_bytes,
                    "SORTBY", "score",
                    "RETURN", "4", "response", "model_used", "prompt", "score",
                    "DIALECT", "2"
                )

                if results and len(results) > 1 and results[0] > 0:
                    # results[1] is doc_id, results[2] is list of field-value pairs
                    fields = results[2]
                    field_dict = {}
                    for i in range(0, len(fields), 2):
                        k = fields[i]
                        v = fields[i + 1]
                        if isinstance(k, bytes):
                            k = k.decode("utf-8", errors="ignore")
                        if isinstance(v, bytes):
                            v = v.decode("utf-8", errors="ignore")
                        field_dict[k] = v

                    # RediSearch COSINE distance D = 1 - cosine_similarity
                    distance = float(field_dict.get("score", 1.0))
                    similarity = max(0.0, min(1.0, 1.0 - distance))

                    if similarity >= threshold:
                        logger.info(
                            f"Semantic Cache HIT! (similarity={similarity:.4f} >= threshold={threshold})"
                        )
                        return CacheLookupResult(
                            is_hit=True,
                            response=field_dict.get("response", ""),
                            model_used=field_dict.get("model_used", "cached"),
                            similarity=similarity,
                            embedding_cost_usd=embedding_cost,
                            cached_prompt=field_dict.get("prompt", ""),
                        )
            except Exception as e:
                logger.warning(f"Redis vector search encountered an error: {e}. Checking in-memory cache.")

        # 2. In-memory fallback cache
        if self._in_memory_cache:
            v_query = np.array(vector, dtype=np.float32)
            v_norm = np.linalg.norm(v_query)
            if v_norm > 0:
                v_query = v_query / v_norm

            best_sim = -1.0
            best_entry = None

            for entry in self._in_memory_cache:
                c_vec = np.array(entry["vector"], dtype=np.float32)
                c_norm = np.linalg.norm(c_vec)
                if c_norm > 0:
                    c_vec = c_vec / c_norm
                sim = float(np.dot(v_query, c_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry and best_sim >= threshold:
                logger.info(f"In-Memory Cache HIT! (similarity={best_sim:.4f} >= threshold={threshold})")
                return CacheLookupResult(
                    is_hit=True,
                    response=best_entry["response"],
                    model_used=best_entry["model_used"],
                    similarity=best_sim,
                    embedding_cost_usd=embedding_cost,
                    cached_prompt=best_entry["prompt"],
                )

        # Cache Miss
        return CacheLookupResult(
            is_hit=False,
            embedding_cost_usd=embedding_cost,
            embedding_vector=vector,
        )

    async def save(
        self,
        prompt: str,
        response: str,
        model_used: str,
        embedding_vector: Optional[List[float]] = None,
    ):
        """Saves a new prompt-response pair and its vector embedding to the cache."""
        if not self.config.caching.enabled:
            return

        vector = embedding_vector
        if vector is None:
            vector, _ = await self.generate_embedding(prompt)

        ttl = self.config.caching.ttl_seconds
        doc_id = f"prompt:{uuid.uuid4()}"

        # 1. Save to Redis
        if self.is_connected and self.redis:
            try:
                vector_bytes = np.array(vector, dtype=np.float32).tobytes()
                mapping = {
                    "prompt": prompt,
                    "response": response,
                    "model_used": model_used,
                    "prompt_vector": vector_bytes,
                }
                await self.redis.hset(doc_id, mapping=mapping)
                await self.redis.expire(doc_id, ttl)
                logger.info(f"Saved prompt vector to Redis cache ({doc_id}) with TTL={ttl}s.")
            except Exception as e:
                logger.warning(f"Failed to save to Redis cache: {e}")

        # 2. Always keep in in-memory fallback cache as well
        self._in_memory_cache.append({
            "prompt": prompt,
            "response": response,
            "model_used": model_used,
            "vector": vector,
        })
        # Keep in-memory cache bounded to 1000 items
        if len(self._in_memory_cache) > 1000:
            self._in_memory_cache.pop(0)


_cache_manager: Optional[SemanticCacheManager] = None


def get_cache_manager() -> SemanticCacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = SemanticCacheManager()
    return _cache_manager
