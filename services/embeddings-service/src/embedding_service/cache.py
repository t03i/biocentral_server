"""
Redis cache component with compression for protein embeddings (ProtT5 and ESM-2)
"""
import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

import blosc2
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    High-performance Redis cache for protein embeddings (ProtT5 and ESM-2) with BLOSC2 compression
    """
    
    def __init__(
        self,
        redis_url: str = "redis://redis-jobs:6379",
        ttl: int = 7 * 24 * 3600,  # 1 week
        key_prefix: str = "protein_emb",
        compression_level: int = 5  # BLOSC2 compression level (1-9)
    ):
        self.redis_url = redis_url
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.compression_level = compression_level
        self._redis: Optional[redis.Redis] = None
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
            "compression_ratio": 0.0
        }
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")
            
    def _sequence_hash(self, sequence: str) -> str:
        """Generate consistent hash for sequence"""
        return hashlib.sha256(sequence.encode('utf-8')).hexdigest()[:16]
        
    def _cache_key(self, sequence_hash: str, model_name: str) -> str:
        """Generate Redis cache key"""
        return f"{self.key_prefix}:{model_name}:{sequence_hash}"
        
    def _compress_embedding(self, embedding: np.ndarray) -> bytes:
        """Compress embedding using BLOSC2"""
        try:
            # Ensure float32 for consistency
            embedding_f32 = embedding.astype(np.float32)
            
            # Store shape information for reconstruction
            shape_bytes = np.array(embedding_f32.shape, dtype=np.int32).tobytes()
            
            # Compress with BLOSC2 (optimized for numerical data)
            compressed_data = blosc2.compress2(
                embedding_f32.tobytes(),
                codec='lz4',  # Fast compression/decompression
                clevel=self.compression_level,
                shuffle=blosc2.Shuffle.BYTE  # Byte shuffle for better compression
            )
            
            # Combine shape info and compressed data
            shape_size = len(shape_bytes)
            result = shape_size.to_bytes(4, 'little') + shape_bytes + compressed_data
            
            # Update compression ratio stats
            original_size = embedding_f32.nbytes
            compressed_size = len(result)
            ratio = compressed_size / original_size if original_size > 0 else 1.0
            self.stats["compression_ratio"] = (
                self.stats["compression_ratio"] * 0.9 + ratio * 0.1
            )  # Exponential moving average
            
            logger.debug(f"Compressed embedding: {original_size}B -> {compressed_size}B "
                        f"(ratio: {ratio:.3f}) shape: {embedding_f32.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
            
    def _decompress_embedding(self, compressed_data: bytes) -> np.ndarray:
        """Decompress embedding from BLOSC2"""
        try:
            # Extract shape information
            shape_size = int.from_bytes(compressed_data[:4], 'little')
            shape_bytes = compressed_data[4:4+shape_size]
            actual_compressed_data = compressed_data[4+shape_size:]
            
            # Reconstruct shape
            shape = tuple(np.frombuffer(shape_bytes, dtype=np.int32))
            
            # Decompress
            decompressed = blosc2.decompress2(actual_compressed_data)
            
            # Reconstruct numpy array with original shape
            embedding = np.frombuffer(decompressed, dtype=np.float32).reshape(shape)
            
            logger.debug(f"Decompressed embedding: {len(compressed_data)}B -> {embedding.nbytes}B "
                        f"shape: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
            
    async def get_embeddings(
        self, 
        sequences: List[str], 
        model_name: str
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Get embeddings from cache for multiple sequences
        
        Args:
            sequences: List of protein sequences
            model_name: Model identifier for cache key (e.g., "prot_t5", "esm2_t33_650M")
            
        Returns:
            Dict mapping sequences to embeddings (None if not cached)
        """
        if not self._redis:
            raise RuntimeError("Redis not connected")
            
        if not sequences:
            return {}
            
        try:
            # Generate cache keys
            sequence_hashes = [self._sequence_hash(seq) for seq in sequences]
            cache_keys = [self._cache_key(hash_val, model_name) for hash_val in sequence_hashes]
            
            # Batch get from Redis
            start_time = time.time()
            compressed_values = await self._redis.mget(cache_keys)
            redis_time = time.time() - start_time
            
            # Process results
            results = {}
            hit_keys = []
            
            for seq, compressed_data in zip(sequences, compressed_values):
                if compressed_data is not None:
                    try:
                        # Decompress and store
                        embedding = self._decompress_embedding(compressed_data)
                        results[seq] = embedding
                        hit_keys.append(self._cache_key(self._sequence_hash(seq), model_name))
                        self.stats["hits"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to decompress cached embedding for sequence: {e}")
                        results[seq] = None
                        self.stats["errors"] += 1
                else:
                    results[seq] = None
                    self.stats["misses"] += 1
                    
            # Refresh TTL for cache hits (sliding window)
            if hit_keys:
                pipe = self._redis.pipeline()
                for key in hit_keys:
                    pipe.expire(key, self.ttl)
                await pipe.execute()
                
            hit_count = len([v for v in results.values() if v is not None])
            logger.info(f"Cache lookup ({model_name}): {hit_count}/{len(sequences)} hits "
                       f"({redis_time*1000:.1f}ms Redis)")
                       
            return results
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            self.stats["errors"] += 1
            # Return empty results on cache failure
            return {seq: None for seq in sequences}
            
    async def set_embeddings(
        self,
        sequence_embeddings: Dict[str, np.ndarray],
        model_name: str
    ) -> bool:
        """
        Store embeddings in cache with compression
        
        Args:
            sequence_embeddings: Dict mapping sequences to embeddings
            model_name: Model identifier for cache key (e.g., "prot_t5", "esm2_t33_650M")
            
        Returns:
            Success status
        """
        if not self._redis:
            raise RuntimeError("Redis not connected")
            
        if not sequence_embeddings:
            return True
            
        try:
            start_time = time.time()
            
            # Prepare batch operations
            pipe = self._redis.pipeline()
            
            for sequence, embedding in sequence_embeddings.items():
                # Compress embedding
                compressed_data = self._compress_embedding(embedding)
                
                # Set in Redis with TTL
                cache_key = self._cache_key(self._sequence_hash(sequence), model_name)
                pipe.setex(cache_key, self.ttl, compressed_data)
                
            # Execute batch
            await pipe.execute()
            
            cache_time = time.time() - start_time
            self.stats["sets"] += len(sequence_embeddings)
            
            logger.info(f"Cached {len(sequence_embeddings)} embeddings ({model_name}) "
                       f"({cache_time*1000:.1f}ms)")
                       
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            self.stats["errors"] += 1
            return False
            
    async def get_cache_stats(self) -> Dict:
        """Get cache statistics and Redis info"""
        stats = self.stats.copy()
        
        if self._redis:
            try:
                # Get Redis memory info
                info = await self._redis.info("memory")
                stats.update({
                    "redis_memory_used": info.get("used_memory", 0),
                    "redis_memory_human": info.get("used_memory_human", "0B"),
                    "redis_connected": True
                })
                
                # Calculate hit rate
                total_requests = stats["hits"] + stats["misses"]
                stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0.0
                
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")
                stats["redis_connected"] = False
        else:
            stats["redis_connected"] = False
            
        return stats
        
    async def clear_cache(self, pattern: str = None) -> int:
        """
        Clear cache entries
        
        Args:
            pattern: Redis key pattern to match (default: all embedding keys)
            
        Returns:
            Number of keys deleted
        """
        if not self._redis:
            raise RuntimeError("Redis not connected")
            
        try:
            pattern = pattern or f"{self.key_prefix}:*"
            
            # Find matching keys
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)
                
            if keys:
                deleted = await self._redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries matching '{pattern}'")
                return deleted
            else:
                logger.info(f"No cache entries found matching '{pattern}'")
                return 0
                
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            raise
            
    async def health_check(self) -> Dict[str, bool]:
        """Check cache health"""
        if not self._redis:
            return {"connected": False, "responsive": False}
            
        try:
            # Test basic operations
            test_key = f"{self.key_prefix}:health_check"
            await self._redis.set(test_key, "ok", ex=10)
            value = await self._redis.get(test_key)
            await self._redis.delete(test_key)
            
            return {
                "connected": True,
                "responsive": value == b"ok"
            }
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {"connected": False, "responsive": False}