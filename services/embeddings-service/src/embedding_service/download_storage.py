"""
Download storage and retrieval functionality for embeddings
"""
import json
import logging
import numpy as np
from typing import List, Dict, Optional

from .cache import EmbeddingCache
from .config import config

logger = logging.getLogger(__name__)


class DownloadStorage:
    """Handles storage and retrieval of embeddings for download"""
    
    def __init__(self, embedding_cache: EmbeddingCache):
        self.cache = embedding_cache
    
    async def store_for_download(
        self, 
        request_id: str, 
        embeddings: List, 
        sequences: List[str], 
        model_config: Dict
    ) -> None:
        """
        Store embeddings as binary data in Redis for download with TTL
        
        Args:
            request_id: Unique identifier for the download request
            embeddings: List of embedding arrays
            sequences: List of protein sequences
            model_config: Model configuration containing precision info
        """
        if not self.cache:
            logger.warning("Cache not available, cannot store for download")
            return
        
        try:
            # Store each embedding as a separate array (sequences typically have different lengths)
            # Convert each embedding to binary and store as a list
            binary_embeddings = []
            
            # Determine the target dtype based on model precision
            model_precision = model_config.get("precision", "fp16").lower()
            if model_precision == "fp16":
                target_dtype = np.float16
            elif model_precision == "fp32":
                target_dtype = np.float32
            else:
                # Default to fp16 if precision is unknown
                target_dtype = np.float16
                logger.warning(f"Unknown model precision '{model_precision}', defaulting to fp16")
            
            for emb in embeddings:
                # Ensure the embedding is a numpy array with model's native dtype
                if not isinstance(emb, np.ndarray):
                    emb_array = np.array(emb, dtype=target_dtype)
                else:
                    emb_array = emb.astype(target_dtype)
                
                binary_embeddings.append({
                    'shape': emb_array.shape,
                    'dtype': str(emb_array.dtype),
                    'data': emb_array.tobytes().hex()  # Convert bytes to hex string for JSON
                })
            
            binary_data = json.dumps(binary_embeddings)
            shape_info = {
                'storage_type': 'list_of_arrays',
                'count': len(binary_embeddings)
            }
            
            # Store binary embeddings and shape info
            binary_key = f"download:{request_id}"
            shape_key = f"download:{request_id}:shape"
            
            await self.cache._redis.setex(
                binary_key,
                config.download_ttl_minutes * 60,  # TTL in seconds
                binary_data  # Store as JSON string
            )
            
            await self.cache._redis.setex(
                shape_key,
                config.download_ttl_minutes * 60,  # TTL in seconds
                json.dumps(shape_info)  # Store shape info as JSON
            )
            
            logger.debug(f"Stored {len(binary_embeddings)} embeddings for {request_id} with {config.download_ttl_minutes}min TTL")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
    
    async def retrieve_for_download(self, request_id: str) -> Optional[List[np.ndarray]]:
        """
        Retrieve stored embeddings for download
        
        Args:
            request_id: Unique identifier for the download request
            
        Returns:
            List of numpy arrays containing the embeddings, or None if not found
        """
        if not self.cache:
            logger.warning("Cache not available, cannot retrieve for download")
            return None
        
        try:
            # Retrieve stored binary embeddings and shape info from Redis
            binary_key = f"download:{request_id}"
            shape_key = f"download:{request_id}:shape"
            
            binary_data = await self.cache._redis.get(binary_key)
            shape_data = await self.cache._redis.get(shape_key)
            
            if not binary_data or not shape_data:
                logger.warning(f"Download not found or expired for request_id: {request_id}")
                return None
            
            # Parse shape info and reconstruct embeddings
            shape_info = json.loads(shape_data)
            binary_embeddings_list = json.loads(binary_data)
            
            embeddings_array = []
            for item in binary_embeddings_list:
                shape = tuple(item['shape'])
                dtype = item['dtype']
                data = bytes.fromhex(item['data'])  # Convert hex string back to bytes
                embeddings_array.append(np.frombuffer(data, dtype=dtype).reshape(shape))
            
            logger.debug(f"Retrieved {len(embeddings_array)} embeddings for {request_id}")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings: {e}")
            return None
    
    async def cleanup_expired_downloads(self) -> int:
        """
        Clean up expired download entries from Redis
        
        Returns:
            Number of cleaned up entries
        """
        if not self.cache:
            return 0
        
        try:
            # Find all download keys
            pattern = "download:*"
            keys = await self.cache._redis.keys(pattern)
            
            # Keys with TTL will be automatically cleaned up by Redis
            # This is just for logging purposes
            expired_count = 0
            for key in keys:
                ttl = await self.cache._redis.ttl(key)
                if ttl == -1:  # No TTL set
                    logger.warning(f"Download key {key} has no TTL set")
                elif ttl == -2:  # Key doesn't exist (shouldn't happen)
                    expired_count += 1
            
            logger.debug(f"Download cleanup check completed, {expired_count} expired keys found")
            return expired_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup downloads: {e}")
            return 0 