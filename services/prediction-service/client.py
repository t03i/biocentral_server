"""
HTTP client for embedding service communication.
"""
import logging
import time
from typing import List, Dict, Any

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class EmbeddingServiceClient:
    """Client for communicating with the embedding service."""
    
    def __init__(self, base_url: str = "http://embeddings-service:8001"):
        self.base_url = base_url.rstrip('/')
        
    async def get_embeddings(self, sequences: List[str], batch_size: int = 8) -> Dict[str, Any]:
        """Fetch embeddings from the embedding service."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            embedding_request = {
                "sequences": sequences,
                "batch_size": batch_size
            }
            
            logger.info(f"Requesting embeddings for {len(sequences)} sequences from {self.base_url}")
            start_time = time.time()
            
            try:
                response = await client.post(
                    f"{self.base_url}/embeddings/compute",
                    json=embedding_request,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                embedding_data = response.json()
                embedding_time = time.time() - start_time
                
                logger.info(f"Retrieved embeddings in {embedding_time:.2f}s")
                
                # Convert from list format to dict format expected by predictor
                embeddings_dict = {}
                for i, embedding in enumerate(embedding_data["embeddings"]):
                    embeddings_dict[str(i)] = embedding
                
                return {
                    "embeddings": embeddings_dict,
                    "timing": {
                        "embedding_computation_ms": embedding_time * 1000,
                        **embedding_data.get("timing", {})
                    }
                }
                
            except httpx.TimeoutException:
                raise HTTPException(status_code=408, detail="Embedding service timeout")
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code, 
                    detail=f"Embedding service error: {e.response.text}"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get embeddings: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of the embedding service."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.base_url}/embeddings/health")
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}