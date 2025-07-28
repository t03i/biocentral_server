import asyncio
import json
import numpy as np
import hashlib
from typing import Dict, List
import redis.asyncio as redis
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import shared utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "shared"))

from queue_manager import QueueManager, JobStatus

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Redis cache for embeddings with compression"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.ttl = 7 * 24 * 3600  # 1 week sliding window
    
    def _cache_key(self, seq_hash: str, embedder_name: str) -> str:
        return f"emb:{embedder_name}:{seq_hash}"
    
    async def get_embeddings_batch(
        self, 
        seq_hashes: List[str], 
        embedder_name: str
    ) -> Dict[str, np.ndarray]:
        """Get embeddings from cache with TTL refresh"""
        
        cache_keys = [self._cache_key(h, embedder_name) for h in seq_hashes]
        values = await self.redis.mget(cache_keys)
        
        results = {}
        refresh_keys = []
        
        for seq_hash, cache_key, value in zip(seq_hashes, cache_keys, values):
            if value:
                results[seq_hash] = np.frombuffer(value, dtype=np.float32)
                refresh_keys.append(cache_key)
        
        # Refresh TTL for hits
        if refresh_keys:
            pipe = self.redis.pipeline()
            for key in refresh_keys:
                pipe.expire(key, self.ttl)
            await pipe.execute()
        
        return results
    
    async def set_embeddings_batch(
        self, 
        embeddings: Dict[str, np.ndarray], 
        embedder_name: str
    ):
        """Store embeddings in cache"""
        
        pipe = self.redis.pipeline()
        for seq_hash, embedding in embeddings.items():
            cache_key = self._cache_key(seq_hash, embedder_name)
            pipe.setex(cache_key, self.ttl, embedding.tobytes())
        
        await pipe.execute()

class EmbeddingService:
    """Embedding service with Redis caching"""
    
    def __init__(self):
        self.queue_manager = QueueManager(redis_url="redis://redis-jobs:6379")
        self.cache = EmbeddingCache(redis_url="redis://redis-jobs:6379")
    
    async def run_worker(self):
        """Main worker loop processing embedding jobs"""
        
        logger.info("Starting embedding worker")
        
        while True:
            try:
                # Wait for jobs
                job_data = await self.queue_manager.redis.brpop(
                    self.queue_manager.EMBEDDING_QUEUE, 
                    timeout=5
                )
                
                if not job_data:
                    continue
                
                _, job_json = job_data
                job_info = json.loads(job_json)
                job_id = job_info["job_id"]
                
                logger.info(f"Processing embedding job {job_id}")
                
                # Update status
                job = await self.queue_manager.get_job(job_id)
                if not job:
                    continue
                
                job.status = JobStatus.EMBEDDING
                await self.queue_manager.update_job(job)
                
                # Process embeddings with caching
                embeddings = await self._compute_embeddings_cached(
                    job_info["sequences"],
                    job_info["embedder_name"]
                )
                
                # Queue for next stage
                await self.queue_manager.queue_for_predictions(job_id, embeddings)
                
                logger.info(f"Completed embedding job {job_id}")
                
            except Exception as e:
                logger.error(f"Embedding worker error: {e}")
                if 'job_id' in locals():
                    job = await self.queue_manager.get_job(job_id)
                    if job:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        await self.queue_manager.update_job(job)
    
    async def _compute_embeddings_cached(
        self, 
        sequences: List[str], 
        embedder_name: str
    ) -> Dict[str, List[float]]:
        """Compute embeddings with caching"""
        
        # Hash sequences for cache lookup
        seq_hashes = [
            hashlib.sha256(seq.encode()).hexdigest()[:16] 
            for seq in sequences
        ]
        
        # Check cache
        cached = await self.cache.get_embeddings_batch(seq_hashes, embedder_name)
        
        # Find missing sequences
        missing_sequences = []
        missing_hashes = []
        
        for seq, seq_hash in zip(sequences, seq_hashes):
            if seq_hash not in cached:
                missing_sequences.append(seq)
                missing_hashes.append(seq_hash)
        
        logger.info(f"Cache: {len(cached)} hits, {len(missing_sequences)} misses")
        
        # Compute missing embeddings (placeholder - integrate with existing biotrainer logic)
        if missing_sequences:
            new_embeddings = await self._compute_embeddings_biotrainer(
                missing_sequences, embedder_name
            )
            
            # Cache new embeddings
            cache_data = {
                seq_hash: emb 
                for seq_hash, emb in zip(missing_hashes, new_embeddings)
            }
            await self.cache.set_embeddings_batch(cache_data, embedder_name)
            
            # Add to results
            cached.update(cache_data)
        
        # Return as lists for JSON serialization
        return {
            f"seq_{i}": cached[seq_hash].tolist()
            for i, seq_hash in enumerate(seq_hashes)
        }
    
    async def _compute_embeddings_biotrainer(
        self, 
        sequences: List[str], 
        embedder_name: str
    ) -> List[np.ndarray]:
        """Compute embeddings using biotrainer (placeholder for integration)"""
        
        # TODO: Integrate with existing biotrainer logic from biocentral_server/embeddings/
        # For now, return dummy embeddings
        embedding_dim = 1280 if "esm2" in embedder_name else 1024
        return [np.random.random(embedding_dim).astype(np.float32) for _ in sequences]

# Global service instance
embedding_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle"""
    global embedding_service
    
    # Startup
    embedding_service = EmbeddingService()
    
    # Start worker task
    worker_task = asyncio.create_task(embedding_service.run_worker())
    
    yield
    
    # Shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Biocentral Embedding Service", 
    version="1.0",
    lifespan=lifespan
)

class EmbeddingRequest(BaseModel):
    sequences: List[str]
    embedder_name: str = "esm2_t33_650M_full"

@app.post("/embeddings/compute")
async def compute_embeddings(request: EmbeddingRequest):
    """Compute embeddings for sequences"""
    try:
        # Submit job to queue
        job_id = await embedding_service.queue_manager.submit_job(
            sequences=request.sequences,
            embedder_name=request.embedder_name
        )
        
        return {"job_id": job_id, "status": "submitted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embeddings/job/{job_id}")
async def get_embedding_job(job_id: str):
    """Get embedding job status and results"""
    try:
        job = await embedding_service.queue_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "embeddings": job.embeddings,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "cache_stats": job.cache_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await embedding_service.queue_manager.redis.ping()
        return {"status": "healthy", "service": "embedding"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)