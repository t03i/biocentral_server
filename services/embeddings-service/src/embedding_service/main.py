"""
Production Protein Embedding Service with Triton gRPC and Redis Caching (ProtT5 and ESM-2)
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from enum import Enum

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator

from .triton_client import TritonClientPool
from .cache import EmbeddingCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
class ModelType(str, Enum):
    PROT_T5 = "prot_t5"
    ESM2_T33_650M = "esm2_t33_650M"
    ESM2_T36_3B = "esm2_t36_3B"

MODEL_CONFIG = {
    ModelType.PROT_T5: {
        "triton_model": "prot_t5_pipeline",
        "embedding_dim": 1024,
        "description": "ProtT5 transformer model for protein embeddings"
    },
    ModelType.ESM2_T33_650M: {
        "triton_model": "esm2_t33_pipeline",
        "embedding_dim": 1280,
        "description": "ESM-2 transformer model (650M parameters)"
    },
    ModelType.ESM2_T36_3B: {
        "triton_model": "esm2_t36_pipeline", 
        "embedding_dim": 2560,
        "description": "ESM-2 transformer model (3B parameters)"
    }
}

# Global components
triton_pools: Dict[str, TritonClientPool] = {}
embedding_cache: Optional[EmbeddingCache] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle"""
    global triton_pools, embedding_cache
    
    # Startup
    logger.info("Starting Protein Embedding Service")
    
    try:
        # Initialize Redis cache
        redis_url = os.getenv("REDIS_URL", "redis://redis-jobs:6379")
        embedding_cache = EmbeddingCache(
            redis_url=redis_url,
            ttl=7 * 24 * 3600,  # 1 week
            compression_level=5
        )
        await embedding_cache.connect()
        
        # Initialize Triton client pools for each model
        for model_type, config in MODEL_CONFIG.items():
            triton_pools[model_type] = TritonClientPool(
                triton_url="triton:8001",
                model_name=config["triton_model"],
                pool_size=4
            )
            await triton_pools[model_type].initialize()
            logger.info(f"✅ Initialized {model_type} Triton pool")
        
        logger.info("✅ Service initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Protein Embedding Service")
    
    for pool in triton_pools.values():
        await pool.close_all()
    if embedding_cache:
        await embedding_cache.disconnect()

app = FastAPI(
    title="Protein Embedding Service",
    description="High-performance protein sequence embedding service using ProtT5 and ESM-2 via Triton",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models
class EmbeddingRequest(BaseModel):
    sequences: List[str] = Field(..., min_items=1, max_items=100)
    model: ModelType = Field(default=ModelType.PROT_T5, description="Model to use for embeddings")
    batch_size: Optional[int] = Field(default=8, ge=1, le=8)
    
    @validator('sequences')
    def validate_sequences(cls, v):
        for i, seq in enumerate(v):
            if not seq or not isinstance(seq, str):
                raise ValueError(f"Sequence {i} is empty or not a string")
            if len(seq) < 10:
                raise ValueError(f"Sequence {i} too short (min 10 residues)")
            if len(seq) > 5000:
                raise ValueError(f"Sequence {i} too long (max 5000 residues)")
            # Basic protein sequence validation
            valid_chars = set('ACDEFGHIKLMNPQRSTVWYXU*-')
            if not set(seq.upper()).issubset(valid_chars):
                invalid_chars = set(seq.upper()) - valid_chars
                raise ValueError(f"Sequence {i} contains invalid characters: {invalid_chars}")
        return v

class EmbeddingResponse(BaseModel):
    embeddings: Dict[str, List[float]]
    cache_stats: Dict[str, float]
    timing: Dict[str, float]
    model_info: Dict[str, str]

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, Dict[str, bool]]
    cache_stats: Optional[Dict] = None
    model_info: Optional[Dict] = None

class CacheStatsResponse(BaseModel):
    stats: Dict
    redis_info: Dict[str, bool]

# Endpoints
@app.post("/embeddings/compute", response_model=EmbeddingResponse)
async def compute_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """
    Compute protein embeddings using ProtT5 or ESM-2 models with intelligent caching
    """
    start_time = time.time()
    
    try:
        sequences = [seq.upper().replace('*', '') for seq in request.sequences]  # Clean sequences
        model_config = MODEL_CONFIG[request.model]
        
        # Check cache first (caching is mandatory)
        cache_start = time.time()
        cached_results = await embedding_cache.get_embeddings(sequences, request.model.value)
        missing_sequences = [seq for seq, emb in cached_results.items() if emb is None]
        cache_time = time.time() - cache_start
        
        cache_hits = len(sequences) - len(missing_sequences)
        logger.info(f"Cache ({request.model}): {cache_hits}/{len(sequences)} hits ({cache_time*1000:.1f}ms)")
        
        # Compute missing embeddings via Triton
        triton_time = 0.0
        new_embeddings = {}
        
        if missing_sequences:
            triton_start = time.time()
            
            # Get client from appropriate pool
            triton_pool = triton_pools[request.model]
            client = await triton_pool.get_client()
            
            try:
                # Compute embeddings
                embedding_arrays = await client.compute_embeddings_batch(
                    missing_sequences, 
                    batch_size=request.batch_size
                )
                
                # Convert to dict
                new_embeddings = {
                    seq: emb for seq, emb in zip(missing_sequences, embedding_arrays)
                }
                
                triton_time = time.time() - triton_start
                logger.info(f"Triton ({request.model}): computed {len(missing_sequences)} embeddings ({triton_time*1000:.1f}ms)")
                
            finally:
                # Return client to pool
                await triton_pool.return_client(client)
            
            # Cache new embeddings in background (mandatory)
            if new_embeddings:
                background_tasks.add_task(
                    embedding_cache.set_embeddings,
                    new_embeddings,
                    request.model.value
                )
        
        # Combine cached and new results
        all_embeddings = {}
        for seq in sequences:
            if seq in cached_results and cached_results[seq] is not None:
                all_embeddings[seq] = cached_results[seq]
            elif seq in new_embeddings:
                all_embeddings[seq] = new_embeddings[seq]
            else:
                raise ValueError(f"Failed to compute embedding for sequence: {seq[:20]}...")
        
        # Convert numpy arrays to lists for JSON serialization
        embeddings_json = {
            seq: emb.tolist() for seq, emb in all_embeddings.items()
        }
        
        total_time = time.time() - start_time
        
        # Prepare response
        cache_stats = await embedding_cache.get_cache_stats()
        
        return EmbeddingResponse(
            embeddings=embeddings_json,
            cache_stats={
                "hit_rate": cache_stats.get("hit_rate", 0.0),
                "compression_ratio": cache_stats.get("compression_ratio", 0.0)
            },
            timing={
                "total_ms": total_time * 1000,
                "cache_ms": cache_time * 1000,
                "triton_ms": triton_time * 1000,
                "cache_hits": len(sequences) - len(missing_sequences),
                "triton_calls": len(missing_sequences)
            },
            model_info={
                "model": request.model.value,
                "triton_model": model_config["triton_model"],
                "embedding_dim": str(model_config["embedding_dim"]),
                "description": model_config["description"],
                "precision": "fp32"
            }
        )
        
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding computation failed: {str(e)}")

@app.get("/embeddings/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for all service components"""
    
    components = {}
    
    # Check Triton pools for each model
    for model_type, triton_pool in triton_pools.items():
        try:
            client = await triton_pool.get_client()
            triton_health = await client.health_check()
            await triton_pool.return_client(client)
            components[f"triton_{model_type}"] = triton_health
        except Exception as e:
            logger.warning(f"Triton health check failed for {model_type}: {e}")
            components[f"triton_{model_type}"] = {"connected": False, "server_ready": False, "model_ready": False}
    
    # Check cache
    if embedding_cache:
        try:
            cache_health = await embedding_cache.health_check()
            components["cache"] = cache_health
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
            components["cache"] = {"connected": False, "responsive": False}
    else:
        components["cache"] = {"connected": False, "responsive": False}
    
    # Determine overall status
    all_healthy = all(
        all(status.values()) for status in components.values()
    )
    
    overall_status = "healthy" if all_healthy else "unhealthy"
    
    # Get additional info if healthy
    cache_stats = None
    model_info = None
    
    if overall_status == "healthy":
        try:
            cache_stats = await embedding_cache.get_cache_stats()
            
            # Get model info from first available model
            first_model = list(triton_pools.keys())[0]
            triton_pool = triton_pools[first_model]
            client = await triton_pool.get_client()
            model_info = await client.get_model_metadata()
            await triton_pool.return_client(client)
            
        except Exception as e:
            logger.warning(f"Failed to get additional health info: {e}")
    
    return HealthResponse(
        status=overall_status,
        components=components,
        cache_stats=cache_stats,
        model_info=model_info
    )

@app.get("/embeddings/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get detailed cache statistics"""
    if not embedding_cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    try:
        stats = await embedding_cache.get_cache_stats()
        health = await embedding_cache.health_check()
        
        return CacheStatsResponse(
            stats=stats,
            redis_info=health
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")

@app.post("/embeddings/cache/clear")
async def clear_cache(pattern: Optional[str] = None):
    """Clear cache entries (admin endpoint)"""
    if not embedding_cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    try:
        deleted = await embedding_cache.clear_cache(pattern)
        return {"message": f"Cleared {deleted} cache entries", "pattern": pattern or "all"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@app.get("/embeddings/model/info")
async def get_model_info(model: ModelType = ModelType.PROT_T5):
    """Get Triton model information for specific model"""
    if model not in triton_pools:
        raise HTTPException(status_code=404, detail=f"Model {model} not available")
    
    try:
        triton_pool = triton_pools[model]
        client = await triton_pool.get_client()
        model_info = await client.get_model_metadata()
        await triton_pool.return_client(client)
        
        # Add our model config info
        config = MODEL_CONFIG[model]
        model_info.update({
            "model_type": model.value,
            "embedding_dim": config["embedding_dim"],
            "description": config["description"]
        })
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info for {model}: {e}")
        raise HTTPException(status_code=500, detail=f"Model info failed: {str(e)}")

@app.get("/embeddings/models")
async def list_available_models():
    """List all available embedding models"""
    return {
        "available_models": {
            model_type.value: {
                "triton_model": config["triton_model"],
                "embedding_dim": config["embedding_dim"], 
                "description": config["description"],
                "status": "available" if model_type in triton_pools else "unavailable"
            }
            for model_type, config in MODEL_CONFIG.items()
        }
    }

# Basic root endpoint
@app.get("/")
async def root():
    return {
        "service": "Protein Embedding Service",
        "version": "1.0.0", 
        "status": "running",
        "supported_models": list(MODEL_CONFIG.keys()),
        "endpoints": [
            "/embeddings/compute",
            "/embeddings/health",
            "/embeddings/cache/stats", 
            "/embeddings/model/info",
            "/embeddings/models"
        ]
    }

