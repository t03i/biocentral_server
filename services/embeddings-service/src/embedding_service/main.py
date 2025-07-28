"""
Production Protein Embedding Service with Triton gRPC and Redis Caching (ProtT5 and ESM-2)
"""
import logging
import time
import io
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import StreamingResponse
from pydantic import Field

from .config import config
from .models import (
    ModelType, MODEL_CONFIG,
    EmbeddingResponse, HealthResponse, CacheStatsResponse,
    ModelInfoResponse, ModelsResponse, ConfigResponse, RootResponse
)
from .utils import validate_protein_sequences
from .triton_client import TritonClientPool, TritonEmbeddingClient
from .cache import EmbeddingCache

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global components
triton_pools: Dict[str, TritonClientPool] = {}
embedding_cache: Optional[EmbeddingCache] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle"""
    global triton_pools, embedding_cache
    
    # Startup
    logger.info("Starting Protein Embedding Service")
    logger.info(f"Configuration: min_seq={config.min_sequence_length}, max_seq={config.max_sequence_length}, max_batch={config.max_sequences_per_request}")
    
    try:
        # Initialize Redis cache
        embedding_cache = EmbeddingCache(
            redis_url=config.redis_url,
            ttl=config.redis_ttl_days * 24 * 3600,
            compression_level=config.redis_compression_level
        )
        await embedding_cache.connect()
        
        # Initialize Triton client pools for each model
        for model_type, model_config in MODEL_CONFIG.items():
            triton_pools[model_type] = TritonClientPool(
                triton_url=config.triton_url,
                model_name=model_config["triton_model"],
                pool_size=config.triton_pool_size,
                timeout=config.triton_timeout
            )
            await triton_pools[model_type].initialize()
            
            # Detect actual model precision from Triton
            try:
                raw_client = await triton_pools[model_type].get_client()
                embedding_client = TritonEmbeddingClient(
                    client=raw_client,
                    model_name=model_config["triton_model"],
                    timeout=config.triton_timeout
                )
                metadata = await embedding_client.get_model_metadata()
                if 'outputs' in metadata and len(metadata['outputs']) > 0:
                    model_config["precision"] = metadata['outputs'][0].get('datatype', 'fp16').lower()
                await triton_pools[model_type].return_client(raw_client)
            except Exception as e:
                logger.warning(f"Could not detect precision for {model_type}: {e}")
            
            logger.info(f"✅ Initialized {model_type} Triton pool (precision: {model_config['precision']})")
        
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

# Redis storage for downloads
async def _store_for_download(request_id: str, embeddings: List, sequences: List[str], model_config: Dict):
    """Store embeddings as binary data in Redis for download with 10-minute TTL"""
    if not embedding_cache:
        logger.warning("Cache not available, cannot store for download")
        return
    
    try:
        # Store each embedding as a separate array (sequences typically have different lengths)
        # Convert each embedding to binary and store as a list
        binary_embeddings = []
        for emb in embeddings:
            emb_array = np.array(emb, dtype=np.float16)
            binary_embeddings.append({
                'shape': emb_array.shape,
                'dtype': str(emb_array.dtype),
                'data': emb_array.tobytes()
            })
        
        binary_data = json.dumps(binary_embeddings)
        shape_info = {
            'storage_type': 'list_of_arrays',
            'count': len(binary_embeddings)
        }
        
        # Store binary embeddings and shape info
        binary_key = f"download:{request_id}"
        shape_key = f"download:{request_id}:shape"
        
        await embedding_cache._redis.setex(
            binary_key,
            config.download_ttl_minutes * 60,  # 10 minutes in seconds
            binary_data  # Store as JSON string
        )
        
        await embedding_cache._redis.setex(
            shape_key,
            config.download_ttl_minutes * 60,  # 10 minutes in seconds
            json.dumps(shape_info)  # Store shape info as JSON
        )
        
        logger.debug(f"Stored {len(binary_embeddings)} embeddings for {request_id} with {config.download_ttl_minutes}min TTL")
        
    except Exception as e:
        logger.error(f"Failed to store embeddings: {e}")

# Endpoints
@app.post("/embeddings/compute/{model}", response_model=EmbeddingResponse)
async def compute_embeddings(
    model: str, 
    sequences: List[str], 
    background_tasks: BackgroundTasks,
    pooled: bool = False
):
    """
    Compute protein embeddings using ProtT5 or ESM-2 models with intelligent caching
    
    Args:
        model: Model name (prot_t5, esm2_t33_650M, esm2_t36_3B)
        sequences: List of protein sequences
        pooled: If True, return mean-pooled embeddings (1 vector per sequence). If False, return full sequence embeddings
    
    Example:
        POST /embeddings/compute/prot_t5
        {
            "sequences": [
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "MKTAYIAELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
            ],
            "pooled": false
        }
    """
    start_time = time.time()
    
    try:
        # Validate model
        if model not in [m.value for m in ModelType]:
            available_models = [m.value for m in ModelType]
            raise ValueError(f"Invalid model '{model}'. Available models: {available_models}")
        
        model_type = ModelType(model)
        model_config = MODEL_CONFIG[model_type]
        
        # Validate and clean sequences
        sequences = validate_protein_sequences(sequences)
        
        # Always use unpooled cache key (cache stores full embeddings)
        cache_model_key = f"{model}_full"
        
        # Check cache first (caching is mandatory)
        cache_start = time.time()
        cached_results = await embedding_cache.get_embeddings(sequences, cache_model_key)
        missing_sequences = [seq for seq, emb in cached_results.items() if emb is None]
        cache_time = time.time() - cache_start
        
        cache_hits = len(sequences) - len(missing_sequences)
        cache_hit_rate = cache_hits / len(sequences) if sequences else 0.0
        
        logger.info(f"Cache ({cache_model_key}): {cache_hits}/{len(sequences)} hits ({cache_hit_rate:.1%}, {cache_time*1000:.1f}ms)")
        
        # Compute missing embeddings via Triton (always unpooled for caching)
        triton_time = 0.0
        new_embeddings = {}
        
        if missing_sequences:
            triton_start = time.time()
            
            # Get client from appropriate pool
            triton_pool = triton_pools[model_type]
            raw_client = await triton_pool.get_client()
            
            try:
                # Create embedding client wrapper
                embedding_client = TritonEmbeddingClient(
                    client=raw_client,
                    model_name=model_config["triton_model"],
                    timeout=config.triton_timeout
                )
                
                # Always compute unpooled embeddings for caching
                embedding_arrays = await embedding_client.compute_embeddings_batch(missing_sequences, pooled=False)
                
                # Convert to dict
                new_embeddings = {
                    seq: emb for seq, emb in zip(missing_sequences, embedding_arrays)
                }
                
                triton_time = time.time() - triton_start
                logger.info(f"Triton ({cache_model_key}): computed {len(missing_sequences)} unpooled embeddings ({triton_time*1000:.1f}ms)")
                
            finally:
                # Return client to pool
                await triton_pool.return_client(raw_client)
            
            # Cache new embeddings in background (mandatory) - always unpooled
            if new_embeddings:
                background_tasks.add_task(
                    embedding_cache.set_embeddings,
                    new_embeddings,
                    cache_model_key
                )
        
        # Combine cached and new results in original order, applying pooling if requested
        all_embeddings = []
        for seq in sequences:
            if seq in cached_results and cached_results[seq] is not None:
                # Cached embeddings are always unpooled numpy arrays
                embedding = cached_results[seq]
                if embedding.ndim == 1:  # Old pooled format (shouldn't happen with new logic)
                    embedding = embedding.reshape(1, -1)  # Convert to [1, dim]
                elif pooled and embedding.ndim == 2:
                    # Pool the cached unpooled embedding if requested
                    embedding = np.mean(embedding, axis=0).reshape(1, -1)  # Convert to [1, dim]
                all_embeddings.append(embedding)
            elif seq in new_embeddings:
                # New embeddings are always unpooled from Triton
                embedding = new_embeddings[seq]
                if pooled and embedding.ndim == 2:
                    # Pool the new unpooled embedding if requested
                    embedding = np.mean(embedding, axis=0).reshape(1, -1)  # Convert to [1, dim]
                all_embeddings.append(embedding)
            else:
                raise ValueError(f"Failed to compute embedding for sequence: {seq[:20]}...")
        
        total_time = time.time() - start_time
        
        # Generate download link (binary downloads are now mandatory)
        request_id = f"{int(time.time())}_{hash(str(sequences))}"  # Simple request ID
        download_link = f"/embeddings/download/{model}/numpy/{request_id}"
        
        # Store the requested variant (pooled or unpooled) in Redis for download
        # This ensures the download matches exactly what was requested
        background_tasks.add_task(
            _store_for_download,
            request_id,
            all_embeddings,  # This is already the requested variant (pooled or unpooled)
            sequences,
            model_config
        )
        
        return EmbeddingResponse(
            sequences=sequences,
            cache_stats={
                "hit_rate": cache_hit_rate,
                "total_hits": cache_hits,
                "total_requests": len(sequences),
                "cache_misses": len(missing_sequences)
            },
            timing={
                "total_ms": total_time * 1000,
                "cache_ms": cache_time * 1000,
                "triton_ms": triton_time * 1000,
                "cache_hits": cache_hits,
                "triton_calls": len(missing_sequences)
            },
            model_info={
                "model": model,
                "triton_model": model_config["triton_model"],
                "embedding_dim": model_config["embedding_dim"],
                "sequence_length_range": f"{config.min_sequence_length}-{config.max_sequence_length}",
                "description": model_config["description"],
                "precision": model_config["precision"],
                "pooled": pooled,
                "shape_description": f"[sequences, 1, {model_config['embedding_dim']}] (pooled)" if pooled else f"[sequences, sequence_length, {model_config['embedding_dim']}] (full)"
            },
            download_link=download_link
        )
        
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding computation failed: {str(e)}")

# Download endpoint for compressed format only
@app.get("/embeddings/download/{model}/numpy/{request_id}")
async def download_numpy_compressed(model: str, request_id: str):
    """
    Download embeddings as compressed numpy array (.npz)
    
    Args:
        model: Model name (prot_t5, esm2_t33_650M, esm2_t36_3B)
        request_id: Request ID from the compute endpoint response
    
    Returns:
        Compressed numpy file (.npz) containing multiple arrays:
        - embedding_0, embedding_1, etc.: Individual embeddings for each sequence
        - sequence_count: Number of sequences
        - storage_type: Always "list_of_arrays"
        
        Each embedding array has shape [sequence_length, embedding_dim] for full embeddings
        or [embedding_dim] for pooled embeddings.
    
    Example:
        GET /embeddings/download/prot_t5/numpy/1703123456_12345
        # Returns: embeddings_prot_t5_1703123456_12345.npz
    """
    if not embedding_cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    try:
        # Retrieve stored binary embeddings and shape info from Redis
        binary_key = f"download:{request_id}"
        shape_key = f"download:{request_id}:shape"
        
        binary_data = await embedding_cache._redis.get(binary_key)
        shape_data = await embedding_cache._redis.get(shape_key)
        
        if not binary_data or not shape_data:
            raise HTTPException(status_code=404, detail="Download not found or expired")
        
        # Parse shape info and reconstruct embeddings
        shape_info = json.loads(shape_data)
        binary_embeddings_list = json.loads(binary_data)
        
        embeddings_array = []
        for item in binary_embeddings_list:
            shape = tuple(item['shape'])
            dtype = item['dtype']
            data = item['data']
            embeddings_array.append(np.frombuffer(data, dtype=dtype).reshape(shape))
        
        # Create compressed numpy file in memory
        # Store each embedding with its sequence index
        buffer = io.BytesIO()
        npz_data = {}
        for i, emb in enumerate(embeddings_array):
            npz_data[f'embedding_{i}'] = emb
        npz_data['sequence_count'] = len(embeddings_array)
        npz_data['storage_type'] = 'list_of_arrays'
        np.savez_compressed(buffer, **npz_data)
        buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=embeddings_{model}_{request_id}.npz"}
        )
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/embeddings/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check for all service components
    
    Returns:
        Health status of all components including:
        - Triton server connections for each model
        - Redis cache connectivity
        - Cache statistics (if healthy)
        - Model metadata (if healthy)
        - Service configuration summary
    
    Example Response:
        {
            "status": "healthy",
            "components": {
                "triton_prot_t5": {"connected": true, "server_ready": true, "model_ready": true},
                "triton_esm2_t33_650M": {"connected": true, "server_ready": true, "model_ready": true},
                "triton_esm2_t36_3B": {"connected": true, "server_ready": true, "model_ready": true},
                "cache": {"connected": true, "responsive": true}
            },
            "cache_stats": {
                "total_keys": 1250,
                "memory_usage": "45.2MB",
                "hit_rate": 0.78
            },
            "config_summary": {
                "min_sequence_length": 10,
                "max_sequence_length": 5000,
                "max_sequences_per_request": 100
            }
        }
    """
    
    components = {}
    
    # Check Triton pools for each model
    for model_type, triton_pool in triton_pools.items():
        try:
            raw_client = await triton_pool.get_client()
            embedding_client = TritonEmbeddingClient(
                client=raw_client,
                model_name=MODEL_CONFIG[model_type]["triton_model"],
                timeout=config.triton_timeout
            )
            triton_health = await embedding_client.health_check()
            await triton_pool.return_client(raw_client)
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
            raw_client = await triton_pool.get_client()
            embedding_client = TritonEmbeddingClient(
                client=raw_client,
                model_name=MODEL_CONFIG[first_model]["triton_model"],
                timeout=config.triton_timeout
            )
            model_info = await embedding_client.get_model_metadata()
            await triton_pool.return_client(raw_client)
            
        except Exception as e:
            logger.warning(f"Failed to get additional health info: {e}")
    
    # Add config summary
    config_summary = {
        "min_sequence_length": config.min_sequence_length,
        "max_sequence_length": config.max_sequence_length,
        "max_sequences_per_request": config.max_sequences_per_request,
        "redis_ttl_days": config.redis_ttl_days,
        "triton_pool_size": config.triton_pool_size,
        "binary_downloads_enabled": config.enable_binary_downloads,
        "download_ttl_minutes": config.download_ttl_minutes,
        "log_level": config.log_level
    }
    
    return HealthResponse(
        status=overall_status,
        components=components,
        cache_stats=cache_stats,
        model_info=model_info,
        config_summary=config_summary
    )

@app.get("/embeddings/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get detailed cache statistics
    
    Returns:
        Detailed Redis cache statistics including:
        - Memory usage and key counts
        - Hit/miss rates
        - Connection information
    
    Example Response:
        {
            "stats": {
                "total_keys": 1250,
                "memory_usage": "45.2MB",
                "hit_rate": 0.78,
                "total_requests": 5000,
                "cache_hits": 3900,
                "cache_misses": 1100
            },
            "redis_info": {
                "connected": true,
                "responsive": true,
                "version": "7.0.0",
                "used_memory": "45.2MB"
            }
        }
    """
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
    """
    Clear cache entries (admin endpoint)
    
    Args:
        pattern: Optional Redis pattern to match keys for deletion (e.g., "prot_t5_*")
                 If None, clears all cache entries
    
    Returns:
        Confirmation message with number of deleted entries
    
    Examples:
        POST /embeddings/cache/clear
        # Clears all cache entries
        
        POST /embeddings/cache/clear?pattern=prot_t5_*
        # Clears only ProtT5 cache entries
        
        POST /embeddings/cache/clear?pattern=esm2_*
        # Clears only ESM-2 cache entries
    
    Example Response:
        {
            "message": "Cleared 1250 cache entries",
            "pattern": "prot_t5_*"
        }
    """
    if not embedding_cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    try:
        deleted = await embedding_cache.clear_cache(pattern)
        return {"message": f"Cleared {deleted} cache entries", "pattern": pattern or "all"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@app.get("/embeddings/model/info", response_model=ModelInfoResponse)
async def get_model_info(model: ModelType = ModelType.PROT_T5):
    """
    Get Triton model information for specific model
    
    Args:
        model: Model type to get information for (default: prot_t5)
    
    Returns:
        Detailed model information including:
        - Model configuration and metadata
        - Input/output specifications
        - Performance characteristics
    
    Examples:
        GET /embeddings/model/info?model=prot_t5
        GET /embeddings/model/info?model=esm2_t33_650M
        GET /embeddings/model/info?model=esm2_t36_3B
    
    Example Response:
        {
            "model_type": "prot_t5",
            "triton_model": "prot_t5_pipeline",
            "embedding_dim": 1024,
            "description": "ProtT5 transformer model for protein embeddings",
            "precision": "fp16",
            "inputs": [
                {
                    "name": "input_ids",
                    "datatype": "INT32",
                    "shape": [-1, -1]
                }
            ],
            "outputs": [
                {
                    "name": "hidden_states",
                    "datatype": "FP16",
                    "shape": [-1, -1, 1024]
                }
            ]
        }
    """
    if model not in triton_pools:
        raise HTTPException(status_code=404, detail=f"Model {model} not available")
    
    try:
        triton_pool = triton_pools[model]
        raw_client = await triton_pool.get_client()
        embedding_client = TritonEmbeddingClient(
            client=raw_client,
            model_name=MODEL_CONFIG[model]["triton_model"],
            timeout=config.triton_timeout
        )
        model_info = await embedding_client.get_model_metadata()
        await triton_pool.return_client(raw_client)
        
        # Add our model config info
        model_config = MODEL_CONFIG[model]
        model_info.update({
            "model_type": model.value,
            "embedding_dim": model_config["embedding_dim"],
            "description": model_config["description"],
            "precision": model_config["precision"]
        })
        
        return ModelInfoResponse(**model_info)
        
    except Exception as e:
        logger.error(f"Failed to get model info for {model}: {e}")
        raise HTTPException(status_code=500, detail=f"Model info failed: {str(e)}")

@app.get("/embeddings/models", response_model=ModelsResponse)
async def list_available_models():
    """
    List all available embedding models
    
    Returns:
        Complete list of available models with their configurations
        and current service settings
    
    Example Response:
        {
            "available_models": {
                "prot_t5": {
                    "triton_model": "prot_t5_pipeline",
                    "embedding_dim": 1024,
                    "description": "ProtT5 transformer model for protein embeddings",
                    "precision": "fp16",
                    "status": "available"
                },
                "esm2_t33_650M": {
                    "triton_model": "esm2_t33_pipeline",
                    "embedding_dim": 1280,
                    "description": "ESM-2 transformer model (650M parameters)",
                    "precision": "fp16",
                    "status": "available"
                },
                "esm2_t36_3B": {
                    "triton_model": "esm2_t36_pipeline",
                    "embedding_dim": 2560,
                    "description": "ESM-2 transformer model (3B parameters)",
                    "precision": "fp16",
                    "status": "available"
                }
            },
            "config": {
                "min_sequence_length": 10,
                "max_sequence_length": 5000,
                "max_sequences_per_request": 100,
                "binary_downloads_enabled": true,
                "download_ttl_minutes": 10
            }
        }
    """
    return ModelsResponse(
        available_models={
            model_type.value: {
                "triton_model": model_config["triton_model"],
                "embedding_dim": model_config["embedding_dim"], 
                "description": model_config["description"],
                "precision": model_config["precision"],
                "status": "available" if model_type in triton_pools else "unavailable"
            }
            for model_type, model_config in MODEL_CONFIG.items()
        },
        config={
            "min_sequence_length": config.min_sequence_length,
            "max_sequence_length": config.max_sequence_length,
            "max_sequences_per_request": config.max_sequences_per_request,
            "binary_downloads_enabled": True,
            "download_ttl_minutes": config.download_ttl_minutes
        }
    )

@app.get("/embeddings/config", response_model=ConfigResponse)
async def get_service_config():
    """
    Get current service configuration
    
    Returns:
        Complete service configuration including:
        - Sequence length limits
        - Triton server settings
        - Redis cache configuration
        - Feature flags and settings
    
    Example Response:
        {
            "sequence_limits": {
                "min_length": 10,
                "max_length": 5000,
                "max_sequences_per_request": 100
            },
            "triton": {
                "url": "triton:8000",
                "pool_size": 5,
                "timeout_seconds": 30
            },
            "redis": {
                "url": "redis://***@redis:6379",
                "ttl_days": 30,
                "compression_level": 6
            },
            "features": {
                "binary_downloads": true,
                "full_embeddings": true,
                "log_level": "INFO",
                "download_ttl_minutes": 10
            }
        }
    """
    return ConfigResponse(
        sequence_limits={
            "min_length": config.min_sequence_length,
            "max_length": config.max_sequence_length,
            "max_sequences_per_request": config.max_sequences_per_request
        },
        triton={
            "url": config.triton_url,
            "pool_size": config.triton_pool_size,
            "timeout_seconds": config.triton_timeout
        },
        redis={
            "url": config.redis_url.replace(r'://.*@', '://***@'),  # Hide password
            "ttl_days": config.redis_ttl_days,
            "compression_level": config.redis_compression_level
        },
        features={
            "binary_downloads": True,
            "full_embeddings": config.return_full_embeddings,
            "log_level": config.log_level,
            "download_ttl_minutes": config.download_ttl_minutes
        }
    )

# Basic root endpoint
@app.get("/", response_model=RootResponse)
async def root():
    """
    Root endpoint with service information
    
    Returns:
        Basic service information including:
        - Service name and version
        - Current status
        - Supported models
        - Available endpoints
        - Feature flags
    
    Example Response:
        {
            "service": "Protein Embedding Service",
            "version": "1.0.0",
            "status": "running",
            "supported_models": ["prot_t5", "esm2_t33_650M", "esm2_t36_3B"],
            "endpoints": [
                "/embeddings/compute/{model}",
                "/embeddings/download/{model}/numpy/{request_id}",
                "/embeddings/health",
                "/embeddings/cache/stats",
                "/embeddings/model/info",
                "/embeddings/models",
                "/embeddings/config"
            ],
            "features": {
                "configurable_via_env": true,
                "binary_downloads": true,
                "full_sequence_embeddings": true,
                "intelligent_caching": true,
                "redis_download_storage": true
            }
        }
    """
    return RootResponse(
        service="Protein Embedding Service",
        version="1.0.0", 
        status="running",
        supported_models=list(MODEL_CONFIG.keys()),
        endpoints=[
            "/embeddings/compute/{model}",
            "/embeddings/download/{model}/numpy/{request_id}",
            "/embeddings/health",
            "/embeddings/cache/stats", 
            "/embeddings/model/info",
            "/embeddings/models",
            "/embeddings/config"
        ],
        features={
            "configurable_via_env": True,
            "binary_downloads": True,
            "full_sequence_embeddings": True,
            "intelligent_caching": True,
            "redis_download_storage": True
        }
    )

