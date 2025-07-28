"""
Pydantic models and model configurations for the Protein Embedding Service
"""
from typing import List, Dict, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


# Model configurations
class ModelType(str, Enum):
    PROT_T5 = "prot_t5"
    ESM2_T33_650M = "esm2_t33_650M"
    ESM2_T36_3B = "esm2_t36_3B"


MODEL_CONFIG = {
    ModelType.PROT_T5: {
        "triton_model": "prot_t5_pipeline",
        "embedding_dim": 1024,
        "description": "ProtT5 transformer model for protein embeddings",
        "precision": "fp16"  # Will be detected from Triton
    },
    ModelType.ESM2_T33_650M: {
        "triton_model": "esm2_t33_pipeline",
        "embedding_dim": 1280,
        "description": "ESM-2 transformer model (650M parameters)",
        "precision": "fp16"  # Will be detected from Triton
    },
    ModelType.ESM2_T36_3B: {
        "triton_model": "esm2_t36_pipeline", 
        "embedding_dim": 2560,
        "description": "ESM-2 transformer model (3B parameters)",
        "precision": "fp16"  # Will be detected from Triton
    }
}


# Pydantic models for OpenAPI schemas
class EmbeddingResponse(BaseModel):
    """Response model for embedding computation"""
    sequences: List[str] = Field(
        description="The processed sequences for reference",
        example=[
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "MKTAYIAELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        ]
    )
    cache_stats: Dict[str, float] = Field(
        description="Cache statistics for this request",
        example={
            "hit_rate": 0.8,
            "total_hits": 4,
            "total_requests": 5,
            "cache_misses": 1
        }
    )
    timing: Dict[str, float] = Field(
        description="Timing information for this request",
        example={
            "total_ms": 150.5,
            "cache_ms": 2.1,
            "triton_ms": 148.4,
            "cache_hits": 4,
            "triton_calls": 1
        }
    )
    model_info: Dict[str, Union[str, int]] = Field(
        description="Model information and metadata",
        example={
            "model": "prot_t5",
            "triton_model": "prot_t5_pipeline",
            "embedding_dim": 1024,
            "sequence_length_range": "10-5000",
            "description": "ProtT5 transformer model for protein embeddings",
            "precision": "fp16",
            "pooled": False,
            "shape_description": "[sequences, sequence_length, 1024] (full)"
        }
    )
    download_link: str = Field(
        description="Link to compressed numpy download",
        example="/embeddings/download/prot_t5/numpy/1703123456_12345"
    )


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(
        description="Overall service status",
        example="healthy"
    )
    components: Dict[str, Dict[str, bool]] = Field(
        description="Health status of individual components",
        example={
            "triton_prot_t5": {"connected": True, "server_ready": True, "model_ready": True},
            "triton_esm2_t33_650M": {"connected": True, "server_ready": True, "model_ready": True},
            "triton_esm2_t36_3B": {"connected": True, "server_ready": True, "model_ready": True},
            "cache": {"connected": True, "responsive": True}
        }
    )
    cache_stats: Optional[Dict] = Field(
        None, 
        description="Global cache statistics",
        example={
            "total_keys": 1250,
            "memory_usage": "45.2MB",
            "hit_rate": 0.78
        }
    )
    model_info: Optional[Dict] = Field(
        None, 
        description="Model information from first available model",
        example={
            "name": "prot_t5_pipeline",
            "version": "1",
            "platform": "tensorrt_plan"
        }
    )
    config_summary: Optional[Dict] = Field(
        None, 
        description="Service configuration summary",
        example={
            "min_sequence_length": 10,
            "max_sequence_length": 5000,
            "max_sequences_per_request": 100,
            "redis_ttl_days": 30,
            "triton_pool_size": 5,
            "binary_downloads_enabled": True,
            "download_ttl_minutes": 10,
            "log_level": "INFO"
        }
    )


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics"""
    stats: Dict = Field(
        description="Detailed cache statistics",
        example={
            "total_keys": 1250,
            "memory_usage": "45.2MB",
            "hit_rate": 0.78,
            "total_requests": 5000,
            "cache_hits": 3900,
            "cache_misses": 1100,
            "keyspace_hits": 3900,
            "keyspace_misses": 1100
        }
    )
    redis_info: Dict[str, bool] = Field(
        description="Redis connection information",
        example={
            "connected": True,
            "responsive": True,
            "version": "7.0.0",
            "used_memory": "45.2MB",
            "connected_clients": 5
        }
    )


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str = Field(
        description="Model type identifier",
        example="prot_t5"
    )
    triton_model: str = Field(
        description="Triton model name",
        example="prot_t5_pipeline"
    )
    embedding_dim: int = Field(
        description="Embedding dimension",
        example=1024
    )
    description: str = Field(
        description="Model description",
        example="ProtT5 transformer model for protein embeddings"
    )
    precision: str = Field(
        description="Model precision (fp16/fp32)",
        example="fp16"
    )
    inputs: List[Dict] = Field(
        description="Model input specifications",
        example=[
            {
                "name": "input_ids",
                "datatype": "INT32",
                "shape": [-1, -1]
            }
        ]
    )
    outputs: List[Dict] = Field(
        description="Model output specifications",
        example=[
            {
                "name": "hidden_states",
                "datatype": "FP16",
                "shape": [-1, -1, 1024]
            }
        ]
    )


class ModelsResponse(BaseModel):
    """Response model for available models"""
    available_models: Dict[str, Dict[str, Union[str, int]]] = Field(
        description="Available models and their configurations",
        example={
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
        }
    )
    config: Dict[str, Union[str, int, bool]] = Field(
        description="Service configuration",
        example={
            "min_sequence_length": 10,
            "max_sequence_length": 5000,
            "max_sequences_per_request": 100,
            "binary_downloads_enabled": True,
            "download_ttl_minutes": 10
        }
    )


class ConfigResponse(BaseModel):
    """Response model for service configuration"""
    sequence_limits: Dict[str, int] = Field(
        description="Sequence length limits",
        example={
            "min_length": 10,
            "max_length": 5000,
            "max_sequences_per_request": 100
        }
    )
    triton: Dict[str, Union[str, int]] = Field(
        description="Triton configuration",
        example={
            "url": "triton:8000",
            "pool_size": 5,
            "timeout_seconds": 30
        }
    )
    redis: Dict[str, Union[str, int]] = Field(
        description="Redis configuration",
        example={
            "url": "redis://***@redis:6379",
            "ttl_days": 30,
            "compression_level": 6
        }
    )
    features: Dict[str, Union[str, bool, int]] = Field(
        description="Feature flags",
        example={
            "binary_downloads": True,
            "full_embeddings": True,
            "log_level": "INFO",
            "download_ttl_minutes": 10
        }
    )


class RootResponse(BaseModel):
    """Response model for root endpoint"""
    service: str = Field(
        description="Service name",
        example="Protein Embedding Service"
    )
    version: str = Field(
        description="Service version",
        example="1.0.0"
    )
    status: str = Field(
        description="Service status",
        example="running"
    )
    supported_models: List[str] = Field(
        description="List of supported models",
        example=["prot_t5", "esm2_t33_650M", "esm2_t36_3B"]
    )
    endpoints: List[str] = Field(
        description="Available API endpoints",
        example=[
            "/embeddings/compute/{model}",
            "/embeddings/download/{model}/numpy/{request_id}",
            "/embeddings/health",
            "/embeddings/cache/stats",
            "/embeddings/model/info",
            "/embeddings/models",
            "/embeddings/config"
        ]
    )
    features: Dict[str, bool] = Field(
        description="Available features",
        example={
            "configurable_via_env": True,
            "binary_downloads": True,
            "full_sequence_embeddings": True,
            "intelligent_caching": True,
            "redis_download_storage": True
        }
    ) 