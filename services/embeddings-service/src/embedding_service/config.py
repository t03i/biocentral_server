"""
Configuration management for the Protein Embedding Service
"""
import os


class ServiceConfig:
    """Service configuration from environment variables"""
    
    def __init__(self):
        # Triton settings
        self.triton_url: str = os.getenv("TRITON_URL", "triton:8001")
        self.triton_pool_size: int = int(os.getenv("TRITON_POOL_SIZE", "4"))
        self.triton_timeout: int = int(os.getenv("TRITON_TIMEOUT_SECONDS", "30"))
        
        # Redis settings
        self.redis_url: str = os.getenv("REDIS_URL", "redis://redis-jobs:6379")
        self.redis_ttl_days: int = int(os.getenv("REDIS_TTL_DAYS", "7"))
        self.redis_compression_level: int = int(os.getenv("REDIS_COMPRESSION_LEVEL", "5"))
        
        # Sequence validation settings
        self.min_sequence_length: int = int(os.getenv("MIN_SEQUENCE_LENGTH", "10"))
        self.max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "5000"))
        self.max_sequences_per_request: int = int(os.getenv("MAX_SEQUENCES_PER_REQUEST", "100"))
        
        # Service settings
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.return_full_embeddings: bool = os.getenv("RETURN_FULL_EMBEDDINGS", "true").lower() == "true"
        self.enable_binary_downloads: bool = os.getenv("ENABLE_BINARY_DOWNLOADS", "true").lower() == "true"
        self.download_ttl_minutes: int = int(os.getenv("DOWNLOAD_TTL_MINUTES", "10"))


# Initialize global config instance
config = ServiceConfig() 