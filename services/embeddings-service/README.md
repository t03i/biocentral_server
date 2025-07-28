# ProtT5 Embedding Service

Production-ready FastAPI service for computing protein embeddings using ProtT5 via Triton Inference Server with Redis caching.

## Features

- **Triton gRPC Integration**: High-performance ProtT5 embeddings via your existing Triton ensemble
- **Intelligent Caching**: BLOSC2-compressed Redis cache with sliding TTL
- **Connection Pooling**: Async gRPC client pool for concurrent requests
- **Production Ready**: Comprehensive health checks, monitoring, and error handling
- **Optimized Dependencies**: Minimal container size, only necessary dependencies

## Architecture

```
Client Request → FastAPI → Redis Cache Check → Triton gRPC → ProtT5 Pipeline
                     ↓         ↓ (miss)           ↓              ↓
               Cache Response   Triton Client → Tokenizer → ONNX Model
                     ↓              ↓              ↓         ↓
                JSON Response ← Cache Store ← Embeddings (1024-dim FP16→FP32)
```

## API Endpoints

### `POST /embeddings/compute`

Compute embeddings for protein sequences with intelligent caching.

**Request:**
```json
{
  "sequences": ["MKTAL...", "ACDEF..."],
  "batch_size": 8,
  "use_cache": true
}
```

**Response:**
```json
{
  "embeddings": {
    "MKTAL...": [0.123, 0.456, ...],  // 1024-dim array
    "ACDEF...": [0.789, 0.012, ...]
  },
  "cache_stats": {
    "hit_rate": 0.75,
    "compression_ratio": 0.23
  },
  "timing": {
    "total_ms": 150.2,
    "cache_ms": 5.1,
    "triton_ms": 145.1,
    "cache_hits": 1,
    "triton_calls": 1
  },
  "model_info": {
    "model": "prot_t5_pipeline",
    "embedding_dim": "1024",
    "precision": "fp32"
  }
}
```

### `GET /embeddings/health`
Comprehensive health check for all components.

### `GET /embeddings/cache/stats`
Detailed cache statistics and performance metrics.

### `GET /embeddings/model/info`
Triton model metadata and configuration.

## Quick Start

1. **Ensure Triton is running** with your ProtT5 models:
```bash
# Your model repository structure:
services/model-repository/
├── prot_t5_pipeline/     # Ensemble model
├── _internal_tokenizer/  # Python tokenizer
└── _internal_onnx/       # ONNX ProtT5 model
```

2. **Start the service stack**:
```bash
docker-compose -f docker-compose.services.yml up -d
```

3. **Test the embedding service**:
```bash
curl -X POST http://localhost:8001/embeddings/compute \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": ["MKTALVLLFGAILLAHQQGNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNNNNNNNNNN"],
    "batch_size": 4
  }'
```

4. **Check service health**:
```bash
curl http://localhost:8001/embeddings/health
```

## Performance Features

### Redis Caching

- **BLOSC2 Compression**: ~4x compression ratio for 1024-dim FP32 embeddings
- **Sliding TTL**: 1-week cache with automatic refresh on access
- **Batch Operations**: Efficient multi-get/multi-set operations
- **Memory Optimization**: LRU eviction policy with configurable memory limits

### Triton Integration

- **Connection Pooling**: 4 concurrent gRPC connections by default
- **Batch Processing**: Automatic batching up to model's max_batch_size (8)
- **Error Handling**: Graceful fallback and retry logic
- **Health Monitoring**: Continuous Triton server and model health checks

### Production Optimizations

- **Async Everything**: Fully async FastAPI with async Redis and gRPC clients
- **Background Caching**: Non-blocking cache writes using FastAPI BackgroundTasks
- **Input Validation**: Protein sequence validation with detailed error messages
- **Comprehensive Logging**: Structured logging with performance metrics

## Monitoring

The service provides detailed metrics for monitoring:

- **Cache Performance**: Hit rate, compression ratio, Redis memory usage
- **Triton Performance**: Inference timing, batch efficiency, model health
- **Request Metrics**: Response times, throughput, error rates
- **System Health**: Component status, connection health, resource usage

## Configuration

Key environment variables:
- `REDIS_URL`: Redis connection string (default: redis://redis-jobs:6379)
- `TRITON_URL`: Triton gRPC endpoint (default: triton:8001)


## Integration with Existing System

The service integrates seamlessly with your existing biocentral_server:

- **Caddy Routing**: Routes `/embeddings/*` to new service
- **Shared Redis**: Uses same Redis instance for caching and queues
- **Legacy Compatibility**: Existing endpoints continue to work unchanged