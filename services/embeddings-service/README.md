# Protein Embedding Service

High-performance protein sequence embedding service using ProtT5 and ESM-2 models via Triton Inference Server with Redis caching.

## Features

- **Multiple Models**: Support for ProtT5, ESM-2 (650M), and ESM-2 (3B) models
- **Intelligent Caching**: Redis-based caching with compression for optimal performance
- **Binary Downloads**: Compressed numpy downloads for efficient data transfer
- **Full Sequence Embeddings**: Support for both pooled and full sequence embeddings
- **Differently Sized Sequences**: Proper handling of sequences with different lengths
- **Health Monitoring**: Comprehensive health checks for all components
- **Configurable**: Environment-based configuration for easy deployment

## API Endpoints

### 1. Compute Embeddings

**POST** `/embeddings/compute/{model}`

Compute protein embeddings using the specified model.

**Parameters:**
- `model`: Model name (`prot_t5`, `esm2_t33_650M`, `esm2_t36_3B`)
- `sequences`: List of protein sequences
- `pooled`: Boolean flag for pooled vs full embeddings

**Example Request:**
```bash
curl -X POST "http://localhost:8000/embeddings/compute/prot_t5" \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "MKTAYIAELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ],
    "pooled": false
  }'
```

**Example Response:**
```json
{
  "sequences": [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "MKTAYIAELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
  ],
  "cache_stats": {
    "hit_rate": 0.8,
    "total_hits": 4,
    "total_requests": 5,
    "cache_misses": 1
  },
  "timing": {
    "total_ms": 150.5,
    "cache_ms": 2.1,
    "triton_ms": 148.4,
    "cache_hits": 4,
    "triton_calls": 1
  },
  "model_info": {
    "model": "prot_t5",
    "triton_model": "prot_t5_pipeline",
    "embedding_dim": 1024,
    "sequence_length_range": "10-5000",
    "description": "ProtT5 transformer model for protein embeddings",
    "precision": "fp16",
    "pooled": false,
    "shape_description": "[sequences, sequence_length, 1024] (full)"
  },
  "download_link": "/embeddings/download/prot_t5/numpy/1703123456_12345"
}
```

### 2. Download Embeddings

**GET** `/embeddings/download/{model}/numpy/{request_id}`

Download computed embeddings as compressed numpy arrays.

**Parameters:**
- `model`: Model name used for computation
- `request_id`: Request ID from the compute endpoint response

**Example:**
```bash
curl -O "http://localhost:8000/embeddings/download/prot_t5/numpy/1703123456_12345"
```

**Download Format:**
The downloaded `.npz` file contains multiple arrays:

- **embedding_0, embedding_1, etc.**: Individual embeddings for each sequence
- **sequence_count**: Number of sequences
- **storage_type**: Always "list_of_arrays"

Each embedding array has shape `[sequence_length, embedding_dim]` for full embeddings or `[embedding_dim]` for pooled embeddings.

### 3. Health Check

**GET** `/embeddings/health`

Comprehensive health check for all service components.

**Example Response:**
```json
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
```

### 4. Cache Statistics

**GET** `/embeddings/cache/stats`

Get detailed cache statistics.

**Example Response:**
```json
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
```

### 5. Clear Cache

**POST** `/embeddings/cache/clear`

Clear cache entries (admin endpoint).

**Parameters:**
- `pattern`: Optional Redis pattern to match keys for deletion

**Examples:**
```bash
# Clear all cache entries
curl -X POST "http://localhost:8000/embeddings/cache/clear"

# Clear only ProtT5 cache entries
curl -X POST "http://localhost:8000/embeddings/cache/clear?pattern=prot_t5_*"

# Clear only ESM-2 cache entries
curl -X POST "http://localhost:8000/embeddings/cache/clear?pattern=esm2_*"
```

### 6. Model Information

**GET** `/embeddings/model/info`

Get detailed information about a specific model.

**Parameters:**
- `model`: Model type (default: `prot_t5`)

**Example:**
```bash
curl "http://localhost:8000/embeddings/model/info?model=prot_t5"
```

**Example Response:**
```json
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
```

### 7. List Available Models

**GET** `/embeddings/models`

List all available embedding models.

**Example Response:**
```json
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
```

### 8. Service Configuration

**GET** `/embeddings/config`

Get current service configuration.

**Example Response:**
```json
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
```

### 9. Root Endpoint

**GET** `/`

Basic service information.

**Example Response:**
```json
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
```

## Handling Differently Sized Sequences

The service properly handles sequences of different lengths through Triton's attention mask mechanism:

### Batching with Attention Masks
Triton handles sequences of different lengths efficiently by using attention masks internally. This allows the model to process multiple sequences in a single batch while properly masking padding tokens.

### Download Format
All downloads use a consistent format with individual arrays for each sequence:

**Download Structure:**
```python
# Load the downloaded .npz file
import numpy as np
data = np.load('embeddings_prot_t5_1703123456_12345.npz')

# Get sequence count and load embeddings
sequence_count = data['sequence_count']
embeddings = []
for i in range(sequence_count):
    emb = data[f'embedding_{i}']
    print(f"Sequence {i}: shape {emb.shape}")  # e.g., (67, 1024), (45, 1024)
    embeddings.append(emb)
```

**Example with Different Lengths:**
```python
# Sequence 1: 67 amino acids -> shape (67, 1024)
# Sequence 2: 45 amino acids -> shape (45, 1024)
# Download contains: embedding_0 (67, 1024), embedding_1 (45, 1024)
```

This approach ensures that:
1. **Efficient batching**: Triton processes all sequences in a single batch using attention masks
2. **No data loss**: Each sequence's full embedding is preserved
3. **Consistent format**: All downloads use the same structure regardless of sequence lengths
4. **Easy access**: Clear naming convention for accessing individual embeddings

## Configuration

The service is configured via environment variables:

```bash
# Triton settings
TRITON_URL=triton:8000
TRITON_POOL_SIZE=5
TRITON_TIMEOUT=30

# Redis settings
REDIS_URL=redis://redis:6379
REDIS_TTL_DAYS=30
REDIS_COMPRESSION_LEVEL=6

# Sequence limits
MIN_SEQUENCE_LENGTH=10
MAX_SEQUENCE_LENGTH=5000
MAX_SEQUENCES_PER_REQUEST=100

# Download settings
DOWNLOAD_TTL_MINUTES=10

# Logging
LOG_LEVEL=INFO
```

## Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t protein-embedding-service .
docker run -p 8000:8000 protein-embedding-service
```

## Performance & Scaling

### Current Performance Characteristics

- **Caching**: Redis-based caching with compression reduces computation time by 80%+ for repeated sequences
- **Batch Processing**: Efficient batch processing via Triton Inference Server
- **Binary Downloads**: Compressed numpy format for fast data transfer
- **Connection Pooling**: Triton client pooling for optimal resource utilization

### Scaling Behavior

The service has been load-tested and exhibits the following scaling characteristics:

#### **Concurrent Request Handling**

- **1-4 concurrent requests**: Handled efficiently with all requests processed simultaneously
- **5+ concurrent requests**: Queuing occurs as requests wait for available Triton clients
- **Cache hits**: Sub-100ms response times regardless of concurrency
- **Cache misses**: 3-8 second processing time per request

#### **Resource Limits**

- **Triton Client Pool**: 4 concurrent connections (configurable)
- **Max Sequences per Request**: 100 sequences (configurable)
- **Memory Usage**: ~1GB per large request (100 sequences × 5000 aa)
- **CPU Bottleneck**: CPU-only Triton limits processing speed

#### **Load Test Results**

```
4 Concurrent Requests (Cache Misses):
- Request 1: 6.68s (Triton processing)
- Request 2: 3.53s (Triton processing)
- Request 3: 0.02s (Cache hit)
- Request 4: 0.02s (Cache hit)

6 Concurrent Requests (Cache Misses):
- Request 5: 7.37s (Queued, then Triton processing)
- Request 6: 3.92s (Queued, then Triton processing)
- Request 7-10: 0.02-0.06s (Cache hits)
```

### Optimization Roadmap

#### **Immediate Improvements (This Week)**

1. **Increase Triton Pool Size**: Double pool size from 4 to 8 clients
   ```bash
   TRITON_POOL_SIZE=8
   ```
2. **Add Request Semaphore**: Implement concurrency control
   ```python
   MAX_CONCURRENT_REQUESTS = 8
   request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
   ```
3. **Reduce Request Size Limits**: Lower max sequences per request to 50
   ```bash
   MAX_SEQUENCES_PER_REQUEST=50
   ```

#### **Medium-term Improvements (Next Week)**
1. **GPU Acceleration**: Deploy GPU-enabled Triton for 10x performance improvement
2. **Horizontal Scaling**: Add multiple service instances
   ```yaml
   embedding-service:
     deploy:
       replicas: 3
   ```
3. **Request Batching**: Implement intelligent request batching

#### **Long-term Architecture (Next Month)**
1. **Queue-based Architecture**: Separate API from processing
   ```python
   # Submit job to queue
   job_id = await submit_embedding_job(sequences)
   
   # Poll for results
   result = await get_job_result(job_id)
   ```
2. **Microservice Split**: Separate components
   - API Gateway
   - Embedding Workers
   - Cache Service
   - Download Service
3. **Auto-scaling**: Dynamic scaling based on queue depth

### Performance Projections

| Configuration | Concurrent Requests | Processing Time | Throughput |
|---------------|-------------------|-----------------|------------|
| Current (CPU) | 4 | 3-8s | 8 req/min |
| GPU + Pool 8 | 8 | 0.3-0.8s | 80 req/min |
| GPU + 3 Instances | 24 | 0.3-0.8s | 240 req/min |
| Queue-based | Unlimited | 0.3-0.8s | 300+ req/min |

## Monitoring

- **Health Checks**: Comprehensive health monitoring for all components
- **Cache Statistics**: Detailed cache performance metrics
- **Timing Information**: Request-level timing breakdowns
- **Model Status**: Real-time model availability status

## Implementation Plans

### Immediate Optimizations (Ready for Implementation)

#### **1. Request Semaphore Implementation**
```python
# Add to main.py
from asyncio import Semaphore

# Global semaphore to limit concurrent requests
MAX_CONCURRENT_REQUESTS = 8
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

@app.post("/embeddings/compute/{model}")
async def compute_embeddings(...):
    async with request_semaphore:
        # Process request
        # Return 429 Too Many Requests if semaphore is full
```

**Benefits:**
- Prevents resource exhaustion
- Graceful degradation under load
- Clear error messages for users

#### **2. Enhanced Connection Pool Configuration**
```python
# Update config.py
self.triton_pool_size: int = int(os.getenv("TRITON_POOL_SIZE", "8"))  # Double from 4
self.max_sequences_per_request: int = int(os.getenv("MAX_SEQUENCES_PER_REQUEST", "50"))  # Reduce from 100
```

**Benefits:**
- Increased concurrent capacity (4 → 8 requests)
- Reduced memory pressure per request
- Better resource utilization

#### **3. Request Size Optimization**
```python
# Add request size validation
if len(sequences) > config.max_sequences_per_request:
    raise HTTPException(
        status_code=413, 
        detail=f"Too many sequences. Max: {config.max_sequences_per_request}"
    )
```

### Queue-based Architecture Design

#### **Phase 1: Job Queue Implementation**
```python
# New endpoints for queue-based processing
@app.post("/embeddings/jobs/submit")
async def submit_embedding_job(sequences: List[str], model: str):
    job_id = generate_job_id()
    await redis.lpush("embedding_jobs", json.dumps({
        "job_id": job_id,
        "sequences": sequences,
        "model": model,
        "status": "queued"
    }))
    return {"job_id": job_id, "status": "queued"}

@app.get("/embeddings/jobs/{job_id}")
async def get_job_status(job_id: str):
    result = await redis.get(f"job_result:{job_id}")
    return json.loads(result) if result else {"status": "processing"}
```

#### **Phase 2: Worker Service Separation**
```yaml
# docker-compose.yml
services:
  embedding-api:
    # API gateway only
    ports: ["8000:8000"]
    
  embedding-worker:
    # Processing workers
    deploy:
      replicas: 3
      
  embedding-cache:
    # Dedicated cache service
    image: redis:7-alpine
```

#### **Phase 3: Auto-scaling**
```python
# Auto-scaling based on queue depth
async def scale_workers():
    queue_depth = await redis.llen("embedding_jobs")
    if queue_depth > 10:
        # Scale up workers
        await scale_up_workers()
    elif queue_depth < 2:
        # Scale down workers
        await scale_down_workers()
```