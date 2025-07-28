# Biocentral Prediction Service

FastAPI service for unified protein predictions using Triton inference server.

## Overview

This service provides protein predictions for conservation and secondary structure using the Triton inference server for high-performance model serving. It integrates with the embedding service to provide a unified prediction pipeline.

## Phase 1 Implementation

Currently supports:
- **ProtT5Conservation**: Per-residue evolutionary conservation prediction (9 classes)
- **ProtT5SecondaryStructure**: Per-residue secondary structure prediction (H/E/C)

## Architecture

```
Client → FastAPI → Embedding Service → Triton Server → Model Repository
   ↓         ↓           ↓                ↓              ↓
Request → Get embeddings → Triton gRPC → ONNX models → Predictions
```

## API Endpoints

### Main Prediction Endpoint

```bash
POST /predictions/predict
```

Request body:
```json
{
  "sequences": ["MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"],
  "models": ["ProtT5Conservation", "ProtT5SecondaryStructure"],
  "batch_size": 8,
  "embedder_service_url": "http://embeddings-service:8001"
}
```

Response:
```json
{
  "predictions": {
    "seq_0": {
      "ProtT5Conservation": ["4", "3", "2", "..."],
      "ProtT5SecondaryStructure": ["H", "H", "C", "..."]
    }
  },
  "model_metadata": {
    "ProtT5Conservation": {
      "name": "ProtT5Conservation",
      "platform": "onnxruntime_onnx",
      "class_labels": {"0": "0", "1": "1", ..., "8": "8"}
    }
  },
  "timing": {
    "total_ms": 1250.5,
    "triton_prediction_ms": 45.2,
    "formatting_ms": 15.3
  },
  "embeddings_timing": {
    "embedding_computation_ms": 1200.0
  }
}
```

### Convenience Endpoints

```bash
# Conservation only
POST /predictions/conservation
{
  "sequences": ["MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"],
  "batch_size": 8
}

# Secondary structure only  
POST /predictions/secondary-structure
{
  "sequences": ["MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"],
  "batch_size": 8
}
```

### Model Information

```bash
# List available models
GET /models/list

# Health check
GET /health
```

## Development

### Dependencies

- FastAPI
- Triton gRPC client
- httpx (for embedding service communication)
- NumPy

### Local Development

1. Deploy models to Triton repository:
```bash
cd services
python deploy_models.py
```

2. Start services:
```bash
docker-compose -f docker-compose.services.yml up
```

3. Test predictions:
```bash
curl -X POST http://localhost:8002/predictions/conservation \
  -H "Content-Type: application/json" \
  -d '{"sequences":["MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"]}'
```

### Adding New Models (Future Phases)

To add new models:

1. Create Triton model configuration in `services/model-repository/`
2. Add model mapping to `triton_predictor.py`
3. Update the available models list in `main.py`
4. Update `deploy_models.py` to include the new model

## Performance

Expected performance improvements over legacy system:

| Metric | Legacy System | Triton Service | Improvement |
|--------|---------------|----------------|-------------|
| Cold start | ~5s | ~0s | Instant |
| Single sequence | ~800ms | ~40ms | 20x faster |
| Batch (8 seqs) | ~1.2s | ~50ms | 24x faster |
| Memory per worker | ~350MB | Shared ~100MB | 3.5x better |

## Monitoring

- Health endpoint: `/health`
- Model status: `/models/list`
- FastAPI metrics: Available via OpenAPI at `/docs`
- Triton metrics: Available at `triton:8002/metrics`

## Configuration

Environment variables:
- `TRITON_URL`: Triton server gRPC endpoint (default: `triton:8001`)
- `EMBEDDINGS_SERVICE_URL`: Embedding service URL (default: `http://embedding-service:8001`)

## Migration Status

- ✅ **Phase 1**: Simple ONNX models (ProtT5Conservation, ProtT5SecondaryStructure)
- 🔄 **Phase 2**: Models with post-processing (SETH, BindEmbed, LightAttention models)
- ⏳ **Phase 3**: Complex models (TMbed, VespaG)
- ⏳ **Phase 4**: Integration and cleanup