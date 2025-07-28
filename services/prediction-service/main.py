"""
FastAPI service for unified protein predictions using Triton inference server.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Union, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from triton_predictor import TritonPredictionClient
from client import EmbeddingServiceClient
from models import MODELS_CONFIG, get_available_models, get_model_metadata, get_all_models_metadata, PredictionType


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client instances
triton_client: Optional[TritonPredictionClient] = None
embedding_client: Optional[EmbeddingServiceClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifespan."""
    global triton_client, embedding_client
    logger.info("Starting Prediction Service...")
    
    # Initialize clients
    triton_client = TritonPredictionClient(triton_url="triton:8001")
    await triton_client.connect()
    
    embedding_client = EmbeddingServiceClient(base_url="http://embeddings-service:8001")
    
    # Verify models are ready
    for model_name in get_available_models():
        is_ready = await triton_client.is_model_ready(model_name)
        logger.info(f"Model {model_name} ready: {is_ready}")
    
    logger.info("Prediction Service started successfully")
    yield
    
    # Cleanup
    if triton_client:
        await triton_client.close()
    logger.info("Prediction Service stopped")


app = FastAPI(
    title="Biocentral Prediction Service",
    description="Unified protein prediction service using Triton inference server",
    version="0.1.0",
    lifespan=lifespan
)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request for protein predictions."""
    sequences: List[str] = Field(..., min_items=1, max_items=50, description="Protein sequences to predict")
    models: List[str] = Field(..., min_items=1, description="List of models to use for prediction")
    batch_size: Optional[int] = Field(default=8, ge=1, le=16, description="Batch size for processing")


class PredictionResponse(BaseModel):
    """Response containing predictions."""
    predictions: Dict[str, Dict[str, List[str]]]  # seq_id -> model -> per-residue predictions
    model_metadata: Dict[str, Dict[str, Any]]
    timing: Dict[str, float]
    embeddings_timing: Dict[str, float]


class EmbeddingRequest(BaseModel):
    """Request to embedding service."""
    sequences: List[str]
    batch_size: Optional[int] = 8


async def get_embeddings(sequences: List[str], batch_size: int = 8) -> Dict[str, Any]:
    """Fetch embeddings from the embedding service."""
    if not embedding_client:
        raise HTTPException(status_code=503, detail="Embedding client not initialized")
        
    return await embedding_client.get_embeddings(sequences, batch_size)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    total_models = len(get_available_models())
    
    return {
        "message": "Biocentral Prediction Service",
        "version": "0.1.0",
        "description": "Unified protein prediction service using Triton inference server",
        "models": {
            "total_configured": total_models,
            "available_models": get_available_models()
        },
        "endpoints": {
            "predictions": [
                "/predictions/predict",
                "/predictions/conservation", 
                "/predictions/secondary-structure"
            ],
            "metadata": [
                "/models/list",
                "/models/{model_name}/metadata",
                "/models/ready",
                "/models/numerical-ranges",
                "/models/output-types"
            ],
            "monitoring": [
                "/health",
                "/docs",
                "/openapi.json"
            ]
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not triton_client:
        raise HTTPException(status_code=503, detail="Triton client not initialized")
        
    # Check all models
    model_status = {}
    
    for model_name in get_available_models():
        try:
            is_ready = await triton_client.is_model_ready(model_name)
            model_status[model_name] = "ready" if is_ready else "not_ready"
        except Exception as e:
            model_status[model_name] = f"error: {str(e)}"
    
    # Check embedding service
    embedding_health = await embedding_client.health_check() if embedding_client else {"status": "not_connected"}
    
    ready_models = sum(1 for status in model_status.values() if status == "ready")
    embedding_healthy = embedding_health.get("status") == "healthy"
    
    return {
        "status": "healthy" if embedding_healthy else "degraded",
        "triton_connection": "connected" if triton_client else "disconnected",
        "embedding_service": embedding_health,
        "models": {
            "ready": ready_models,
            "total": len(model_status),
            "status": model_status
        }
    }


@app.post("/predictions/predict", response_model=PredictionResponse)
async def predict_sequences(request: PredictionRequest):
    """
    Unified prediction endpoint - automatically fetches embeddings and runs predictions.
    """
    if not triton_client:
        raise HTTPException(status_code=503, detail="Prediction client not initialized")
    
    start_time = time.time()
    
    try:
        # 1. Get embeddings from embedding service
        embeddings_response = await get_embeddings(
            request.sequences, 
            request.batch_size
        )
        
        embeddings_dict = embeddings_response["embeddings"]
        embeddings_timing = embeddings_response["timing"]
        
        # 2. Run predictions via Triton
        logger.info(f"Running predictions for models: {request.models}")
        prediction_start = time.time()
        
        predictions = await triton_client.predict_multiple_models(
            embeddings=embeddings_dict,
            model_names=request.models,
            batch_size=request.batch_size
        )
        
        prediction_time = time.time() - prediction_start
        total_time = time.time() - start_time
        
        # 3. Get model metadata
        model_metadata = {}
        for model_name in request.models:
            try:
                metadata = await triton_client.get_model_metadata(model_name)
                model_metadata[model_name] = metadata
            except Exception as e:
                logger.error(f"Failed to get metadata for {model_name}: {e}")
                model_metadata[model_name] = {"error": str(e)}
        
        # 4. Map predictions back to original sequences
        final_predictions = {}
        for i, sequence in enumerate(request.sequences):
            seq_key = f"seq_{i}"  # Use a more descriptive key
            final_predictions[seq_key] = predictions.get(str(i), {})
        
        return PredictionResponse(
            predictions=final_predictions,
            model_metadata=model_metadata,
            timing={
                "total_ms": total_time * 1000,
                "triton_prediction_ms": prediction_time * 1000,
                "formatting_ms": (total_time - prediction_time - embeddings_timing.get("embedding_computation_ms", 0) / 1000) * 1000
            },
            embeddings_timing=embeddings_timing
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predictions/conservation")
async def predict_conservation(sequences: List[str], batch_size: int = 8):
    """Convenience endpoint for conservation prediction only."""
    request = PredictionRequest(
        sequences=sequences,
        models=["ProtT5Conservation"],
        batch_size=batch_size
    )
    return await predict_sequences(request)


@app.post("/predictions/secondary-structure") 
async def predict_secondary_structure(sequences: List[str], batch_size: int = 8):
    """Convenience endpoint for secondary structure prediction only."""
    request = PredictionRequest(
        sequences=sequences,
        models=["ProtT5SecondaryStructure"],
        batch_size=batch_size
    )
    return await predict_sequences(request)


# === MODEL METADATA ENDPOINTS ===

@app.get("/models/list")
async def list_models():
    """List all configured models with their status."""
    if not triton_client:
        raise HTTPException(status_code=503, detail="Triton client not initialized")
    
    model_info = {}
    
    for model_name in get_available_models():
        try:
            local_metadata = get_model_metadata(model_name)
            is_ready = await triton_client.is_model_ready(model_name)
            
            model_info[model_name] = {
                "name": model_name,
                "description": local_metadata.description if local_metadata else "No description",
                "output_type": local_metadata.output_type.value if local_metadata else "unknown",
                "output_classes": local_metadata.output_classes if local_metadata else "unknown",
                "triton_status": "ready" if is_ready else "not_ready",
                "implemented": is_ready  # Whether model is actually available
            }
        except Exception as e:
            model_info[model_name] = {
                "name": model_name,
                "triton_status": "error",
                "error": str(e),
                "implemented": False
            }
    
    # Summary statistics
    total_models = len(model_info)
    ready_models = sum(1 for m in model_info.values() if m.get("triton_status") == "ready")
    output_types = {}
    for m in model_info.values():
        output_type = m.get("output_type", "unknown")
        output_types[output_type] = output_types.get(output_type, 0) + 1
    
    return {
        "models": model_info,
        "summary": {
            "total_models": total_models,
            "ready_models": ready_models,
            "models_by_output_type": output_types
        }
    }


@app.get("/models/{model_name}/metadata")
async def get_model_metadata_endpoint(model_name: str):
    """Get detailed metadata for a specific model."""
    if not triton_client:
        raise HTTPException(status_code=503, detail="Triton client not initialized")
    
    if model_name not in get_available_models():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        metadata = await triton_client.get_model_metadata(model_name)
        return {"model": model_name, "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")




@app.get("/models/ready")
async def get_ready_models():
    """Get only models that are ready for inference."""
    if not triton_client:
        raise HTTPException(status_code=503, detail="Prediction client not initialized")
    
    ready_models = {}
    
    for model_name in get_available_models():
        try:
            if await triton_client.is_model_ready(model_name):
                metadata = get_model_metadata(model_name)
                ready_models[model_name] = {
                    "name": model_name,
                    "description": metadata.description if metadata else "No description",
                    "output_type": metadata.output_type.value if metadata else "unknown",
                    "class_labels": metadata.class_labels if metadata else {}
                }
        except Exception as e:
            logger.error(f"Error checking model {model_name}: {e}")
    
    return {
        "ready_models": ready_models,
        "count": len(ready_models)
    }


@app.get("/models/numerical-ranges")
async def get_numerical_ranges():
    """Get numerical range information for all models with continuous outputs."""
    numerical_models = {}
    
    for model_name, config in get_all_models_metadata().items():
        model_ranges = []
        for output in config.outputs:
            if output.value_type == ValueType.FLOAT and output.value_range:
                model_ranges.append({
                    "output_name": output.name,
                    "description": output.description,
                    "value_range": output.value_range,
                    "unit": output.unit,
                    "range_description": f"{output.value_range[0]} to {output.value_range[1]}" + (f" {output.unit}" if output.unit else "")
                })
        
        if model_ranges:
            numerical_models[model_name] = {
                "description": config.description,
                "outputs": model_ranges
            }
    
    return {
        "numerical_models": numerical_models,
        "count": len(numerical_models),
        "summary": f"Found {len(numerical_models)} models with numerical ranges"
    }
    

@app.get("/models/output-types")
async def get_output_types():
    """Get detailed output type information for all models."""
    output_analysis = {}
    
    for model_name, config in get_all_models_metadata().items():
        model_outputs = []
        for output in config.outputs:
            output_info = {
                "name": output.name,
                "description": output.description,
                "output_type": output.output_type.value,
                "value_type": output.value_type.value,
            }
            
            if output.classes:
                output_info["num_classes"] = len(output.classes)
                output_info["class_labels"] = {k: v.label for k, v in output.classes.items()}
            elif output.value_range:
                output_info["value_range"] = output.value_range
                output_info["unit"] = output.unit
                
            model_outputs.append(output_info)
        
        output_analysis[model_name] = {
            "description": config.description,
            "outputs": model_outputs
        }
    
    return {
        "models": output_analysis,
        "summary": {
            "total_models": len(output_analysis),
            "categorical_outputs": sum(1 for config in get_all_models_metadata().values() 
                                     for output in config.outputs if output.classes),
            "numerical_outputs": sum(1 for config in get_all_models_metadata().values() 
                                   for output in config.outputs if output.value_range)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)