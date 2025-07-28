import asyncio
import json
import numpy as np
from typing import Dict, List, Any
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

class ModelInterface:
    """Base class for prediction models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    async def predict_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Run batch prediction - to be implemented by subclasses"""
        raise NotImplementedError
    
    def format_output(self, raw_output: np.ndarray, seq_id: str) -> Dict[str, Any]:
        """Format model output - to be implemented by subclasses"""
        return {"raw_output": raw_output.tolist()}

class ProteinFunctionModel(ModelInterface):
    """Protein function classification model"""
    
    def __init__(self):
        super().__init__("protein_function_classifier")
        self.class_names = ["enzyme", "transport", "binding", "structural", "unknown"]
    
    async def predict_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Placeholder prediction - TODO: integrate with existing models"""
        batch_size = embeddings.shape[0]
        # Return dummy predictions for now
        return np.random.random((batch_size, len(self.class_names))).astype(np.float32)
    
    def format_output(self, raw_output: np.ndarray, seq_id: str) -> Dict[str, Any]:
        probabilities = raw_output.tolist()
        predicted_class_idx = int(np.argmax(raw_output))
        
        return {
            "model_type": "classification",
            "sequence_id": seq_id,
            "predicted_class": self.class_names[predicted_class_idx],
            "confidence": float(np.max(raw_output)),
            "class_probabilities": {
                class_name: prob 
                for class_name, prob in zip(self.class_names, probabilities)
            }
        }

class StabilityModel(ModelInterface):
    """Protein stability regression model"""
    
    def __init__(self):
        super().__init__("stability_predictor")
    
    async def predict_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Placeholder prediction - TODO: integrate with existing models"""
        batch_size = embeddings.shape[0]
        # Return dummy predictions for now
        return np.random.random((batch_size, 1)).astype(np.float32)
    
    def format_output(self, raw_output: np.ndarray, seq_id: str) -> Dict[str, Any]:
        stability_score = float(raw_output[0])
        
        return {
            "model_type": "regression",
            "sequence_id": seq_id,
            "stability_score": stability_score,
            "stability_level": (
                "high" if stability_score > 0.7 else 
                "medium" if stability_score > 0.3 else 
                "low"
            )
        }

class PredictionService:
    """Unified prediction service handling multiple models"""
    
    def __init__(self):
        self.queue_manager = QueueManager(redis_url="redis://redis-jobs:6379")
        
        # Register available models
        self.models = {
            "protein_function": ProteinFunctionModel(),
            "stability": StabilityModel(),
            # TODO: Add more models from existing biocentral_server/predict/models/
        }
    
    async def run_worker(self):
        """Main worker loop processing prediction jobs"""
        
        logger.info("Starting prediction worker")
        
        while True:
            try:
                # Wait for jobs
                job_data = await self.queue_manager.redis.brpop(
                    self.queue_manager.PREDICTION_QUEUE,
                    timeout=5
                )
                
                if not job_data:
                    continue
                
                _, job_json = job_data
                job_info = json.loads(job_json)
                job_id = job_info["job_id"]
                
                logger.info(f"Processing prediction job {job_id}")
                
                # Process predictions
                predictions = await self._process_predictions(
                    job_info["embeddings"],
                    job_info["models"]
                )
                
                # Update job results
                await self.queue_manager.complete_job(
                    job_id, predictions=predictions
                )
                
                logger.info(f"Completed prediction job {job_id}")
                
            except Exception as e:
                logger.error(f"Prediction worker error: {e}")
                if 'job_id' in locals():
                    job = await self.queue_manager.get_job(job_id)
                    if job:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        await self.queue_manager.update_job(job)
    
    async def _process_predictions(
        self, 
        embeddings: Dict[str, List[float]], 
        model_names: List[str]
    ) -> Dict[str, Any]:
        """Process predictions for all requested models"""
        
        results = {}
        
        # Convert embeddings to numpy
        embedding_matrix = np.array([
            embeddings[seq_id] for seq_id in sorted(embeddings.keys())
        ], dtype=np.float32)
        
        seq_ids = sorted(embeddings.keys())
        
        # Process each model
        for model_name in model_names:
            if model_name not in self.models:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            model = self.models[model_name]
            
            # Run batch prediction
            batch_outputs = await model.predict_batch(embedding_matrix)
            
            # Format outputs for each sequence
            model_results = {}
            for i, seq_id in enumerate(seq_ids):
                formatted_output = model.format_output(batch_outputs[i], seq_id)
                model_results[seq_id] = formatted_output
            
            results[model_name] = model_results
        
        return results

# Global service instance
prediction_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle"""
    global prediction_service
    
    # Startup
    prediction_service = PredictionService()
    
    # Start worker task
    worker_task = asyncio.create_task(prediction_service.run_worker())
    
    yield
    
    # Shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Biocentral Prediction Service", 
    version="1.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    embeddings: Dict[str, List[float]]
    models: List[str]

@app.post("/predictions/predict")
async def predict_from_embeddings(request: PredictionRequest):
    """Run predictions on pre-computed embeddings"""
    try:
        # Process predictions directly (not queued for now)
        predictions = await prediction_service._process_predictions(
            request.embeddings,
            request.models
        )
        
        return {"predictions": predictions, "status": "completed"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/models")
async def list_available_models():
    """List available prediction models"""
    return {
        "models": list(prediction_service.models.keys()),
        "model_info": {
            name: {
                "name": model.model_name,
                "type": "classification" if isinstance(model, ProteinFunctionModel) else "regression"
            }
            for name, model in prediction_service.models.items()
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await prediction_service.queue_manager.redis.ping()
        return {"status": "healthy", "service": "prediction"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)