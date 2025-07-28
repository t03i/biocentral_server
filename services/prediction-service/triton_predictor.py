"""
Triton client for prediction models.
"""
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import tritonclient.grpc.aio as triton_grpc
from tritonclient.grpc.aio import InferInput, InferRequestedOutput

from models import MODELS_CONFIG, ModelMetadata, get_model_metadata


logger = logging.getLogger(__name__)


class TritonPredictionClient:
    """Async Triton client for protein prediction models."""
    
    def __init__(self, triton_url: str = "triton:8001"):
        self.triton_url = triton_url
        self._client: Optional[triton_grpc.InferenceServerClient] = None
        
        # Build model mappings from configuration
        self.model_map = {
            name: config.triton_name for name, config in MODELS_CONFIG.items()
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self):
        """Initialize Triton client connection."""
        try:
            self._client = triton_grpc.InferenceServerClient(
                url=self.triton_url,
                verbose=False
            )
            
            # Test connection
            if not await self._client.is_server_ready():
                raise RuntimeError(f"Triton server at {self.triton_url} is not ready")
                
            logger.info(f"Connected to Triton server at {self.triton_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise
    
    async def close(self):
        """Close Triton client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Closed Triton client connection")
    
    async def is_model_ready(self, model_name: str) -> bool:
        """Check if a model is ready for inference."""
        if not self._client:
            raise RuntimeError("Client not connected")
            
        triton_model = self.model_map.get(model_name)
        if not triton_model:
            return False
            
        try:
            return await self._client.is_model_ready(triton_model)
        except Exception as e:
            logger.error(f"Error checking model readiness for {model_name}: {e}")
            return False
    
    async def predict_batch(
        self, 
        embeddings: Dict[str, List[float]], 
        model_name: str,
        batch_size: int = 8
    ) -> Dict[str, List[str]]:
        """
        Run batch prediction for a specific model.
        
        Args:
            embeddings: Dictionary mapping sequence IDs to embeddings
            model_name: Name of the model to use (ProtT5Conservation, ProtT5SecondaryStructure)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping sequence IDs to per-residue predictions
        """
        if not self._client:
            raise RuntimeError("Client not connected")
            
        triton_model = self.model_map.get(model_name)
        if not triton_model:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_metadata = get_model_metadata(model_name)
        if not model_metadata:
            raise ValueError(f"No metadata found for model: {model_name}")
        
        # Convert embeddings to numpy array
        seq_ids = sorted(embeddings.keys())
        embedding_matrix = np.array([
            embeddings[seq_id] for seq_id in seq_ids
        ], dtype=np.float32)
        
        logger.info(f"Running {model_name} prediction for {len(seq_ids)} sequences, shape: {embedding_matrix.shape}")
        
        try:
            # Prepare Triton inputs
            inputs = [
                InferInput("input", embedding_matrix.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(embedding_matrix)
            
            # Prepare outputs - use standard output name
            outputs = [
                InferRequestedOutput("output")
            ]
            
            # Run inference
            response = await self._client.infer(
                model_name=triton_model,
                inputs=inputs,
                outputs=outputs
            )
            
            # Get prediction results
            predictions = response.as_numpy("output")  # Shape varies by model
            
            # Format results based on model output type
            results = {}
            
            if model_metadata.output_type == "per_residue" or model_metadata.output_type.value == "per_residue":
                # Per-residue predictions (most models)
                if len(predictions.shape) == 3:  # (batch_size, seq_len, num_classes)
                    class_predictions = np.argmax(predictions, axis=-1)  # Shape: (batch_size, seq_len)
                else:  # (batch_size, seq_len) - already class indices
                    class_predictions = predictions
                
                for i, seq_id in enumerate(seq_ids):
                    seq_predictions = [
                        model_metadata.class_labels[int(pred)] for pred in class_predictions[i]
                    ]
                    results[seq_id] = seq_predictions
                    
            else:  # per_sequence predictions
                if len(predictions.shape) == 2:  # (batch_size, num_classes)
                    class_predictions = np.argmax(predictions, axis=-1)  # Shape: (batch_size,)
                else:  # (batch_size,) - already class indices
                    class_predictions = predictions
                
                for i, seq_id in enumerate(seq_ids):
                    pred_class = model_metadata.class_labels[int(class_predictions[i])]
                    results[seq_id] = [pred_class]  # Keep as list for consistency
                
            logger.info(f"Completed {model_name} prediction for {len(seq_ids)} sequences")
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            raise
    
    async def predict_multiple_models(
        self,
        embeddings: Dict[str, List[float]],
        model_names: List[str],
        batch_size: int = 8
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Run predictions for multiple models on the same embeddings.
        
        Args:
            embeddings: Dictionary mapping sequence IDs to embeddings
            model_names: List of model names to run
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping sequence IDs to model predictions
        """
        # Run all models concurrently
        tasks = [
            self.predict_batch(embeddings, model_name, batch_size)
            for model_name in model_names
        ]
        
        model_results = await asyncio.gather(*tasks)
        
        # Reorganize results by sequence ID
        results = {}
        for seq_id in embeddings.keys():
            results[seq_id] = {}
            for i, model_name in enumerate(model_names):
                results[seq_id][model_name] = model_results[i][seq_id]
                
        return results
    
    async def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get combined metadata for a specific model (local + Triton)."""
        if not self._client:
            raise RuntimeError("Client not connected")
            
        triton_model = self.model_map.get(model_name)
        if not triton_model:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Get local model metadata
        local_metadata = get_model_metadata(model_name)
        if not local_metadata:
            raise ValueError(f"No local metadata found for {model_name}")
            
        try:
            # Get Triton server metadata
            triton_metadata = await self._client.get_model_metadata(triton_model)
            triton_config = await self._client.get_model_config(triton_model)
            
            return {
                # Local metadata
                "name": local_metadata.name,
                "description": local_metadata.description,
                "authors": local_metadata.authors,
                "citation": local_metadata.citation,
                "license": local_metadata.license,
                "outputs": [
                    {
                        "name": output.name,
                        "description": output.description,
                        "output_type": output.output_type.value,
                        "value_type": output.value_type.value,
                        "classes": {k: {"label": v.label, "description": v.description} for k, v in output.classes.items()} if output.classes else None,
                        "value_range": output.value_range,
                        "unit": output.unit
                    }
                    for output in local_metadata.outputs
                ],
                "model_size": local_metadata.model_size,
                "model_size_mb": local_metadata.model_size_mb,
                "testset_performance": local_metadata.testset_performance,
                "training_data_link": local_metadata.training_data_link,
                "embedder": local_metadata.embedder,
                
                # Triton metadata
                "triton_name": triton_model,
                "platform": triton_config.platform,
                "max_batch_size": triton_config.max_batch_size,
                "inputs": [
                    {"name": inp.name, "shape": list(inp.dims), "dtype": inp.data_type} 
                    for inp in triton_config.input
                ],
                "outputs": [
                    {"name": out.name, "shape": list(out.dims), "dtype": out.data_type} 
                    for out in triton_config.output
                ],
                "versions": triton_metadata.versions,
                "triton_state": "READY" if await self.is_model_ready(model_name) else "NOT_READY"
            }
            
        except Exception as e:
            logger.error(f"Failed to get Triton metadata for {model_name}: {e}")
            # Return local metadata only if Triton is unavailable
            return {
                "name": local_metadata.name,
                "description": local_metadata.description,
                "authors": local_metadata.authors,
                "citation": local_metadata.citation,
                "license": local_metadata.license,
                "outputs": [
                    {
                        "name": output.name,
                        "description": output.description,
                        "output_type": output.output_type.value,
                        "value_type": output.value_type.value,
                        "classes": {k: {"label": v.label, "description": v.description} for k, v in output.classes.items()} if output.classes else None,
                        "value_range": output.value_range,
                        "unit": output.unit
                    }
                    for output in local_metadata.outputs
                ],
                "model_size": local_metadata.model_size,
                "model_size_mb": local_metadata.model_size_mb,
                "testset_performance": local_metadata.testset_performance,
                "training_data_link": local_metadata.training_data_link,
                "embedder": local_metadata.embedder,
                "triton_error": str(e)
            }