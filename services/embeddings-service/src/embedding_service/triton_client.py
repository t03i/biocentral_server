"""
Triton gRPC client for ProtT5 embeddings
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
import numpy as np

import tritonclient.grpc.aio as triton_grpc
from tritonclient.grpc import service_pb2


logger = logging.getLogger(__name__)

class TritonClientPool:
    """Pool of Triton gRPC clients for efficient connection management"""
    
    def __init__(self, triton_url: str, model_name: str, pool_size: int = 4, timeout: float = 30.0):
        self.triton_url = triton_url
        self.model_name = model_name
        self.pool_size = pool_size
        self.timeout = timeout
        self._clients = asyncio.Queue()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the client pool"""
        if self._initialized:
            return
            
        logger.debug(f"Initializing Triton client pool for {self.model_name}")
        
        for i in range(self.pool_size):
            client = triton_grpc.InferenceServerClient(
                url=self.triton_url,
                verbose=False
            )
            await self._clients.put(client)
        
        # Test connection with first client
        test_client = await self._clients.get()
        try:
            await test_client.is_server_ready()
            logger.info(f"✅ Triton client pool initialized for {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Triton server: {e}")
            raise
        finally:
            await self._clients.put(test_client)
        
        self._initialized = True
    
    async def get_client(self) -> triton_grpc.InferenceServerClient:
        """Get a client from the pool"""
        if not self._initialized:
            raise RuntimeError("Client pool not initialized")
        return await self._clients.get()
    
    async def return_client(self, client: triton_grpc.InferenceServerClient):
        """Return a client to the pool"""
        await self._clients.put(client)
    
    async def close_all(self):
        """Close all clients in the pool"""
        while not self._clients.empty():
            client = await self._clients.get()
            await client.close()
        self._initialized = False

class TritonEmbeddingClient:
    """Wrapper for Triton embedding computations"""
    
    def __init__(self, client: triton_grpc.InferenceServerClient, model_name: str, timeout: float = 30.0):
        self._client = client
        self.model_name = model_name
        self.timeout = timeout
    
    async def compute_embeddings_batch(self, sequences: List[str], pooled: bool = False) -> List[np.ndarray]:
        """
        Compute embeddings for a batch of sequences
        
        Args:
            sequences: List of protein sequences
            pooled: If True, return mean-pooled embeddings (1D per sequence). If False, return full sequence embeddings (2D per sequence)
        
        Returns:
            List of numpy arrays, one per sequence. Shape depends on pooled parameter:
            - pooled=True: Each array has shape [embedding_dim] 
            - pooled=False: Each array has shape [sequence_length, embedding_dim]
        """
        logger.debug(f"Computing embeddings for {len(sequences)} sequences (pooled={pooled})")
        
        try:
            # Process all sequences in a single batch (Triton handles dynamic batching)
            embeddings = await self._process_batch(sequences, pooled=pooled)
            logger.debug(f"Successfully computed {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to process sequences: {e}")
            raise

    async def _process_batch(self, sequences: List[str], pooled: bool = False) -> List[np.ndarray]:
        """Process a single batch through Triton"""
        
        try:
            logger.debug(f"Processing batch of {len(sequences)} sequences (pooled={pooled})")
            
            # Prepare input tensor for Triton ensemble
            # Model expects 2D input: [batch_size, 1] for sequence data
            sequences_array = np.array(sequences, dtype=object).reshape(-1, 1)
            logger.debug(f"Input sequences array shape: {sequences_array.shape}, dtype: {sequences_array.dtype}")
            
            # Create Triton input
            try:
                inputs = [
                    triton_grpc.InferInput(
                        "sequences", 
                        sequences_array.shape, 
                        "BYTES"
                    )
                ]
                inputs[0].set_data_from_numpy(sequences_array)
                logger.debug(f"Created Triton input tensor with shape {sequences_array.shape}")
            except Exception as e:
                logger.error(f"Failed to create input tensor: {e}")
                raise
            
            # Create Triton output request
            outputs = [
                triton_grpc.InferRequestedOutput("embeddings")
            ]
            
            # Make inference request
            try:
                logger.debug(f"Making Triton inference request to model {self.model_name}")
                response = await self._client.infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs,
                    timeout=int(self.timeout)
                )
                logger.debug(f"Received Triton response successfully")
                
                # Extract embeddings (FP16 -> FP32 for consistency)
                embeddings_tensor = response.as_numpy("embeddings").astype(np.float32)
                
                # Log tensor shape for debugging
                logger.debug(f"Received embeddings tensor shape: {embeddings_tensor.shape}")
                
                # Validate tensor shape: should be [batch_size, sequence_length, embedding_dim]
                if len(embeddings_tensor.shape) != 3:
                    raise ValueError(f"Expected 3D embeddings tensor, got shape {embeddings_tensor.shape}")
                
                batch_size, seq_len, embed_dim = embeddings_tensor.shape
                if batch_size != len(sequences):
                    raise ValueError(f"Batch size mismatch: expected {len(sequences)}, got {batch_size}")
                
                # Process embeddings based on pooling preference
                embeddings = []
                for i in range(len(sequences)):
                    # Get embedding for sequence i: [sequence_length, embedding_dim]
                    seq_embeddings = embeddings_tensor[i]
                    
                    if pooled:
                        # Mean pool across sequence dimension to get [embedding_dim]
                        pooled_embedding = np.mean(seq_embeddings, axis=0)
                        embeddings.append(pooled_embedding)
                        logger.debug(f"Sequence {i}: pooled to shape {pooled_embedding.shape}")
                    else:
                        # Keep full sequence embeddings: [sequence_length, embedding_dim]
                        embeddings.append(seq_embeddings)
                        logger.debug(f"Sequence {i}: full shape {seq_embeddings.shape}")
                
                logger.debug(f"Processed {len(embeddings)} embeddings (pooled={pooled})")
                return embeddings
                
            except Exception as e:
                logger.error(f"Triton inference failed: {e}")
                raise
                
        except Exception as e:
            import traceback
            logger.error(f"Full _process_batch error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def health_check(self) -> Dict[str, bool]:
        """Check Triton server and model health"""
        try:
            # Check server health
            server_ready = await self._client.is_server_ready()
            server_live = await self._client.is_server_live()
            
            # Check model health
            try:
                model_ready = await self._client.is_model_ready(self.model_name)
            except Exception:
                model_ready = False
            
            return {
                "connected": True,
                "server_ready": server_ready,
                "server_live": server_live,
                "model_ready": model_ready
            }
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {
                "connected": False,
                "server_ready": False,
                "server_live": False,
                "model_ready": False
            }
    
    async def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata from Triton"""
        try:
            metadata = await self._client.get_model_metadata(self.model_name)
            
            # Convert protobuf to dict
            result = {
                "name": metadata.name,
                "platform": metadata.platform,
                "versions": list(metadata.versions),
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": list(inp.shape)
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": list(out.shape)
                    }
                    for out in metadata.outputs
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise