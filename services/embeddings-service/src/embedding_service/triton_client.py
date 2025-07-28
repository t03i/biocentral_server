"""
Triton gRPC client for ProtT5 embeddings
"""
import asyncio
import logging
from typing import List, Dict, Optional
import numpy as np

import tritonclient.grpc.aio as triton_grpc
from tritonclient.grpc import service_pb2

logger = logging.getLogger(__name__)

class TritonEmbeddingClient:
    """
    Async Triton gRPC client optimized for ProtT5 embeddings
    """
    
    def __init__(
        self, 
        triton_url: str = "triton:8001",
        model_name: str = "prot_t5_pipeline",
        max_batch_size: int = 8,
        timeout: float = 30.0
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self._client: Optional[triton_grpc.InferenceServerClient] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
        
    async def connect(self):
        """Connect to Triton server"""
        try:
            self._client = triton_grpc.InferenceServerClient(url=self.triton_url)
            
            # Verify server is ready
            if not await self._client.is_server_ready():
                raise ConnectionError(f"Triton server at {self.triton_url} not ready")
                
            # Verify model is ready
            if not await self._client.is_model_ready(self.model_name):
                raise ConnectionError(f"Model {self.model_name} not ready")
                
            logger.info(f"Connected to Triton server at {self.triton_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from Triton server"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Triton server")
            
    async def get_model_metadata(self) -> Dict:
        """Get model metadata from Triton"""
        if not self._client:
            raise RuntimeError("Client not connected. Use async context manager or call connect()")
            
        try:
            metadata = await self._client.get_model_metadata(self.model_name)
            return {
                "name": metadata.name,
                "platform": metadata.platform,
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
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise
            
    async def compute_embeddings_batch(
        self, 
        sequences: List[str],
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Compute embeddings for a batch of sequences using ProtT5 pipeline
        
        Args:
            sequences: List of protein sequences
            batch_size: Override default batch size
            
        Returns:
            List of embedding arrays (1024-dim FP16)
        """
        if not self._client:
            raise RuntimeError("Client not connected. Use async context manager or call connect()")
            
        if not sequences:
            return []
            
        effective_batch_size = min(
            batch_size or self.max_batch_size,
            self.max_batch_size,
            len(sequences)
        )
        
        all_embeddings = []
        
        # Process sequences in batches
        for i in range(0, len(sequences), effective_batch_size):
            batch_sequences = sequences[i:i + effective_batch_size]
            
            try:
                batch_embeddings = await self._process_batch(batch_sequences)
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//effective_batch_size + 1}: {e}")
                raise
                
        return all_embeddings
        
    async def _process_batch(self, sequences: List[str]) -> List[np.ndarray]:
        """Process a single batch through Triton"""
        
        # Prepare input tensor for Triton ensemble
        sequences_array = np.array(sequences, dtype=object)
        
        # Create Triton input
        inputs = [
            triton_grpc.InferInput(
                "sequences", 
                sequences_array.shape, 
                "BYTES"
            )
        ]
        inputs[0].set_data_from_numpy(sequences_array)
        
        # Create Triton output request
        outputs = [
            triton_grpc.InferRequestedOutput("embeddings")
        ]
        
        # Make inference request
        try:
            response = await self._client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=self.timeout
            )
            
            # Extract embeddings (FP16 -> FP32 for consistency)
            embeddings_tensor = response.as_numpy("embeddings")
            
            # Convert to list of individual embeddings
            embeddings = [
                embeddings_tensor[i].astype(np.float32) 
                for i in range(len(sequences))
            ]
            
            logger.debug(f"Successfully processed batch of {len(sequences)} sequences")
            return embeddings
            
        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            raise
            
    async def compute_single_embedding(self, sequence: str) -> np.ndarray:
        """Compute embedding for a single sequence"""
        embeddings = await self.compute_embeddings_batch([sequence])
        return embeddings[0]
        
    async def health_check(self) -> Dict[str, bool]:
        """Check Triton server and model health"""
        if not self._client:
            return {"connected": False, "server_ready": False, "model_ready": False}
            
        try:
            server_ready = await self._client.is_server_ready()
            model_ready = await self._client.is_model_ready(self.model_name)
            
            return {
                "connected": True,
                "server_ready": server_ready,
                "model_ready": model_ready
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"connected": False, "server_ready": False, "model_ready": False}


class TritonClientPool:
    """
    Connection pool for Triton clients to handle concurrent requests
    """
    
    def __init__(
        self,
        triton_url: str = "triton:8001",
        model_name: str = "prot_t5_pipeline",
        pool_size: int = 4
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.pool_size = pool_size
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._initialized = False
        
    async def initialize(self):
        """Initialize the connection pool"""
        if self._initialized:
            return
            
        for _ in range(self.pool_size):
            client = TritonEmbeddingClient(
                triton_url=self.triton_url,
                model_name=self.model_name
            )
            await client.connect()
            await self._pool.put(client)
            
        self._initialized = True
        logger.info(f"Initialized Triton client pool with {self.pool_size} connections")
        
    async def get_client(self) -> TritonEmbeddingClient:
        """Get a client from the pool"""
        if not self._initialized:
            await self.initialize()
        return await self._pool.get()
        
    async def return_client(self, client: TritonEmbeddingClient):
        """Return a client to the pool"""
        await self._pool.put(client)
        
    async def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            client = await self._pool.get()
            await client.disconnect()
        logger.info("Closed all connections in Triton client pool")