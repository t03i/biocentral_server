import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "pending"
    EMBEDDING = "embedding"
    PREDICTING = "predicting"
    COMPLETED = "completed"
    FAILED = "failed"

class PipelineJob(BaseModel):
    job_id: str
    sequences: List[str]
    models: List[str] = []
    embedder_name: str = "esm2_t33_650M_full"
    status: JobStatus = JobStatus.PENDING
    created_at: float
    completed_at: Optional[float] = None
    
    # Results
    embeddings: Optional[Dict[str, List[float]]] = None
    predictions: Optional[Dict[str, Any]] = None
    
    # Metadata
    error: Optional[str] = None
    timing: Dict[str, float] = {}
    cache_stats: Dict[str, Any] = {}

class QueueManager:
    """Manages Redis queues and job lifecycle"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        
        # Queue names
        self.EMBEDDING_QUEUE = "embedding_jobs"
        self.PREDICTION_QUEUE = "prediction_jobs"
        self.RESULT_UPDATES = "result_updates"
        
        # Storage keys
        self.JOB_STATUS_KEY = "job_status"
        self.JOB_RESULTS_KEY = "job_results"
        
        # Job TTL (1 week to match cache pattern)
        self.JOB_TTL = 7 * 24 * 3600
    
    async def submit_job(
        self,
        sequences: List[str],
        models: List[str] = None,
        embedder_name: str = "esm2_t33_650M_full"
    ) -> str:
        """Submit pipeline job and return job_id"""
        
        job_id = str(uuid.uuid4())
        job = PipelineJob(
            job_id=job_id,
            sequences=sequences,
            models=models or [],
            embedder_name=embedder_name,
            status=JobStatus.PENDING,
            created_at=time.time()
        )
        
        # Store job metadata
        await self.redis.hset(
            self.JOB_STATUS_KEY,
            job_id,
            job.json()
        )
        await self.redis.expire(self.JOB_STATUS_KEY, self.JOB_TTL)
        
        # Queue for embedding processing
        await self.redis.lpush(
            self.EMBEDDING_QUEUE,
            json.dumps({
                "job_id": job_id,
                "sequences": sequences,
                "embedder_name": embedder_name
            })
        )
        
        logger.info(f"Submitted job {job_id}: {len(sequences)} sequences, {len(models or [])} models")
        return job_id
    
    async def get_job(self, job_id: str) -> Optional[PipelineJob]:
        """Get current job status and results"""
        job_data = await self.redis.hget(self.JOB_STATUS_KEY, job_id)
        if not job_data:
            return None
        return PipelineJob.parse_raw(job_data)
    
    async def update_job(self, job: PipelineJob):
        """Update job in Redis"""
        await self.redis.hset(
            self.JOB_STATUS_KEY,
            job.job_id,
            job.json()
        )
    
    async def wait_for_completion(
        self,
        job_id: str,
        timeout: float = 300,
        poll_interval: float = 1.0
    ) -> PipelineJob:
        """Wait for job completion with polling"""
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            job = await self.get_job(job_id)
            
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status == JobStatus.COMPLETED:
                return job
            elif job.status == JobStatus.FAILED:
                raise Exception(f"Job failed: {job.error}")
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Job {job_id} timeout after {timeout}s")
    
    async def queue_for_predictions(self, job_id: str, embeddings: Dict[str, List[float]]):
        """Queue job for prediction processing"""
        job = await self.get_job(job_id)
        if not job:
            return
        
        job.embeddings = embeddings
        job.status = JobStatus.PREDICTING
        await self.update_job(job)
        
        # Queue for predictions if models requested
        if job.models:
            await self.redis.lpush(
                self.PREDICTION_QUEUE,
                json.dumps({
                    "job_id": job_id,
                    "embeddings": embeddings,
                    "models": job.models
                })
            )
    
    async def complete_job(
        self,
        job_id: str,
        predictions: Dict[str, Any] = None
    ):
        """Mark job as completed with results"""
        job = await self.get_job(job_id)
        if not job:
            return
        
        if predictions:
            job.predictions = predictions
        
        job.status = JobStatus.COMPLETED
        job.completed_at = time.time()
        
        await self.update_job(job)