#!/usr/bin/env python3
"""
Isolated testing script for the protein embedding service.
Tests both ProtT5 and ESM-2 models with comprehensive validation.
"""

import asyncio
import json
import httpx
import time
from typing import Dict, List

# Test configuration
EMBEDDING_SERVICE_URL = "http://localhost:8001"
TEST_SEQUENCES = [
    "MKLLPKRKETIDVELKEAVKSLKDKDLTDLKNDLTDLKDKDLTDLKNDLT",  # Test sequence 1
    "ACDEFGHIKLMNPQRSTVWY",  # Standard amino acids
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Longer sequence
]

class EmbeddingServiceTester:
    def __init__(self, base_url: str = EMBEDDING_SERVICE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    async def test_health_check(self) -> bool:
        """Test service health endpoint"""
        print("🔍 Testing health check...")
        try:
            response = await self.client.get(f"{self.base_url}/embeddings/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data['status']}")
                print(f"   Components: {health_data['components']}")
                return health_data["status"] == "healthy"
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
            
    async def test_model_info(self, model: str = "prot_t5") -> bool:
        """Test model info endpoint"""
        print(f"🔍 Testing model info for {model}...")
        try:
            response = await self.client.get(f"{self.base_url}/embeddings/model/info?model={model}")
            if response.status_code == 200:
                model_info = response.json()
                print(f"✅ Model info retrieved: {model_info.get('model_type', 'Unknown')}")
                print(f"   Embedding dim: {model_info.get('embedding_dim', 'Unknown')}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
            
    async def test_available_models(self) -> bool:
        """Test available models endpoint"""
        print("🔍 Testing available models...")
        try:
            response = await self.client.get(f"{self.base_url}/embeddings/models")
            if response.status_code == 200:
                models_data = response.json()
                available_models = models_data.get("available_models", {})
                print(f"✅ Available models: {list(available_models.keys())}")
                for model, info in available_models.items():
                    print(f"   {model}: {info['status']} ({info['embedding_dim']}D)")
                return True
            else:
                print(f"❌ Available models failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Available models error: {e}")
            return False
            
    async def test_embeddings(self, model: str, sequences: List[str]) -> bool:
        """Test embedding computation"""
        print(f"🔍 Testing embeddings for {model} with {len(sequences)} sequences...")
        
        payload = {
            "sequences": sequences,
            "model": model,
            "batch_size": 4
        }
        
        start_time = time.time()
        try:
            response = await self.client.post(
                f"{self.base_url}/embeddings/compute",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                end_time = time.time()
                
                embeddings = data["embeddings"]
                timing = data["timing"]
                model_info = data["model_info"]
                
                print(f"✅ Embeddings computed successfully in {end_time - start_time:.2f}s")
                print(f"   Model: {model_info['model']} ({model_info['embedding_dim']}D)")
                print(f"   Cache hits: {timing['cache_hits']}/{len(sequences)}")
                print(f"   Triton calls: {timing['triton_calls']}")
                print(f"   Total time: {timing['total_ms']:.1f}ms")
                print(f"   Cache time: {timing['cache_ms']:.1f}ms")
                print(f"   Triton time: {timing['triton_ms']:.1f}ms")
                
                # Validate embedding dimensions
                expected_dim = int(model_info['embedding_dim'])
                for i, (seq, emb) in enumerate(embeddings.items()):
                    if len(emb) != expected_dim:
                        print(f"❌ Wrong embedding dimension for sequence {i}: {len(emb)} vs {expected_dim}")
                        return False
                        
                print(f"   All embeddings have correct dimension: {expected_dim}")
                return True
                
            else:
                print(f"❌ Embeddings failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Embeddings error: {e}")
            return False
            
    async def test_cache_functionality(self, model: str) -> bool:
        """Test caching by running same sequences twice"""
        print(f"🔍 Testing cache functionality for {model}...")
        
        test_seq = ["MKLLPKRKETIDVELKEAVK"]
        payload = {
            "sequences": test_seq,
            "model": model,
            "batch_size": 1
        }
        
        # First request (should miss cache)
        response1 = await self.client.post(f"{self.base_url}/embeddings/compute", json=payload)
        if response1.status_code != 200:
            print(f"❌ First request failed: {response1.status_code}")
            return False
            
        data1 = response1.json()
        cache_hits1 = data1["timing"]["cache_hits"]
        
        # Small delay to ensure caching is complete
        await asyncio.sleep(0.5)
        
        # Second request (should hit cache)
        response2 = await self.client.post(f"{self.base_url}/embeddings/compute", json=payload)
        if response2.status_code != 200:
            print(f"❌ Second request failed: {response2.status_code}")
            return False
            
        data2 = response2.json()
        cache_hits2 = data2["timing"]["cache_hits"]
        
        if cache_hits2 > cache_hits1:
            print(f"✅ Cache working: hits increased from {cache_hits1} to {cache_hits2}")
            print(f"   Second request time: {data2['timing']['total_ms']:.1f}ms")
            return True
        else:
            print(f"❌ Cache not working: hits {cache_hits1} -> {cache_hits2}")
            return False
            
    async def test_cache_stats(self) -> bool:
        """Test cache statistics endpoint"""
        print("🔍 Testing cache statistics...")
        try:
            response = await self.client.get(f"{self.base_url}/embeddings/cache/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Cache stats retrieved:")
                print(f"   Redis connected: {stats['redis_info']['connected']}")
                print(f"   Hit rate: {stats['stats'].get('hit_rate', 0):.2%}")
                print(f"   Total hits: {stats['stats'].get('hits', 0)}")
                print(f"   Total misses: {stats['stats'].get('misses', 0)}")
                return True
            else:
                print(f"❌ Cache stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cache stats error: {e}")
            return False
            
    async def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Starting comprehensive embedding service test...\n")
        
        # Basic connectivity tests
        if not await self.test_health_check():
            print("❌ Health check failed - stopping tests")
            return False
            
        if not await self.test_available_models():
            print("❌ Models not available - stopping tests")
            return False
            
        # Test each available model
        models_to_test = ["prot_t5", "esm2_t33_650M", "esm2_t36_3B"]
        successful_models = []
        
        for model in models_to_test:
            print(f"\n📋 Testing model: {model}")
            
            # Test model info
            if not await self.test_model_info(model):
                print(f"⚠️  Model {model} info failed - skipping")
                continue
                
            # Test embeddings
            if not await self.test_embeddings(model, TEST_SEQUENCES):
                print(f"⚠️  Model {model} embeddings failed - skipping")
                continue
                
            # Test caching
            if not await self.test_cache_functionality(model):
                print(f"⚠️  Model {model} caching failed")
                
            successful_models.append(model)
            print(f"✅ Model {model} tests completed successfully")
            
        # Test cache stats
        print(f"\n📊 Testing cache functionality...")
        await self.test_cache_stats()
        
        # Summary
        print(f"\n🎯 Test Summary:")
        print(f"   Successful models: {successful_models}")
        print(f"   Failed models: {set(models_to_test) - set(successful_models)}")
        
        if len(successful_models) > 0:
            print("✅ Embedding service is working!")
            return True
        else:
            print("❌ No models working - check setup")
            return False

async def main():
    """Main test function"""
    async with EmbeddingServiceTester() as tester:
        success = await tester.run_comprehensive_test()
        return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        exit(1) 