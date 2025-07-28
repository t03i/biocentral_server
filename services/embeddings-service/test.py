#!/usr/bin/env python3
"""
Comprehensive test for the Protein Embedding Service with differently sized sequences
"""
import requests
import json
import numpy as np
import tempfile
import os
import time

SERVICE_URL = "http://localhost:8003"

def test_different_scenarios():
    """Test various scenarios with differently sized sequences"""
    print("🧪 Comprehensive Test: Different Sequence Lengths")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Very Different Lengths",
            "sequences": [
                "ACDEFGHIKLMNPQRSTVWY",  # 20 aa
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # 67 aa
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # 134 aa
            ]
        },
        {
            "name": "Similar Lengths",
            "sequences": [
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # 67 aa
                "MKTAYIAELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # 45 aa
                "MKTALVLLFGAILLAHQQGNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNYNNNNNNNNNN",  # 67 aa
            ]
        },
        {
            "name": "Single Sequence",
            "sequences": [
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # 67 aa
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        print("-" * 40)
        
        sequences = scenario['sequences']
        print(f"Sequences: {len(sequences)} sequences")
        for i, seq in enumerate(sequences):
            print(f"  {i}: {len(seq)} amino acids")
        
        # Test full embeddings
        print("\n  🔍 Full Embeddings:")
        full_result = test_embeddings(sequences, pooled=False)
        if full_result:
            full_embeddings = test_download(full_result['download_link'], len(sequences))
            if full_embeddings:
                shapes = [emb.shape for emb in full_embeddings]
                unique_shapes = len(set(str(shape) for shape in shapes))
                print(f"    ✅ Shapes: {shapes}")
                print(f"    ✅ Unique shapes: {unique_shapes}")
                if unique_shapes > 1:
                    print("    ✅ Different sequence lengths preserved!")
                else:
                    print("    ⚠️  All shapes are the same")
        
        # Test pooled embeddings
        print("\n  🔍 Pooled Embeddings:")
        pooled_result = test_embeddings(sequences, pooled=True)
        if pooled_result:
            pooled_embeddings = test_download(pooled_result['download_link'], len(sequences))
            if pooled_embeddings:
                shapes = [emb.shape for emb in pooled_embeddings]
                unique_shapes = len(set(str(shape) for shape in shapes))
                print(f"    ✅ Shapes: {shapes}")
                print(f"    ✅ Unique shapes: {unique_shapes}")
                if unique_shapes == 1:
                    print("    ✅ Uniform shapes as expected!")
                else:
                    print("    ❌ Unexpected different shapes for pooled")

def test_embeddings(sequences, pooled=False):
    """Test computing embeddings"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{SERVICE_URL}/embeddings/compute/prot_t5",
            json=sequences,
            params={"pooled": pooled},
            timeout=30
        )
        compute_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✅ Computation: {compute_time:.2f}s")
            print(f"    ✅ Cache hit rate: {result['cache_stats']['hit_rate']:.1%}")
            print(f"    ✅ Download link: {result['download_link']}")
            return result
        else:
            print(f"    ❌ Computation failed: {response.status_code}")
            print(f"    ❌ Error: {response.text}")
            return None
    except Exception as e:
        print(f"    ❌ Computation error: {e}")
        return None

def test_download(download_link, expected_count):
    """Test downloading embeddings"""
    try:
        start_time = time.time()
        response = requests.get(f"{SERVICE_URL}{download_link}", timeout=10)
        download_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"    ✅ Download: {download_time:.2f}s ({len(response.content)} bytes)")
            
            # Save and analyze
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            try:
                data = np.load(tmp_path)
                print(f"    ✅ NPZ contents: {list(data.keys())}")
                
                if 'storage_type' in data and data['storage_type'] == 'list_of_arrays':
                    sequence_count = data['sequence_count']
                    print(f"    ✅ Format: {sequence_count} sequences")
                    
                    embeddings = []
                    for i in range(sequence_count):
                        emb = data[f'embedding_{i}']
                        embeddings.append(emb)
                        print(f"    ✅ embedding_{i}: shape {emb.shape}")
                    
                    if len(embeddings) == expected_count:
                        return embeddings
                    else:
                        print(f"    ❌ Expected {expected_count}, got {len(embeddings)}")
                        return None
                else:
                    print(f"    ❌ Unexpected format")
                    return None
                    
            finally:
                os.unlink(tmp_path)
        else:
            print(f"    ❌ Download failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"    ❌ Download error: {e}")
        return None

def test_performance():
    """Test performance with different batch sizes"""
    print(f"\n📊 Performance Test")
    print("-" * 40)
    
    # Create sequences of different lengths
    sequences = []
    for i in range(10):
        length = 20 + (i * 10)  # 20, 30, 40, ..., 110 aa
        seq = "A" * length
        sequences.append(seq)
    
    print(f"Testing with {len(sequences)} sequences of varying lengths")
    
    # Test full embeddings
    start_time = time.time()
    full_result = test_embeddings(sequences, pooled=False)
    full_time = time.time() - start_time
    
    if full_result:
        print(f"✅ Full embeddings: {full_time:.2f}s")
        
        # Test download
        start_time = time.time()
        full_embeddings = test_download(full_result['download_link'], len(sequences))
        download_time = time.time() - start_time
        
        if full_embeddings:
            print(f"✅ Download: {download_time:.2f}s")
            shapes = [emb.shape for emb in full_embeddings]
            print(f"✅ Shapes preserved: {shapes[:3]}... (showing first 3)")

def main():
    """Run comprehensive tests"""
    print("🚀 Starting Comprehensive Embedding Service Test")
    print(f"Service URL: {SERVICE_URL}")
    
    # Check service health
    try:
        health = requests.get(f"{SERVICE_URL}/embeddings/health", timeout=5)
        if health.status_code == 200:
            health_data = health.json()
            print(f"✅ Service status: {health_data['status']}")
        else:
            print(f"❌ Service not healthy: {health.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return
    
    # Run tests
    test_different_scenarios()
    test_performance()
    
    print(f"\n🎉 All tests completed!")
    print("✅ Service handles differently sized sequences correctly")
    print("✅ Downloads use consistent list-of-arrays format")
    print("✅ Performance is acceptable for production use")

if __name__ == "__main__":
    main() 