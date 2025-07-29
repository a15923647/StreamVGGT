"""
Test script for KV cache optimization in StreamVGGT.
Tests memory efficiency and functionality of the optimization.
"""

import sys
import os
sys.path.append('src/')

import torch
import torch.nn as nn
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.kv_cache_manager import KVCacheManager, compute_pointmap_similarity
import numpy as np
import time
import gc


def create_dummy_frame(batch_size=1, height=280, width=280):
    """Create a dummy frame for testing."""
    return {
        "img": torch.randn(batch_size, 3, height, width)
    }


def measure_memory_usage():
    """Measure current memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        return 0.0


def test_kv_cache_optimization():
    """Test KV cache optimization functionality."""
    print("Testing KV Cache Optimization...")
    
    # Test parameters
    num_frames = 5  # Reduce for memory efficiency
    max_cached_frames = 3
    img_size = 280  # Smaller compatible size
    
    print(f"Testing with {num_frames} frames, max cached = {max_cached_frames}")
    
    # Create models
    model_with_opt = StreamVGGT(
        img_size=img_size, 
        max_cached_frames=max_cached_frames,
        enable_kv_cache_optimization=True
    )
    
    model_without_opt = StreamVGGT(
        img_size=img_size,
        enable_kv_cache_optimization=False
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model_with_opt = model_with_opt.to(device)
        model_without_opt = model_without_opt.to(device)
    
    # Create test frames
    frames = []
    for i in range(num_frames):
        frame = create_dummy_frame()
        if device == "cuda":
            frame["img"] = frame["img"].to(device)
        frames.append(frame)
    
    print(f"Created {num_frames} test frames")
    
    # Test with optimization
    print("\n--- Testing WITH KV Cache Optimization ---")
    torch.cuda.empty_cache() if device == "cuda" else None
    gc.collect()
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    with torch.no_grad():
        try:
            output_with_opt = model_with_opt.inference(frames)
            end_time = time.time()
            end_memory = measure_memory_usage()
            
            print(f"✓ Success! Processed {num_frames} frames")
            print(f"  Time: {end_time - start_time:.2f}s")
            print(f"  Memory used: {end_memory - start_memory:.1f} MB")
            print(f"  Output frames: {len(output_with_opt.ress)}")
            
            # Check cache manager state
            if hasattr(model_with_opt, 'kv_cache_manager'):
                cached_frames = len(model_with_opt.kv_cache_manager.cached_frame_indices)
                print(f"  Cached frames: {cached_frames} (indices: {model_with_opt.kv_cache_manager.cached_frame_indices})")
            
        except Exception as e:
            print(f"✗ Failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test without optimization (if we have CUDA, otherwise skip for memory)
    if device == "cuda":
        print("\n--- Testing WITHOUT KV Cache Optimization ---")
        torch.cuda.empty_cache()
        gc.collect()
        
        start_memory = measure_memory_usage()
        start_time = time.time()
        
        with torch.no_grad():
            try:
                output_without_opt = model_without_opt.inference(frames)
                end_time = time.time()
                end_memory = measure_memory_usage()
                
                print(f"✓ Success! Processed {num_frames} frames")
                print(f"  Time: {end_time - start_time:.2f}s")
                print(f"  Memory used: {end_memory - start_memory:.1f} MB")
                print(f"  Output frames: {len(output_without_opt.ress)}")
                
            except Exception as e:
                print(f"✗ Failed with error: {e}")
    else:
        print("\n--- Skipping test without optimization (CPU mode) ---")


def test_pointmap_similarity():
    """Test pointmap similarity calculation."""
    print("\n\nTesting Pointmap Similarity Calculation...")
    
    # Create test pointmaps
    h, w = 64, 64
    
    # Identical pointmaps
    pointmap1 = torch.randn(h, w, 3)
    pointmap2 = pointmap1.clone()
    conf1 = torch.rand(h, w)
    conf2 = conf1.clone()
    
    similarity = compute_pointmap_similarity(pointmap1, pointmap2, conf1, conf2)
    print(f"Identical pointmaps similarity: {similarity:.4f} (should be ~1.0)")
    
    # Random pointmaps
    pointmap3 = torch.randn(h, w, 3)
    pointmap4 = torch.randn(h, w, 3)
    conf3 = torch.rand(h, w)
    conf4 = torch.rand(h, w)
    
    similarity = compute_pointmap_similarity(pointmap3, pointmap4, conf3, conf4)
    print(f"Random pointmaps similarity: {similarity:.4f} (should be ~0.0)")
    
    # Similar pointmaps (with noise)
    pointmap5 = pointmap1 + torch.randn_like(pointmap1) * 0.1
    similarity = compute_pointmap_similarity(pointmap1, pointmap5, conf1, conf1)
    print(f"Similar pointmaps similarity: {similarity:.4f} (should be >0.5)")


def test_cache_manager():
    """Test KV cache manager functionality."""
    print("\n\nTesting KV Cache Manager...")
    
    cache_manager = KVCacheManager(max_cached_frames=3)
    
    # Create test pointmaps for multiple frames
    pointmaps = []
    confs = []
    for i in range(5):
        # Create slightly different pointmaps
        base = torch.randn(32, 32, 3)
        pointmap = base + torch.randn_like(base) * 0.1 * i
        conf = torch.rand(32, 32)
        pointmaps.append(pointmap)
        confs.append(conf)
    
    print("Testing cache updates...")
    
    for i, (pointmap, conf) in enumerate(zip(pointmaps, confs)):
        should_update, frames_to_keep = cache_manager.should_update_cache(i, pointmap, conf)
        print(f"Frame {i}: should_update={should_update}, frames_to_keep={frames_to_keep}")
    
    print(f"Final cached frame indices: {cache_manager.cached_frame_indices}")


if __name__ == "__main__":
    print("StreamVGGT KV Cache Optimization Test")
    print("=" * 50)
    
    test_pointmap_similarity()
    test_cache_manager()
    test_kv_cache_optimization()
    
    print("\n" + "=" * 50)
    print("Test completed!")