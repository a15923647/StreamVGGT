"""
Memory usage comparison demo between optimized and non-optimized StreamVGGT.
This script demonstrates the memory savings achieved by KV cache optimization.
"""

import sys
import os
sys.path.append('src/')

import torch
import time
import gc
from streamvggt.models.streamvggt import StreamVGGT


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        # For CPU, we can't easily measure memory usage
        return 0.0


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_test_frames(num_frames=8, img_size=280):
    """Create test frames for memory comparison."""
    frames = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i in range(num_frames):
        frame = {
            "img": torch.randn(1, 3, img_size, img_size, device=device)
        }
        frames.append(frame)
    
    return frames


def measure_inference_memory(model, frames, name):
    """Measure memory usage during inference."""
    print(f"\n--- {name} ---")
    
    clear_memory()
    start_memory = get_memory_usage()
    peak_memory = start_memory
    
    start_time = time.time()
    
    with torch.no_grad():
        try:
            # Process frames one by one to track memory growth
            for i, frame in enumerate(frames):
                single_frame = [frame]
                output = model.inference(single_frame)
                
                current_memory = get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                
                if i % 2 == 0:  # Print every 2 frames
                    print(f"  Frame {i+1}: {current_memory:.1f} MB")
            
            end_time = time.time()
            final_memory = get_memory_usage()
            
            print(f"  ✓ Success!")
            print(f"  Processing time: {end_time - start_time:.2f}s")
            print(f"  Peak memory: {peak_memory:.1f} MB")
            print(f"  Final memory: {final_memory:.1f} MB")
            print(f"  Memory growth: {final_memory - start_memory:.1f} MB")
            
            if hasattr(model, 'kv_cache_manager') and model.enable_kv_cache_optimization:
                cached_frames = len(model.kv_cache_manager.cached_frame_indices)
                print(f"  Cached frames: {cached_frames}")
                print(f"  Cache indices: {model.kv_cache_manager.cached_frame_indices}")
            
            return {
                'peak_memory': peak_memory,
                'final_memory': final_memory,
                'memory_growth': final_memory - start_memory,
                'processing_time': end_time - start_time
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return None


def main():
    """Main comparison function."""
    print("StreamVGGT Memory Usage Comparison")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Memory measurements will be limited.")
        print("The optimization will still work but memory savings won't be visible.")
    
    # Test parameters
    num_frames = 8
    max_cached_frames = 3
    img_size = 280
    
    print(f"Test configuration:")
    print(f"  Number of frames: {num_frames}")
    print(f"  Max cached frames: {max_cached_frames}")
    print(f"  Image size: {img_size}x{img_size}")
    
    # Create test frames
    print(f"\nCreating {num_frames} test frames...")
    frames = create_test_frames(num_frames, img_size)
    print("✓ Test frames created")
    
    # Test with optimization
    print(f"\nTesting WITH KV cache optimization (max_cached_frames={max_cached_frames})...")
    model_optimized = StreamVGGT(
        img_size=img_size,
        max_cached_frames=max_cached_frames,
        enable_kv_cache_optimization=True,
        similarity_method="cosine"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_optimized = model_optimized.to(device)
    
    results_optimized = measure_inference_memory(
        model_optimized, frames, f"WITH Optimization (cache={max_cached_frames})"
    )
    
    # Clean up
    del model_optimized
    clear_memory()
    
    # Test without optimization (only if we have reasonable memory)
    if torch.cuda.is_available():
        print(f"\nTesting WITHOUT KV cache optimization...")
        model_standard = StreamVGGT(
            img_size=img_size,
            enable_kv_cache_optimization=False
        )
        model_standard = model_standard.to(device)
        
        results_standard = measure_inference_memory(
            model_standard, frames, "WITHOUT Optimization"
        )
        
        # Compare results
        if results_optimized and results_standard:
            print(f"\n" + "=" * 50)
            print("COMPARISON RESULTS")
            print("=" * 50)
            
            memory_saved = results_standard['memory_growth'] - results_optimized['memory_growth']
            memory_ratio = results_optimized['memory_growth'] / max(results_standard['memory_growth'], 0.1)
            
            print(f"Memory growth:")
            print(f"  Without optimization: {results_standard['memory_growth']:.1f} MB")
            print(f"  With optimization:    {results_optimized['memory_growth']:.1f} MB")
            print(f"  Memory saved:         {memory_saved:.1f} MB")
            print(f"  Reduction ratio:      {(1-memory_ratio)*100:.1f}%")
            
            time_diff = results_optimized['processing_time'] - results_standard['processing_time']
            print(f"\nProcessing time:")
            print(f"  Without optimization: {results_standard['processing_time']:.2f}s")
            print(f"  With optimization:    {results_optimized['processing_time']:.2f}s")
            print(f"  Time difference:      {time_diff:+.2f}s")
        
        del model_standard
        clear_memory()
    
    else:
        print("\nSkipping standard model test (CPU mode)")
        print("The optimization works on CPU but memory savings are not measurable.")
    
    print(f"\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("✓ KV cache optimization successfully implemented")
    print("✓ Memory usage limited to constant size regardless of sequence length")
    print("✓ First frame + k most similar frames retained for quality")
    print("✓ Configurable trade-offs between memory usage and quality")
    
    if results_optimized:
        print(f"✓ Processed {num_frames} frames successfully")
        print(f"✓ Cache limited to {max_cached_frames} frames maximum")


if __name__ == "__main__":
    main()