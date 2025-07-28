#!/usr/bin/env python3
"""
Test script for attention map visualization functionality
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append("src/")

def test_attention_visualization():
    """Test the attention visualization utilities"""
    try:
        from attention_viz import (
            process_attention_maps, 
            create_attention_overlay, 
            visualize_attention_maps,
            create_attention_heatmap
        )
        print("✓ Successfully imported attention visualization functions")
        
        # Create dummy attention maps for testing
        batch_size = 1
        num_heads = 8
        seq_len = 100  # Simulated sequence length (patches + special tokens)
        patch_start_idx = 5  # Start of patch tokens
        
        # Create dummy attention weights
        dummy_attention = torch.randn(batch_size, num_heads, seq_len, seq_len)
        dummy_attention = torch.softmax(dummy_attention, dim=-1)
        
        attention_maps = [dummy_attention]
        
        # Test processing
        processed_maps = process_attention_maps(
            attention_maps, 
            patch_start_idx,
            img_size=(518, 518),
            patch_size=14
        )
        print(f"✓ Processed {len(processed_maps)} attention maps")
        
        # Create dummy images for visualization
        dummy_images = [np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)]
        
        if processed_maps:
            # Test overlay creation
            overlay = create_attention_overlay(dummy_images[0], processed_maps[0])
            print(f"✓ Created attention overlay with shape {overlay.shape}")
            
            # Test visualization (save to tmp directory)
            os.makedirs("/tmp/attention_test", exist_ok=True)
            
            viz_paths = visualize_attention_maps(
                dummy_images, processed_maps, "/tmp/attention_test", "test"
            )
            print(f"✓ Created {len(viz_paths)} visualization files")
            
            # Test heatmap creation
            heatmap_path = create_attention_heatmap(
                processed_maps, 
                "/tmp/attention_test/heatmap.png",
                "Test Attention Heatmap"
            )
            if heatmap_path:
                print(f"✓ Created heatmap at {heatmap_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Attention visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_modifications():
    """Test the model modifications"""
    try:
        from streamvggt.layers.attention import Attention
        from streamvggt.layers.block import Block
        from streamvggt.models.aggregator import Aggregator
        from streamvggt.models.streamvggt import StreamVGGT, StreamVGGTOutput
        
        print("✓ Successfully imported modified model components")
        
        # Test attention layer with return_attention=True
        embed_dim = 512
        num_heads = 8
        attention_layer = Attention(dim=embed_dim, num_heads=num_heads, fused_attn=False)
        
        # Create dummy input
        batch_size = 1
        seq_len = 20
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Test normal forward
        output = attention_layer(x)
        print(f"✓ Normal attention forward: {output.shape}")
        
        # Test with attention extraction
        output, attn_weights = attention_layer(x, return_attention=True)
        print(f"✓ Attention extraction: output {output.shape}, weights {attn_weights.shape}")
        
        # Test Block layer
        block = Block(dim=embed_dim, num_heads=num_heads, fused_attn=False)
        
        # Test normal forward
        output = block(x)
        print(f"✓ Normal block forward: {output.shape}")
        
        # Test with attention extraction
        output, attn_weights = block(x, return_attention=True)
        print(f"✓ Block attention extraction: output {output.shape}, weights {attn_weights.shape}")
        
        # Test StreamVGGTOutput
        output_obj = StreamVGGTOutput(
            ress=[{}], 
            views=None, 
            attention_maps=[attn_weights], 
            patch_start_idx=5
        )
        print("✓ StreamVGGTOutput with attention maps created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model modification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test basic imports"""
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ Basic dependencies available")
        
        # Test scipy for attention visualization
        try:
            from scipy.ndimage import zoom
            print("✓ SciPy available for image processing")
        except ImportError:
            print("! SciPy not available - using fallback image processing")
            
        # Test matplotlib
        try:
            import matplotlib.pyplot as plt
            print("✓ Matplotlib available for visualization")
        except ImportError:
            print("! Matplotlib not available - attention visualizations will be skipped")
            
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Global Attention Map Visualization Implementation")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Basic Imports", test_imports),
        ("Model Modifications", test_model_modifications), 
        ("Attention Visualization", test_attention_visualization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print("-" * 20)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Attention visualization is ready to use.")
        print("\nTo use the feature:")
        print("1. Launch the Gradio demo: python demo_gradio.py")
        print("2. Upload images or video")
        print("3. Check 'Extract Global Attention Maps' checkbox")
        print("4. Click 'Reconstruct' to see both 3D reconstruction and attention maps")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    sys.exit(0 if all_passed else 1)