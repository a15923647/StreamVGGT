# Global Attention Map Visualization

This document describes the global attention map visualization feature implemented for StreamVGGT.

## Overview

The StreamVGGT model uses transformer attention mechanisms with both frame-level and global attention blocks. This feature extracts and visualizes the **global attention maps** to help users understand where the model focuses when processing input frames.

## Features Implemented

### 1. Attention Extraction
- Modified `Attention` class to support returning attention weights with `return_attention=True`
- Updated `Block` class to propagate attention weights through transformer layers
- Enhanced `Aggregator` to capture global attention maps during processing
- Extended `StreamVGGT` model to support attention extraction during inference

### 2. Attention Processing
- Created `attention_viz.py` with comprehensive processing utilities
- Converts raw attention weights to spatial attention maps
- Handles multi-head attention aggregation
- Processes attention maps into frame-by-frame visualizations

### 3. Gradio Interface Integration
- Added "Extract Global Attention Maps" checkbox to enable/disable feature
- Integrated attention visualization gallery for per-frame attention maps
- Added attention heatmap display for summary visualization
- Maintains compatibility with existing 3D reconstruction workflow

## Usage

### In Gradio Demo

1. Launch the demo:
   ```bash
   python demo_gradio.py
   ```

2. Upload video or images as usual

3. ✅ **Check "Extract Global Attention Maps"** checkbox

4. Click "Reconstruct"

5. View results:
   - **3D Point Cloud + Camera Poses** (existing functionality)
   - **🔥 Global Attention Heatmaps** (NEW)
   - **🖼️ Per-frame Attention Overlays** (NEW)

### Programmatic Usage

```python
import torch
from streamvggt.models.streamvggt import StreamVGGT

# Load model
model = StreamVGGT()
model.eval()

# Prepare frames
frames = [{"img": image_tensor}]

# Extract attention maps
with torch.no_grad():
    output = model.inference(frames, extract_attention=True)

# Access attention maps
if hasattr(output, 'attention_maps'):
    attention_maps = output.attention_maps
    patch_start_idx = output.patch_start_idx
    
    # Process attention maps
    from attention_viz import process_attention_maps
    processed_maps = process_attention_maps(
        attention_maps, 
        patch_start_idx,
        img_size=(518, 518),
        patch_size=14
    )
```

## Technical Details

### Architecture Modifications

1. **Attention Layer (`src/streamvggt/layers/attention.py`)**
   - Added `return_attention` parameter to forward method
   - Returns attention weights when `fused_attn=False` and `return_attention=True`
   - Maintains backward compatibility

2. **Block Layer (`src/streamvggt/layers/block.py`)**
   - Propagates `return_attention` parameter through transformer blocks
   - Returns both output tokens and attention weights when requested

3. **Aggregator (`src/streamvggt/models/aggregator.py`)**
   - Added `extract_attention` parameter to forward method
   - Captures attention maps from global attention blocks
   - Returns attention maps alongside regular outputs

4. **StreamVGGT Model (`src/streamvggt/models/streamvggt.py`)**
   - Added `extract_attention` parameter to inference methods
   - Enhanced `StreamVGGTOutput` dataclass to include attention maps
   - Integrates attention extraction into model pipeline

### Attention Processing Pipeline

1. **Raw Attention Extraction**
   - Shape: `[batch_size, num_heads, seq_len, seq_len]`
   - Captured from global transformer attention blocks
   - Contains attention weights between all token pairs

2. **Spatial Processing**
   - Extracts patch-to-patch attention (ignores special tokens)
   - Averages across attention heads and batch dimension
   - Reshapes to spatial grid: `[num_patches_h, num_patches_w]`

3. **Visualization Generation**
   - Creates attention overlays on original images
   - Generates per-frame attention visualizations
   - Produces summary heatmaps for temporal analysis

## File Structure

```
StreamVGGT/
├── src/
│   ├── attention_viz.py                 # NEW: Attention visualization utilities
│   └── streamvggt/
│       ├── layers/
│       │   ├── attention.py            # MODIFIED: Added attention extraction
│       │   └── block.py                # MODIFIED: Added attention propagation
│       └── models/
│           ├── aggregator.py           # MODIFIED: Added global attention capture
│           └── streamvggt.py           # MODIFIED: Added extraction parameter
├── demo_gradio.py                      # MODIFIED: Added attention visualization UI
├── test_core.py                        # NEW: Core functionality tests
├── demo_attention_feature.py           # NEW: Feature demonstration
└── ATTENTION_VISUALIZATION.md          # NEW: This documentation
```

## API Reference

### Core Functions

#### `Attention.forward(..., return_attention=False)`
Extract attention weights from attention layer.

**Parameters:**
- `return_attention` (bool): Whether to return attention weights

**Returns:**
- When `return_attention=False`: `output_tokens`
- When `return_attention=True`: `(output_tokens, attention_weights)`

#### `process_attention_maps(attention_maps, patch_start_idx, img_size, patch_size)`
Process raw attention weights into spatial attention maps.

**Parameters:**
- `attention_maps` (List[torch.Tensor]): Raw attention weights
- `patch_start_idx` (int): Index where patch tokens start
- `img_size` (Tuple[int, int]): Original image size (H, W)
- `patch_size` (int): Size of each patch

**Returns:**
- `List[np.ndarray]`: Processed spatial attention maps

#### `visualize_attention_maps(images, attention_maps, save_dir, prefix)`
Create attention visualization plots.

**Parameters:**
- `images` (List[np.ndarray]): Input images
- `attention_maps` (List[np.ndarray]): Processed attention maps
- `save_dir` (str): Directory to save visualizations
- `prefix` (str): Prefix for saved files

**Returns:**
- `List[str]`: Paths to saved visualization files

## Configuration

### Default Settings
- **Image Size**: 518 × 518 pixels
- **Patch Size**: 14 × 14 pixels  
- **Spatial Grid**: 37 × 37 patches
- **Attention Heads**: 16 heads
- **Embedding Dimension**: 1024

### Customization
The attention visualization can be customized by modifying parameters in `attention_viz.py`:

```python
# Overlay transparency
alpha = 0.6  # Range: 0.0 (transparent) to 1.0 (opaque)

# Colormap for attention heatmaps
colormap = 'jet'  # Options: 'jet', 'hot', 'viridis', etc.

# Image processing method
# Default: scipy.ndimage.zoom (when available)
# Fallback: simple nearest neighbor upsampling
```

## Performance Considerations

- **Inference Speed**: Attention extraction adds minimal computational overhead
- **Memory Usage**: Storing attention weights increases memory requirements
- **Visualization**: Creating plots requires matplotlib and may take additional time
- **Recommendation**: Enable attention extraction only when needed for analysis

## Dependencies

### Required (Core Functionality)
- `torch >= 2.0`
- `numpy`

### Optional (Full Visualization)
- `matplotlib` (for plots and heatmaps)
- `scipy` (for high-quality image resizing)
- `gradio` (for web interface)

### Installation
```bash
pip install torch numpy matplotlib scipy gradio
```

## Troubleshooting

### Common Issues

1. **"No attention maps extracted"**
   - Ensure `extract_attention=True` is set
   - Check that model is using `fused_attn=False` for attention blocks

2. **"Visualization failed"**
   - Install matplotlib: `pip install matplotlib`
   - Check that processed attention maps are not empty

3. **"Import errors"**
   - Install missing dependencies
   - Some features gracefully degrade without optional dependencies

### Debug Tips

```python
# Check if attention extraction is working
output = model.inference(frames, extract_attention=True)
print(f"Has attention maps: {hasattr(output, 'attention_maps')}")
print(f"Number of attention maps: {len(output.attention_maps) if hasattr(output, 'attention_maps') else 0}")

# Verify attention weights shape
if hasattr(output, 'attention_maps') and output.attention_maps:
    attn = output.attention_maps[0]
    print(f"Attention shape: {attn.shape}")
    print(f"Attention range: [{attn.min():.6f}, {attn.max():.6f}]")
```

## Examples

See `demo_attention_feature.py` for a comprehensive demonstration of all features.

## Contributing

When modifying the attention visualization feature:

1. Maintain backward compatibility with existing functionality
2. Add appropriate error handling for missing dependencies
3. Update tests in `test_core.py`
4. Update this documentation

## License

This feature follows the same license as the main StreamVGGT project.