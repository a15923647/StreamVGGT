"""
Attention visualization utilities for StreamVGGT
"""

import torch
import numpy as np
import os
from typing import List, Tuple, Optional

# Don't import matplotlib globally - import it when needed

def process_attention_maps(attention_maps: List[torch.Tensor], 
                          patch_start_idx: int,
                          img_size: Tuple[int, int] = (518, 518),
                          patch_size: int = 14) -> List[np.ndarray]:
    """
    Process raw attention maps to extract meaningful attention patterns
    
    Args:
        attention_maps: List of attention weight tensors from global attention blocks
        patch_start_idx: Index where patch tokens start (after camera/register tokens)
        img_size: Original image size (H, W)
        patch_size: Size of each patch
        
    Returns:
        List of processed attention maps as numpy arrays
    """
    processed_maps = []
    
    H, W = img_size
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    for attn_weights in attention_maps:
        if attn_weights is None:
            continue
            
        # attn_weights shape: [B, num_heads, seq_len, seq_len]
        B, num_heads, seq_len, _ = attn_weights.shape
        
        # Extract patch-to-patch attention (ignore special tokens)
        patch_attn = attn_weights[:, :, patch_start_idx:, patch_start_idx:]
        
        # Average across heads and batch
        patch_attn = patch_attn.mean(dim=(0, 1))  # [num_patches, num_patches]
        
        # Reshape to spatial dimensions
        num_patches = patch_attn.shape[0]
        patches_per_frame = num_patches_h * num_patches_w
        
        if num_patches % patches_per_frame == 0:
            num_frames = num_patches // patches_per_frame
            
            # Reshape to [num_frames * num_patches_h * num_patches_w, num_frames * num_patches_h * num_patches_w]
            # Then aggregate cross-frame attention
            patch_attn_np = patch_attn.cpu().numpy()
            
            # For visualization, we'll show the average attention from each patch
            avg_attn = np.mean(patch_attn_np, axis=1)  # Average attention received by each patch
            
            # Reshape to spatial grid for each frame
            for frame_idx in range(num_frames):
                start_idx = frame_idx * patches_per_frame
                end_idx = (frame_idx + 1) * patches_per_frame
                
                frame_attn = avg_attn[start_idx:end_idx]
                spatial_attn = frame_attn.reshape(num_patches_h, num_patches_w)
                processed_maps.append(spatial_attn)
        
    return processed_maps


def create_attention_overlay(image: np.ndarray, 
                           attention_map: np.ndarray, 
                           alpha: float = 0.6,
                           colormap: str = 'jet') -> np.ndarray:
    """
    Create an overlay of attention map on the original image
    
    Args:
        image: Original image as numpy array [H, W, 3], values in [0, 255]
        attention_map: Attention map as numpy array [H_patches, W_patches]
        alpha: Transparency of attention overlay
        colormap: Matplotlib colormap name
        
    Returns:
        Overlaid image as numpy array [H, W, 3]
    """
    # Normalize attention map
    attn_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Resize attention map to match image size
    try:
        from scipy.ndimage import zoom
        H, W = image.shape[:2]
        H_attn, W_attn = attn_norm.shape
        
        zoom_factors = (H / H_attn, W / W_attn)
        attn_resized = zoom(attn_norm, zoom_factors, order=1)
    except ImportError:
        # Fallback: simple upsampling without scipy
        H, W = image.shape[:2]
        H_attn, W_attn = attn_norm.shape
        
        # Simple nearest neighbor upsampling
        attn_resized = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                src_i = min(int(i * H_attn / H), H_attn - 1)
                src_j = min(int(j * W_attn / W), W_attn - 1)
                attn_resized[i, j] = attn_norm[src_i, src_j]
    
    # Apply colormap
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        attn_colored = cmap(attn_resized)[:, :, :3]  # Remove alpha channel
        attn_colored = (attn_colored * 255).astype(np.uint8)
    except ImportError:
        # Fallback: simple red overlay
        attn_colored = np.zeros_like(image)
        attn_colored[:, :, 0] = (attn_resized * 255).astype(np.uint8)
    
    # Blend with original image
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    blended = (1 - alpha) * image + alpha * attn_colored
    return blended.astype(np.uint8)


def visualize_attention_maps(images: List[np.ndarray],
                           attention_maps: List[np.ndarray],
                           save_dir: str,
                           prefix: str = "attention") -> List[str]:
    """
    Create visualization plots for attention maps
    
    Args:
        images: List of input images as numpy arrays
        attention_maps: List of processed attention maps
        save_dir: Directory to save visualization plots
        prefix: Prefix for saved file names
        
    Returns:
        List of saved file paths
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return []
        
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    
    num_images = len(images)
    num_maps = len(attention_maps)
    
    # Create a grid visualization
    for i in range(min(num_images, num_maps)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(images[i])
        axes[0].set_title(f'Original Image {i+1}')
        axes[0].axis('off')
        
        # Attention map
        im1 = axes[1].imshow(attention_maps[i], cmap='jet')
        axes[1].set_title(f'Global Attention Map {i+1}')
        axes[1].axis('off')
        try:
            plt.colorbar(im1, ax=axes[1])
        except:
            pass
        
        # Overlay
        overlay = create_attention_overlay(images[i], attention_maps[i])
        axes[2].imshow(overlay)
        axes[2].set_title(f'Attention Overlay {i+1}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(save_dir, f"{prefix}_frame_{i+1}.png")
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_paths.append(save_path)
        except Exception as e:
            print(f"Warning: Failed to save {save_path}: {e}")
        plt.close()
    
    # Create summary plot with all attention maps
    if len(attention_maps) > 1:
        try:
            n_cols = min(4, len(attention_maps))
            n_rows = (len(attention_maps) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, attn_map in enumerate(attention_maps):
                row = i // n_cols
                col = i % n_cols
                
                im = axes[row, col].imshow(attn_map, cmap='jet')
                axes[row, col].set_title(f'Frame {i+1}')
                axes[row, col].axis('off')
                try:
                    plt.colorbar(im, ax=axes[row, col])
                except:
                    pass
            
            # Hide unused subplots
            for i in range(len(attention_maps), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            summary_path = os.path.join(save_dir, f"{prefix}_summary.png")
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(summary_path)
        except Exception as e:
            print(f"Warning: Failed to create summary plot: {e}")
    
    return saved_paths


def create_attention_heatmap(attention_maps: List[np.ndarray], 
                           save_path: str,
                           title: str = "Global Attention Heatmaps") -> str:
    """
    Create a single heatmap visualization of attention maps
    
    Args:
        attention_maps: List of attention maps
        save_path: Path to save the heatmap
        title: Title for the plot
        
    Returns:
        Path to saved file
    """
    if not attention_maps:
        return None
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping heatmap creation")
        return None
        
    # Stack attention maps for visualization
    if len(attention_maps) == 1:
        combined_attn = attention_maps[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(combined_attn, cmap='jet', aspect='auto')
        ax.set_title(title)
        try:
            plt.colorbar(im, ax=ax)
        except:
            pass
        ax.axis('off')
    else:
        # Create temporal-spatial heatmap
        max_h = max(attn.shape[0] for attn in attention_maps)
        max_w = max(attn.shape[1] for attn in attention_maps)
        
        # Pad and stack
        padded_maps = []
        for attn in attention_maps:
            padded = np.zeros((max_h, max_w))
            h, w = attn.shape
            padded[:h, :w] = attn
            padded_maps.append(padded)
        
        combined_attn = np.stack(padded_maps, axis=0)  # [num_frames, H, W]
        
        # Show as time-series heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Flatten spatial dimensions for each frame
        flattened = combined_attn.reshape(len(attention_maps), -1)
        
        im = ax.imshow(flattened.T, cmap='jet', aspect='auto', origin='lower')
        ax.set_title(title)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Spatial Location (flattened)')
        try:
            plt.colorbar(im, ax=ax)
        except:
            pass
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    except Exception as e:
        print(f"Warning: Failed to save heatmap: {e}")
        plt.close()
        return None