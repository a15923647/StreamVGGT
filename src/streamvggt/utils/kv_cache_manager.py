"""
KV Cache Management for StreamVGGT to prevent GPU OOM.
Implements frame selection based on pointmap similarity.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def compute_pointmap_similarity(pointmap1: torch.Tensor, pointmap2: torch.Tensor, 
                              conf1: torch.Tensor = None, conf2: torch.Tensor = None,
                              similarity_method: str = "cosine") -> float:
    """
    Compute similarity between two pointmaps.
    
    Args:
        pointmap1: First pointmap tensor [H, W, 3]
        pointmap2: Second pointmap tensor [H, W, 3] 
        conf1: Confidence map for first pointmap [H, W]
        conf2: Confidence map for second pointmap [H, W]
        similarity_method: Method to use ("cosine", "l2", "weighted_cosine")
        
    Returns:
        Similarity score (higher = more similar)
    """
    # Handle batch dimension if present
    if pointmap1.dim() == 4:
        pointmap1 = pointmap1.squeeze(0)
    if pointmap2.dim() == 4:
        pointmap2 = pointmap2.squeeze(0)
    
    # Flatten pointmaps
    flat1 = pointmap1.view(-1, 3)  # [H*W, 3]
    flat2 = pointmap2.view(-1, 3)  # [H*W, 3]
    
    # Filter out invalid points (e.g., NaN, inf, or very large values)
    valid_mask1 = torch.isfinite(flat1).all(dim=1)
    valid_mask2 = torch.isfinite(flat2).all(dim=1)
    valid_mask = valid_mask1 & valid_mask2
    
    if valid_mask.sum() == 0:
        return 0.0
    
    flat1 = flat1[valid_mask]
    flat2 = flat2[valid_mask]
    
    # Apply confidence weighting if available
    if conf1 is not None and conf2 is not None:
        if conf1.dim() == 3:
            conf1 = conf1.squeeze(0)
        if conf2.dim() == 3:
            conf2 = conf2.squeeze(0)
        
        conf1_flat = conf1.view(-1)[valid_mask]
        conf2_flat = conf2.view(-1)[valid_mask]
        weights = (conf1_flat * conf2_flat).sqrt()
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = None
    else:
        weights = None
    
    if similarity_method == "cosine":
        # Flatten to vectors
        vec1 = flat1.view(-1)  # [H*W*3]
        vec2 = flat2.view(-1)  # [H*W*3]
        
        if weights is not None:
            # Apply weights by repeating for 3 channels
            weights_expanded = weights.unsqueeze(1).expand(-1, 3).reshape(-1)
            vec1 = vec1 * weights_expanded
            vec2 = vec2 * weights_expanded
        
        similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1)
        return similarity.item()
        
    elif similarity_method == "l2":
        # L2 distance (convert to similarity by negating and normalizing)
        diff = flat1 - flat2
        if weights is not None:
            diff = diff * weights.unsqueeze(1)
        
        l2_dist = torch.norm(diff, dim=1).mean()
        # Convert distance to similarity (higher = more similar)
        similarity = 1.0 / (1.0 + l2_dist.item())
        return similarity
        
    elif similarity_method == "weighted_cosine":
        # Point-wise cosine similarity
        norms1 = torch.norm(flat1, dim=1, keepdim=True)
        norms2 = torch.norm(flat2, dim=1, keepdim=True)
        
        # Avoid division by zero
        norms1 = torch.clamp(norms1, min=1e-8)
        norms2 = torch.clamp(norms2, min=1e-8)
        
        normalized1 = flat1 / norms1
        normalized2 = flat2 / norms2
        
        point_similarities = (normalized1 * normalized2).sum(dim=1)
        
        if weights is not None:
            similarity = (point_similarities * weights).sum() / weights.sum()
        else:
            similarity = point_similarities.mean()
        
        return similarity.item()
    
    else:
        raise ValueError(f"Unknown similarity method: {similarity_method}")


class KVCacheManager:
    """
    Manages KV cache to prevent GPU OOM by limiting to first frame + k most similar frames.
    """
    
    def __init__(self, max_cached_frames: int = 5, similarity_method: str = "cosine"):
        """
        Args:
            max_cached_frames: Maximum number of frames to keep in cache (including first frame)
            similarity_method: Method for computing pointmap similarity
        """
        self.max_cached_frames = max_cached_frames
        self.similarity_method = similarity_method
        self.frame_pointmaps = []  # Store pointmaps for each frame
        self.frame_confidences = []  # Store confidence maps for each frame
        self.cached_frame_indices = [0]  # Always include first frame (index 0)
        
    def should_update_cache(self, current_frame_idx: int, current_pointmap: torch.Tensor, 
                          current_conf: torch.Tensor = None) -> Tuple[bool, List[int]]:
        """
        Determine if cache should be updated and which frames to keep.
        
        Args:
            current_frame_idx: Index of current frame
            current_pointmap: Pointmap of current frame [B, H, W, 3] or [H, W, 3]
            current_conf: Confidence map of current frame [B, H, W] or [H, W]
            
        Returns:
            (should_update, frames_to_keep): Whether to update cache and which frame indices to keep
        """
        # Always keep first frame
        if current_frame_idx == 0:
            self.frame_pointmaps = [current_pointmap.clone().cpu()]
            if current_conf is not None:
                self.frame_confidences = [current_conf.clone().cpu()]
            self.cached_frame_indices = [0]
            return False, [0]
        
        # Store current frame's data
        self.frame_pointmaps.append(current_pointmap.clone().cpu())
        if current_conf is not None:
            self.frame_confidences.append(current_conf.clone().cpu())
        
        # If we haven't reached the limit, keep all frames
        if len(self.frame_pointmaps) <= self.max_cached_frames:
            self.cached_frame_indices = list(range(len(self.frame_pointmaps)))
            return False, self.cached_frame_indices
        
        # Compute similarities between current frame and all previous frames
        similarities = []
        current_pointmap_cpu = current_pointmap.cpu()
        current_conf_cpu = current_conf.cpu() if current_conf is not None else None
        
        for i, stored_pointmap in enumerate(self.frame_pointmaps[:-1]):  # Exclude current frame
            stored_conf = self.frame_confidences[i] if self.frame_confidences else None
            similarity = compute_pointmap_similarity(
                stored_pointmap, current_pointmap_cpu,
                stored_conf, current_conf_cpu,
                self.similarity_method
            )
            similarities.append((similarity, i))
        
        # Sort by similarity (highest first) and always keep frame 0
        similarities.sort(reverse=True)
        
        # Select top k-1 most similar frames (excluding current) + frame 0
        frames_to_keep = [0]  # Always keep first frame
        
        # Add k-1 most similar frames (excluding frame 0 to avoid duplicates)
        added_count = 1  # Already added frame 0
        for similarity, frame_idx in similarities:
            if frame_idx != 0 and added_count < self.max_cached_frames - 1:  # -1 for current frame
                frames_to_keep.append(frame_idx)
                added_count += 1
        
        # Add current frame
        frames_to_keep.append(current_frame_idx)
        frames_to_keep.sort()
        
        # Update cached indices
        self.cached_frame_indices = frames_to_keep
        
        # Return whether cache needs updating and which frames to keep
        return True, frames_to_keep
    
    def filter_kv_cache(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], 
                       frames_to_keep: List[int], current_frame_idx: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Filter KV cache to keep only specified frames.
        
        Args:
            past_key_values: List of (key, value) tuples for each layer
            frames_to_keep: Frame indices to keep in cache
            current_frame_idx: Current frame index
            
        Returns:
            Filtered KV cache
        """
        if past_key_values is None or len(past_key_values) == 0:
            return past_key_values
        
        filtered_kv = []
        
        for layer_kv in past_key_values:
            if layer_kv is None:
                filtered_kv.append(None)
                continue
                
            key, value = layer_kv
            if key is None or value is None:
                filtered_kv.append(None)
                continue
            
            # Key/Value shape: [batch, heads, frames, seq_len, head_dim]
            # We need to select frames based on frames_to_keep
            
            # Create mask for frames to keep
            total_frames = key.shape[2]
            if current_frame_idx >= total_frames:
                # If current frame index exceeds cache size, just return current cache
                filtered_kv.append((key, value))
                continue
            
            # Map frame indices to cache positions
            cache_indices = []
            for frame_idx in frames_to_keep:
                if frame_idx < total_frames:
                    cache_indices.append(frame_idx)
            
            if not cache_indices:
                filtered_kv.append(None)
                continue
            
            # Select frames
            selected_key = key[:, :, cache_indices, :, :]
            selected_value = value[:, :, cache_indices, :, :]
            
            filtered_kv.append((selected_key, selected_value))
        
        return filtered_kv
    
    def reset(self):
        """Reset the cache manager state."""
        self.frame_pointmaps = []
        self.frame_confidences = []
        self.cached_frame_indices = [0]