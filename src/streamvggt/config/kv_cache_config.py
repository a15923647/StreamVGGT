"""
Configuration parameters for StreamVGGT KV cache optimization.
"""

# KV Cache Optimization Settings
KV_CACHE_CONFIG = {
    # Enable/disable KV cache optimization
    "enable_kv_cache_optimization": True,
    
    # Maximum number of frames to keep in KV cache (including first frame)
    # Recommended: 3-7 frames for good balance of quality and memory efficiency
    "max_cached_frames": 5,
    
    # Method for computing pointmap similarity
    # Options: "cosine", "l2", "weighted_cosine"
    "similarity_method": "cosine",
    
    # Whether to use confidence weighting in similarity calculation
    "use_confidence_weighting": True,
    
    # Minimum similarity threshold for frame selection
    # Frames below this threshold may be filtered out more aggressively
    "min_similarity_threshold": 0.0,
}

# Memory optimization settings
MEMORY_CONFIG = {
    # Enable gradient checkpointing to save memory
    "gradient_checkpointing": True,
    
    # Use mixed precision training/inference
    "use_mixed_precision": True,
    
    # Clear cache between frames (may slow down inference)
    "clear_cache_between_frames": False,
}

# Default model parameters with optimization
DEFAULT_MODEL_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    **KV_CACHE_CONFIG
}