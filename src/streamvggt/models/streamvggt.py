import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from streamvggt.utils.kv_cache_manager import KVCacheManager
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, 
                 max_cached_frames=5, similarity_method="cosine", enable_kv_cache_optimization=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        
        # KV cache optimization
        self.enable_kv_cache_optimization = enable_kv_cache_optimization
        self.max_cached_frames = max_cached_frames
        self.similarity_method = similarity_method
        
        if self.enable_kv_cache_optimization:
            self.kv_cache_manager = KVCacheManager(
                max_cached_frames=max_cached_frames,
                similarity_method=similarity_method
            )
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):        
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        
        all_ress = []
        processed_frames = []

        # Reset KV cache manager for new sequence
        if self.enable_kv_cache_optimization:
            self.kv_cache_manager.reset()

        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0) 
            
            # Get initial aggregator output to extract pointmap
            initial_aggregator_output = self.aggregator(
                images, 
                past_key_values=[None] * self.aggregator.depth,  # Temporary pass without cache
                use_cache=False, 
                past_frame_idx=i
            )
            
            if isinstance(initial_aggregator_output, tuple) and len(initial_aggregator_output) >= 2:
                initial_aggregated_tokens, patch_start_idx = initial_aggregator_output[:2]
            else:
                initial_aggregated_tokens, patch_start_idx = initial_aggregator_output
            
            # Get pointmap and confidence for KV cache optimization
            current_pointmap = None
            current_conf = None
            
            if self.enable_kv_cache_optimization and self.point_head is not None:
                with torch.cuda.amp.autocast(enabled=False):
                    pts3d, pts3d_conf = self.point_head(
                        initial_aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    current_pointmap = pts3d[:, 0]  # Remove sequence dimension
                    current_conf = pts3d_conf[:, 0]
            
            # Update KV cache based on pointmap similarity
            should_update_cache = False
            frames_to_keep = list(range(i + 1))  # Default: keep all frames
            
            if self.enable_kv_cache_optimization and current_pointmap is not None:
                should_update_cache, frames_to_keep = self.kv_cache_manager.should_update_cache(
                    i, current_pointmap, current_conf
                )
                
                # Filter past KV cache if needed
                if should_update_cache and i > 0:
                    past_key_values = self.kv_cache_manager.filter_kv_cache(
                        past_key_values, frames_to_keep, i
                    )
            
            # Run aggregator with (potentially filtered) cache
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=len(frames_to_keep) - 1 if self.enable_kv_cache_optimization else i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    if self.enable_kv_cache_optimization and current_pointmap is not None:
                        # Reuse already computed pointmap to avoid redundant computation
                        pts3d = current_pointmap.unsqueeze(1)  # Add back sequence dimension
                        pts3d_conf = current_conf.unsqueeze(1)
                    else:
                        pts3d, pts3d_conf = self.point_head(
                            aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                        )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            })
            processed_frames.append(frame)
        
        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output