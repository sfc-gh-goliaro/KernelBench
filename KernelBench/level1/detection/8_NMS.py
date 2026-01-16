import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torchvision.ops as ops

class Model(nn.Module):
    """
    Non-Maximum Suppression (NMS)
    
    Used by: All detection models
    
    Filter overlapping boxes by IoU threshold, keep top scores.
    
    Shapes:
        boxes: (num_boxes, 4) in xyxy format
        scores: (num_boxes,)
        Output: indices of kept boxes
    """
    
    def __init__(self, iou_threshold: float = 0.45, score_threshold: float = 0.25):
        """
        Initialize NMS.
        
        Args:
            iou_threshold: IoU threshold for suppression
            score_threshold: Minimum score to keep
        """
        super(Model, self).__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
    
    def forward(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply NMS.
        
        Args:
            boxes: Box coordinates (num_boxes, 4) in xyxy format
            scores: Confidence scores (num_boxes,)
            
        Returns:
            Indices of kept boxes
        """
        # Filter by score threshold
        mask = scores > self.score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        
        if boxes.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # Apply NMS
        keep = ops.nms(boxes, scores, self.iou_threshold)
        
        return keep


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"num_boxes": 10000, "iou_threshold": 0.45, "score_threshold": 0.25},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("detection", "8_NMS")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    # Random boxes in xyxy format - use uniform_pos for coordinates
    boxes = DISTRIBUTIONS[dist_name]((p["num_boxes"], 4), dtype=dtype, device=device) * 640
    boxes[:, 2:] += boxes[:, :2]  # Ensure x2 > x1, y2 > y1
    scores = DISTRIBUTIONS[dist_name]((p["num_boxes"],), dtype=dtype, device=device)
    return [boxes, scores]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["iou_threshold"], p["score_threshold"]]
