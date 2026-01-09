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

num_boxes = 10000

def get_inputs():
    # Random boxes in xyxy format
    boxes = torch.rand(num_boxes, 4, device='cuda') * 640
    boxes[:, 2:] += boxes[:, :2]  # Ensure x2 > x1, y2 > y1
    scores = torch.rand(num_boxes, device='cuda')
    return [boxes, scores]

def get_init_inputs():
    return [0.45, 0.25]

