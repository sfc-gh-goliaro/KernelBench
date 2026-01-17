import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Model(nn.Module):
    """
    Hungarian Matcher (Object Detection)

    Used by: DETR, Deformable DETR, DINO

    Performs optimal bipartite matching between predicted and ground truth
    objects using the Hungarian algorithm. Matches predictions to targets
    based on classification and box regression costs.

    Shapes:
        pred_logits: (batch, num_queries, num_classes)
        pred_boxes: (batch, num_queries, 4) normalized cxcywh
        target_labels: List of (num_targets,) per image
        target_boxes: List of (num_targets, 4) per image
        Output: List of (pred_indices, target_indices) per image
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        """
        Initialize Hungarian matcher.

        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 bounding box cost
            cost_giou: Weight for GIoU cost
        """
        super(Model, self).__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def box_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from center format to corner format."""
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

    def generalized_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute generalized IoU between two sets of boxes."""
        # Convert to xyxy
        boxes1 = self.box_cxcywh_to_xyxy(boxes1)
        boxes2 = self.box_cxcywh_to_xyxy(boxes2)

        # Intersection
        inter_x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0].unsqueeze(0))
        inter_y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1].unsqueeze(0))
        inter_x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2].unsqueeze(0))
        inter_y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3].unsqueeze(0))

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

        iou = inter_area / (union + 1e-6)

        # Enclosing box
        enc_x1 = torch.min(boxes1[:, 0:1], boxes2[:, 0].unsqueeze(0))
        enc_y1 = torch.min(boxes1[:, 1:2], boxes2[:, 1].unsqueeze(0))
        enc_x2 = torch.max(boxes1[:, 2:3], boxes2[:, 2].unsqueeze(0))
        enc_y2 = torch.max(boxes1[:, 3:4], boxes2[:, 3].unsqueeze(0))

        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

        giou = iou - (enc_area - union) / (enc_area + 1e-6)

        return giou

    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                target_labels: List[torch.Tensor],
                target_boxes: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.

        Args:
            pred_logits: Predicted class logits (batch, num_queries, num_classes)
            pred_boxes: Predicted boxes (batch, num_queries, 4)
            target_labels: Ground truth labels per image
            target_boxes: Ground truth boxes per image

        Returns:
            List of (pred_indices, target_indices) matchings per image
        """
        batch_size, num_queries, num_classes = pred_logits.shape
        device = pred_logits.device

        # Get probabilities
        pred_probs = pred_logits.softmax(-1)

        indices = []

        for b in range(batch_size):
            num_targets = target_labels[b].shape[0]

            if num_targets == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device)
                ))
                continue

            # Classification cost: -prob of correct class
            cost_class = -pred_probs[b, :, target_labels[b]]  # (num_queries, num_targets)

            # L1 box cost
            cost_bbox = torch.cdist(pred_boxes[b], target_boxes[b], p=1)

            # GIoU cost
            cost_giou = -self.generalized_iou(pred_boxes[b], target_boxes[b])

            # Total cost matrix
            cost_matrix = (self.cost_class * cost_class +
                          self.cost_bbox * cost_bbox +
                          self.cost_giou * cost_giou)

            # Hungarian matching (greedy approximation for GPU efficiency)
            # In practice, scipy.optimize.linear_sum_assignment is used on CPU
            cost_np = cost_matrix.detach()

            # Greedy matching (simplified)
            pred_idx = []
            tgt_idx = []
            used_preds = set()
            used_tgts = set()

            for _ in range(min(num_queries, num_targets)):
                # Find minimum cost assignment
                min_val = float('inf')
                min_p, min_t = -1, -1

                for p in range(num_queries):
                    if p in used_preds:
                        continue
                    for t in range(num_targets):
                        if t in used_tgts:
                            continue
                        if cost_np[p, t] < min_val:
                            min_val = cost_np[p, t]
                            min_p, min_t = p, t

                if min_p >= 0:
                    pred_idx.append(min_p)
                    tgt_idx.append(min_t)
                    used_preds.add(min_p)
                    used_tgts.add(min_t)

            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=device),
                torch.tensor(tgt_idx, dtype=torch.long, device=device)
            ))

        return indices


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
num_queries = 100
num_classes = 91  # COCO classes
num_targets = 10  # Average targets per image

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    pred_logits = torch.randn(batch_size, num_queries, num_classes, device='cuda')
    pred_boxes = torch.rand(batch_size, num_queries, 4, device='cuda')

    # Ground truth per image
    target_labels = [torch.randint(0, num_classes, (num_targets,), device='cuda')
                    for _ in range(batch_size)]
    target_boxes = [torch.rand(num_targets, 4, device='cuda')
                   for _ in range(batch_size)]

    return [pred_logits, pred_boxes, target_labels, target_boxes]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return []
