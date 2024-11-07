import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor_boxes, class_labels, gt_boxes):
    """
    Args:
        anchor_boxes: Tensor of anchors in format (x1, y1, x2, y2); shape is (B, N_anchors, 4)
        class_labels: Ground truth class labels with shape (B, num_objects, 1)
        gt_boxes: Ground truth bounding boxes with shape (B, num_objects, 4)
    Returns:
        assigned_classes: Tensor of assigned class targets with shape (B, N_anchors, 1)
        assigned_bboxes: Tensor of assigned bbox targets with shape (B, N_anchors, 4)
    """
    # Extract batch size and the number of anchors per image
    batch_size, num_anchors, _ = anchor_boxes.shape
    device = anchor_boxes.device  # Ensuring all tensors are on the same device

    # Initialize tensors for storing target classes and bounding boxes for each anchor
    assigned_classes = torch.zeros((batch_size, num_anchors, 1), dtype=torch.int, device=device)
    assigned_bboxes = torch.zeros((batch_size, num_anchors, 4), device=device)

    # Process each image in the batch
    for img_idx in range(batch_size):  
        # Compute IoU between anchors and ground truth boxes for the current image
        iou_matrix = compute_bbox_iou(anchor_boxes[img_idx], gt_boxes[img_idx])

        # Find the max IoU and the corresponding ground truth box index for each anchor
        max_iou_vals, max_iou_indices = iou_matrix.max(dim=1)

        # Set targets based on IoU thresholds
        assigned_classes[img_idx][max_iou_vals < 0.4] = 0  # If IoU < 0.4, mark as background (class = 0)
        
        # If IoU is between 0.4 and 0.5, we ignore this anchor (set to -1)
        ignore_mask = (max_iou_vals >= 0.4) & (max_iou_vals < 0.5)
        assigned_classes[img_idx][ignore_mask] = -1  
        
        # Anchors with IoU >= 0.5 are considered as positive matches
        positive_mask = max_iou_vals >= 0.5  

        # Assign the class label to anchors that have IoU >= 0.5 with a ground truth box
        # We use .to(torch.int) to ensure the types match
        assigned_classes[img_idx][positive_mask] = class_labels[img_idx][max_iou_indices[positive_mask]].to(torch.int)

        # For positive anchors, we also assign the corresponding bounding box as the target
        assigned_bboxes[img_idx][positive_mask] = gt_boxes[img_idx][max_iou_indices[positive_mask]]

    return assigned_classes, assigned_bboxes




def compute_bbox_targets(anchor_boxes, target_boxes):
    """
    Args:
        anchor_boxes: Tensor of anchor boxes, shape (A, 4)
        target_boxes: Tensor of ground truth boxes, shape (A, 4)
    Returns:
        reg_targets: Tensor of calculated offsets, shape (A, 4)
    """
    """Step 1: Calculate anchor box widths and heights
    Width is the difference between x-coordinates (xmax - xmin)
    Height is the difference between y-coordinates (ymax - ymin)"""
    anchor_width = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_height = anchor_boxes[:, 3] - anchor_boxes[:, 1]

    """Step 2: Calculate the center of each anchor box
    Anchor center x-coordinate: start from xmin and add half of the width
    Anchor center y-coordinate: start from ymin and add half of the height"""
    anchor_center_x = anchor_boxes[:, 0] + 0.5 * anchor_width
    anchor_center_y = anchor_boxes[:, 1] + 0.5 * anchor_height

    """Step 3: Calculate ground-truth box widths and heights
    Width of target box: xmax - xmin
    Height of target box: ymax - ymin"""
    target_width = target_boxes[:, 2] - target_boxes[:, 0]
    target_height = target_boxes[:, 3] - target_boxes[:, 1]

    """Step 4: Calculate the center of each ground-truth box
    Target center x-coordinate, adding half of the width to xmin
    Target center y-coordinate, adding half of the height to ymin"""
    target_center_x = target_boxes[:, 0] + 0.5 * target_width
    target_center_y = target_boxes[:, 1] + 0.5 * target_height

    """Step 5: Clamp widths and heights to prevent division by zero
    This ensures stability when normalizing by these values"""
    anchor_width = torch.clamp(anchor_width, min=1)  # Minimum width of 1
    anchor_height = torch.clamp(anchor_height, min=1)  # Minimum height of 1
    target_width = torch.clamp(target_width, min=1)  # Minimum width for target
    target_height = torch.clamp(target_height, min=1)  # Minimum height for target

    """Step 6: Calculate x and y center offsets (deltas)
    delta_x: normalized horizontal offset from anchor center to target center
    delta_y: normalized vertical offset from anchor center to target center"""
    delta_x = (target_center_x - anchor_center_x) / anchor_width
    delta_y = (target_center_y - anchor_center_y) / anchor_height

    """Step 7: Calculate width and height scaling factors (deltas)
    delta_w: scale factor in log space for width adjustment
    log scale is used for stability in learning varying sizes
    delta_h: scale factor in log space for height adjustment"""
    delta_w = torch.log(target_width / anchor_width)
    delta_h = torch.log(target_height / anchor_height)

    return torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)


def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        bounding_boxes: (N, 4) tensor containing (xmin, ymin, xmax, ymax)
        offsets: (N, 4) tensor with (delta_x, delta_y, delta_log_w, delta_log_h)
    Returns:
        adjusted_boxes: (N, 4) tensor with the adjusted (xmin, ymin, xmax, ymax)
    """
    """Step 1: Calculate original widths, heights, and centers of the bounding boxes
    Width is the difference between xmax and xmin
    Height is the difference between ymax and ymin
    Center x-coordinate of each box, calculated from xmin + half width
    Center y-coordinate of each box, calculated from ymin + half height"""
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_heights = boxes[:, 3] - boxes[:, 1]
    box_center_x = boxes[:, 0] + 0.5 * box_widths
    box_center_y = boxes[:, 1] + 0.5 * box_heights

    """Step 2: Apply the predicted offsets (deltas) to adjust the centers and sizes
    New center x-coordinate: adjusting original center by delta_x * width
    New center y-coordinate: adjusting original center by delta_y * height
    Adjusted width: scaling original width by exp(delta_log_w)
    Adjusted height: scaling original height by exp(delta_log_h)"""
    new_center_x = box_center_x + deltas[:, 0] * box_widths
    new_center_y = box_center_y + deltas[:, 1] * box_heights 
    new_widths = box_widths * torch.exp(deltas[:, 2])
    new_heights = box_heights * torch.exp(deltas[:, 3])

    """Step 3: Convert the adjusted centers and sizes back to corner coordinates
    # Calculate new xmin (x1) by moving half of the new width to the left of the center
    # Calculate new ymin (y1) by moving half of the new height up from the center
    # Calculate new xmax (x2) by moving half of the new width to the right of the center
    # Calculate new ymax (y2) by moving half of the new height down from the center"""
    x1 = new_center_x - 0.5 * new_widths    
    y1 = new_center_y - 0.5 * new_heights    
    x2 = new_center_x + 0.5 * new_widths    
    y2 = new_center_y + 0.5 * new_heights

    """Step 4: Stack the corner coordinates to get the final adjusted bounding boxes
    Stack x1, y1, x2, y2 into a tensor with shape (N, 4), where each row represents a box"""
    new_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    
    # Return the adjusted bounding boxes with new (xmin, ymin, xmax, ymax) values
    return new_boxes


def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        boxes: (N, 4) tensor containing bounding boxes in (xmin, ymin, xmax, ymax) format
        confidence_scores: (N,) tensor of confidence scores for each bounding box
        iou_threshold: IoU threshold for deciding whether to suppress a bounding box
    Returns:
        selected_indices: (K,) tensor of indices for boxes that are kept
    """
    """Step 1: Sort scores in descending order and get the sorted indices
    This way, we start with the box having the highest confidence"""
    scores_sorted, sorted_indices = scores.sort(descending=True)
    selected_indices = []

    """Step 2: Perform NMS"""
    while sorted_indices.numel() > 0:  # While there are boxes to process
        # Take the index of the box with the highest score
        current_idx = sorted_indices[0].item()
        # Add this index to the list of kept boxes
        selected_indices.append(current_idx)

        # If only one box remains, our process is compelte
        if sorted_indices.numel() == 1:
            break

        """Step 3: Compute IoU between the selected box and the remaining boxes
        current_box is the box with the highest score, shape (1, 4)
        remaining_boxes are all other boxes, shape (N-1, 4)"""
        current_box = bboxes[current_idx].unsqueeze(0)
        remaining_boxes = bboxes[sorted_indices[1:]]

        "Calculate IoUs between the current box and each of the remaining boxes"
        ious = compute_bbox_iou(current_box, remaining_boxes).squeeze(0)

        """Step 4: Keep only boxes with IoU less than or equal to the threshold
        Find indices of boxes with IoU <= threshold"""
        below_threshold_indices = (ious <= threshold).nonzero(as_tuple=False).squeeze(1)
        
        """Update the sorted_indices list to only include boxes that passed the IoU filter"""
        sorted_indices = sorted_indices[1:][below_threshold_indices]

    # Convert selected_indices to a tensor and return
    return torch.tensor(selected_indices, dtype=torch.long)

