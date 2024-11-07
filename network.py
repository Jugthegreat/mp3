import torch
import math
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5  # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine
    )



class Anchors(nn.Module):
	
    def __init__(
        self,
        stride,
        scales=[4, 4 * math.pow(2, 1 / 3), 4 * math.pow(2, 2 / 3)],
        aspect_ratios=[0.5, 1, 2],
    ):
        """
        Args:
            stride: The stride of the feature map relative to the input image
            scales: List of anchor scales (sqrt of area), multiplied by the stride
            aspect_ratios: List of anchor aspect ratios (height/width)
        """
        super(Anchors, self).__init__()
        self.stride = stride
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_deltas = []

        # Precompute the width and height offsets for each anchor size and aspect ratio
        """Here we use 2 for loops to iterate over each scales and also for each scales it 
        iterates over each aspect ratio and calculate anchor width and height. Hence we will 
        have 9 values"""
        for scale in scales:
            for ar in aspect_ratios:
                width_of_anchor = scale * stride * math.sqrt(1 / ar)
                height_of_anchor = scale * stride * math.sqrt(ar)
                self.anchor_deltas.append(
                    [-width_of_anchor / 2, -height_of_anchor / 2, width_of_anchor / 2, height_of_anchor / 2]
                )
                """Then we convert anchor height and width to xmin xmax ymin and ymax and stire it as 4 corners of
                our anchor"""

        # Convert to a tensor and make the shape (9, 4) for convenience
        self.anchor_deltas = torch.tensor(self.anchor_deltas).float()

    def forward(self, feature_map):
        """
        Args:
            feature_map: Tensor of shape (B, C, H, W), where B is batch size, 
                         C is the number of channels, H and W are height and width.
        Returns:
            anchors: Tensor of anchor boxes in the format (x1, y1, x2, y2) 
                     with shape (B, A*4, H, W) where A is the number of anchor types.
        """
        """Extract shape of features liek Batch size, number of channels and grid height and grid width"""
        B, C, grid_height, grid_width = feature_map.shape
        device = feature_map.device

        # Generate a grid of center coordinates (H * W, 2) in terms of the stride
        y_coordinates = torch.arange(grid_height, device=device) * self.stride
        x_coordinates = torch.arange(grid_width, device=device) * self.stride
        """Generates grid for each center"""
        y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        center_points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)  # Shape: (H * W, 2)

        # Adjust anchor_deltas for broadcasting with centers
        anchor_deltas = self.anchor_deltas.to(device).unsqueeze(0)  # Shape: (1, 9, 4)

        # Repeat centers to match the number of anchor types
        centers_repeated = center_points.unsqueeze(1).repeat(1, anchor_deltas.shape[1], 1)  # Shape: (H * W, 9, 2)

        # Compute the top-left and bottom-right corners of each anchor box
        anchors_combined = torch.cat([centers_repeated + anchor_deltas[:, :, :2], centers_repeated + anchor_deltas[:, :, 2:]], dim=-1)

        # Reshape and reorder to match expected output shape (B, A*4, H, W)
        anchors_combined = anchors_combined.view(grid_height, grid_width, -1).permute(2, 0, 1).unsqueeze(0)
        anchors = anchors_combined.repeat(B, 1, 1, 1)

        return anchors





class RetinaNet(nn.Module):
    def __init__(self, p67=False, fpn=False,num_anchors=9):
        super(RetinaNet, self).__init__()
        self.resnet = [
            create_feature_extractor(
                resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
                return_nodes={
                    "layer2.3.relu_2": "conv3",
                    "layer3.5.relu_2": "conv4",
                    "layer4.2.relu_2": "conv5",
                },
            )
        ]
        self.resnet[0].eval()
        self.cls_head, self.bbox_head = self.get_heads(10, num_anchors)

        self.p67 = p67
        self.fpn = fpn

        anchors = nn.ModuleList()

        self.p5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            group_norm(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
        )
        self._init(self.p5)
        anchors.append(Anchors(stride=32))

        if self.p67:
            self.p6 = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p6)
            self.p7 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p7)
            anchors.append(Anchors(stride=64))
            anchors.append(Anchors(stride=128))

        if self.fpn:
            self.p4_lateral = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                group_norm(256),
            )
            self.p4 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p4)
            self._init(self.p4_lateral)
            anchors.append(Anchors(stride=16))

            self.p3_lateral = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), group_norm(256)
            )
            self.p3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p3)
            self._init(self.p3_lateral)
            anchors.append(Anchors(stride=8))

        self.anchors = anchors

    def _init(self, modules):
        for layer in modules.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def to(self, device):
        super(RetinaNet, self).to(device)
        self.anchors.to(device)
        self.resnet[0].to(device)
        return self

    def get_heads(self, num_classes, num_anchors, prior_prob=0.01):
        cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(
                256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            ),
        )
        bbox_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, padding=1),
        )

        # Initialization
        for modules in [cls_head, bbox_head]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(cls_head[-1].bias, bias_value)

        return cls_head, bbox_head

    def get_ps(self, feats):
        conv3, conv4, conv5 = feats["conv3"], feats["conv4"], feats["conv5"]
        p5 = self.p5(conv5)
        outs = [p5]

        if self.p67:
            p6 = self.p6(conv5)
            outs.append(p6)

            p7 = self.p7(p6)
            outs.append(p7)

        if self.fpn:
            p4 = self.p4(
                self.p4_lateral(conv4)
                + nn.Upsample(size=conv4.shape[-2:], mode="nearest")(p5)
            )
            outs.append(p4)

            p3 = self.p3(
                self.p3_lateral(conv3)
                + nn.Upsample(size=conv3.shape[-2:], mode="nearest")(p4)
            )
            outs.append(p3)
        # outs = [outs[:]]
        return outs

    def forward(self, x):
        with torch.no_grad():
            feats = self.resnet[0](x)

        feats = self.get_ps(feats)

        # apply the class head and box head on top of layers
        outs = []
        for f, a in zip(feats, self.anchors):
            cls = self.cls_head(f)
            bbox = self.bbox_head(f)
            outs.append((cls, bbox, a(f)))
        return outs

