"""BEVSegmentationHead — grid-transform BEV then classifier -> per-class logits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVGridTransform(nn.Module):
    """Resample the BEV feature from the backbone's grid/extent to the segmentation
    output grid/extent. The decoder BEV covers input_scope ([-51.2,51.2] step .8, 128x128)
    but the map task wants output_scope ([-50,50] step .5, 200x200) -- a different extent
    AND resolution. grid_sample handles both with one bilinear resampling."""

    def __init__(self, input_scope, output_scope, prescale_factor=1):
        super().__init__()
        self.input_scope = input_scope       # [(min,max,step)] per axis, the SOURCE grid
        self.output_scope = output_scope     # [(min,max,step)] per axis, the TARGET grid
        self.prescale_factor = prescale_factor

    def forward(self, x):
        if self.prescale_factor != 1:
            x = F.interpolate(x, scale_factor=self.prescale_factor,
                              mode="bilinear", align_corners=False)
        coords = []
        # for each axis, build the target cell-center positions, expressed in the SOURCE's
        # normalized [-1,1] coords (what grid_sample expects)
        for (imin, imax, _), (omin, omax, ostep) in zip(self.input_scope, self.output_scope):
            v = torch.arange(omin + ostep / 2, omax, ostep, device=x.device)   # target metric centers
            v = (v - imin) / (imax - imin) * 2 - 1                              # -> source [-1,1]
            coords.append(v)
        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)               # (Hout, Wout, 2) sampling locations
        grid = torch.stack([grid] * x.shape[0], dim=0)   # add batch -> (B, Hout, Wout, 2)
        return F.grid_sample(x, grid, mode="bilinear", align_corners=False)    # (B, C, Hout, Wout)


class BEVSegmentationHead(nn.Module):
    """Map segmentation head: resample BEV to the output grid, then a small conv classifier
    emits one logit per map class PER CELL. sigmoid (NOT softmax) because map classes are
    independent and can overlap (a cell can be both drivable_area and divider)."""

    def __init__(self, in_channels, grid_transform, classes):
        super().__init__()
        self.classes = classes
        self.transform = BEVGridTransform(**grid_transform)
        # classifier — two 3×3 convs + a 1×1 conv → 6 logits per cell
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),   # 512->512
            nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),   # 512->512
            nn.BatchNorm2d(in_channels), nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),                         # 512->6 logits
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]                          # take the shared (B,512,180,180) BEV feature
        x = self.transform(x)                # resample to (B,512,200,200) output grid
        x = self.classifier(x)               # per-cell logits (B,6,200,200)
        return torch.sigmoid(x)              # independent per-class probabilities
