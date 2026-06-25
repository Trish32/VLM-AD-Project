"""Image backbone (ResNet-50) and BEV encoder backbone (CustomResNet).

Pure-PyTorch / MPS reimplementation of FlashOcc (BEVDetOCC). No mmcv/mmdet3d.

The image backbone reuses ``torchvision.models.resnet50`` because the official
FlashOcc checkpoint stores the ResNet weights with *exactly* the torchvision
key layout (``conv1``, ``bn1``, ``layer{1..4}.{i}.conv{1,2,3}`` /
``downsample.{0,1}``).  We only need the stride-16 (layer3, 1024ch) and
stride-32 (layer4, 2048ch) feature maps because the config uses
``out_indices=(2, 3)``.

The BEV encoder backbone is a ``CustomResNet`` of ``BasicBlock`` stages that
mirrors ``projects/mmdet3d_plugin/models/backbones/resnet.py`` of the official
repo.  Its downsample branch is a *single* 3x3 conv (with bias), not the usual
1x1 conv + BN, so the block naming matches the checkpoint
(``layers.{s}.{b}.conv1/bn1/conv2/bn2`` and ``layers.{s}.0.downsample``).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNetImageBackbone(nn.Module):
    """ResNet-50 producing the stride-16 and stride-32 feature maps.

    Returns ``[C4 (B,1024,H/16,W/16), C5 (B,2048,H/32,W/32)]`` matching the
    config's ``out_indices=(2, 3)``.
    """

    def __init__(self):
        super().__init__()
        net = resnet50(weights=None)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        """Args: x (B*N, 3, 256, 704). Returns [C4, C5].
        All B×6 images run through ResNet-50 (cameras folded into the batch dim). 
        Stride doubles at each stage: stem 4x -> layer1 4x -> layer2 8x ->
        layer3 16x -> layer4 32x. 
        For a 256x704 input C4(stride-16, 1024ch) is 16x44 and C5(stride-32, 2049ch) 8x22.
        The view transformer's depth grid is tied to the C4 stride (downsample
        =16), so C4 is the feature actually splatted into BEV; C5 only adds
        global context to it via the neck.
        """
        x = self.relu(self.bn1(self.conv1(x)))  # 7x7 s2 conv  -> stride 2
        x = self.maxpool(x)                     # 3x3 s2 pool  -> stride 4
        x = self.layer1(x)                      # stride 4   (256 ch)
        x = self.layer2(x)                      # stride 8   (512 ch)
        c4 = self.layer3(x)     # (B, 1024, H/16, W/16)  stride 16
        c5 = self.layer4(c4)    # (B, 2048, H/32, W/32)  stride 32
        return [c4, c5]


class BasicBlock(nn.Module):
    """ResNet BasicBlock matching mmdet's naming (conv1/bn1/conv2/bn2).

    ``downsample`` (if present) is a single conv module supplied by the caller.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class CustomResNet(nn.Module):
    """BEV encoder backbone: 3 BasicBlock stages(100/50/25), each downsampling by 2.

    For flashocc-r50: numC_input=64, num_channels=[128, 256, 512],
    num_layer=[2, 2, 2], stride=[2, 2, 2].  Returns the output of each stage.
    """

    def __init__(self, numC_input, num_channels, num_layer=(2, 2, 2),
                 stride=(2, 2, 2)):
        super().__init__()
        layers = []
        curr = numC_input
        for i in range(len(num_layer)):
            # First block of the stage downsamples; its downsample branch is a
            # single 3x3 conv (with bias) -> matches checkpoint key layout.
            downsample = nn.Conv2d(curr, num_channels[i], 3, stride[i], 1)
            blocks = [BasicBlock(curr, num_channels[i], stride=stride[i],
                                 downsample=downsample)]
            curr = num_channels[i]
            for _ in range(num_layer[i] - 1):
                blocks.append(BasicBlock(curr, num_channels[i]))
            layers.append(nn.Sequential(*blocks))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Args: x (B, 64, 200, 200) lifted BEV feature.

        Builds a 3-level BEV feature pyramid by halving spatial size and
        doubling channels each stage. The neck (FPN_LSS) later fuses the
        finest (stage 0) and coarsest (stage 2) levels back to full 200x200.
        """
        feats = []
        for layer in self.layers:
            x = layer(x)            # each stage: stride 2, channels x2
            feats.append(x)
        return feats     # [(B,128,100,100), (B,256,50,50), (B,512,25,25)]
