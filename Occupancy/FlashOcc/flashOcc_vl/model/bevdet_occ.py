"""BEVDetOCC: the full flashocc-r50 model, pure-PyTorch / MPS.

Pipeline (single keyframe, 6 cameras):

    imgs (B,6,3,256,704)
      -> ResNet-50 backbone        -> [C4 (1024,16,44), C5 (2048,8,22)]
      -> CustomFPN neck            -> (B*6, 256, 16, 44)
      -> LSSViewTransformer        -> BEV (B, 64, 200, 200)
      -> CustomResNet BEV backbone -> [128@100, 256@50, 512@25]
      -> FPN_LSS BEV neck          -> (B, 256, 200, 200)
      -> BEVOCCHead2D              -> (B, 200, 200, 16, 18) class logits

The geometry that maps each camera frustum into the shared key-ego BEV frame
follows the official ``prepare_inputs`` / ``get_ego_coor``.
"""
import torch
import torch.nn as nn

from .backbones import ResNetImageBackbone, CustomResNet
from .necks import CustomFPN, FPN_LSS
from .view_transformer import LSSViewTransformer
from .occ_head import BEVOCCHead2D


class BEVDetOCC(nn.Module):
    def __init__(self, grid_config, input_size, numC_Trans=64,
                 num_classes=18, Dz=16):
        """
        Args:
            grid_config: BEV/depth grid spec, dict with keys x/y/z/depth, each
                [lower, upper, interval]. For flashocc-r50 this gives a
                200x200x16 voxel grid and 88 depth bins.
            input_size: (H, W) of the network input images, here (256, 704).
            numC_Trans: channel width of the lifted BEV feature (64).
            num_classes: occupancy classes incl. 'free' (18).
            Dz: number of height bins the 2D BEV head expands into (16).
        """
        super().__init__()
        # 2D image stream: per-camera CNN feature extractor + pyramid neck.
        self.img_backbone = ResNetImageBackbone()
        self.img_neck = CustomFPN(in_channels=(1024, 2048), out_channels=256,
                                  out_ids=(0,))
        # Lift-Splat-Shoot: projects 2D features into the shared 3D/BEV grid.
        self.img_view_transformer = LSSViewTransformer(
            grid_config=grid_config, input_size=input_size, in_channels=256,
            out_channels=numC_Trans, downsample=16, collapse_z=True)
        # BEV stream: a small ResNet + FPN that refine the top-down feature map.
        self.img_bev_encoder_backbone = CustomResNet(
            numC_input=numC_Trans,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8])
        self.img_bev_encoder_neck = FPN_LSS(
            in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256)
        # Occupancy head: BEV (2D) -> per-voxel (3D) class logits.
        self.occ_head = BEVOCCHead2D(in_dim=256, out_dim=256, Dz=Dz,
                                     num_classes=num_classes)

    # ---- geometry -------------------------------------------------------
    @staticmethod
    def prepare_inputs(sensor2egos, ego2globals):
        """Transform every camera into the *key* ego frame (camera 0's ego).

        Each of the 6 cameras is captured at a slightly different timestamp, so
        each has its own ego pose. To build one consistent BEV grid we express
        every camera in a single reference frame -- the ego pose of camera 0
        ("key ego"). The transform chain for camera i is:

            point_in_keyego = (global<-keyego)^-1 @ (global<-ego_i) @ (ego_i<-cam_i) @ point_in_cam_i

        i.e. lift cam_i -> its ego -> global, then pull global -> key ego.

        Args:
            sensor2egos: (B, N, 4, 4)  cam_i -> ego_i        (calibrated_sensor)
            ego2globals: (B, N, 4, 4)  ego_i -> global       (ego_pose)
        Returns:
            sensor2keyegos: (B, N, 4, 4)  cam_i -> key ego (camera 0's ego)
        """
        # Done on CPU in float64: MPS has no float64 and 4x4 inverses are
        # numerically sensitive (global translations are ~1e3 metres).
        sensor2egos = sensor2egos.cpu().double()
        ego2globals = ego2globals.cpu().double()
        keyego2global = ego2globals[:, 0:1, ...]                  # (B,1,4,4) cam0's ego->global
        global2keyego = torch.inverse(keyego2global)              # (B,1,4,4) global->key ego
        # Broadcast the (B,1,..) key transform across all N cameras.
        sensor2keyegos = global2keyego @ ego2globals @ sensor2egos
        return sensor2keyegos.float()

    # ---- forward --------------------------------------------------------
    def image_encoder(self, imgs):
        """Run the 2D backbone+neck on all cameras at once.

        The N cameras are folded into the batch dim (B*N) so the CNN treats
        them as independent images, then unfolded back to (B, N, ...).

        Args:
            imgs: (B, N, 3, H, W)
        Returns:
            (B, N, 256, fH, fW)  per-camera neck features (fH=H/16, fW=W/16)
        """
        B, N, C, H, W = imgs.shape
        x = imgs.view(B * N, C, H, W)           # fold cameras into batch
        feats = self.img_backbone(x)            # [C4 (s16), C5 (s32)]
        x = self.img_neck(feats)                # (B*N, 256, fH, fW)
        _, c, fh, fw = x.shape
        return x.view(B, N, c, fh, fw)          # unfold cameras

    def extract_img_feat(self, img_inputs):
        """Full image-to-BEV-feature path (everything except the occ head).

        Args:
            img_inputs: 7-tuple (imgs, sensor2egos, ego2globals, intrins,
                post_rots, post_trans, bda) -- see the loader for shapes.
        Returns:
            bev_feat: (B, 256, 200, 200),  depth: (B*N, D, fH, fW)
        """
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            img_inputs
        # (1) unify camera extrinsics into the key-ego frame
        sensor2keyegos = self.prepare_inputs(sensor2egos, ego2globals)
        # (2) 2D features per camera
        x = self.image_encoder(imgs)            # (B, N, 256, fH, fW)
        # (3) lift-splat into a top-down BEV feature map
        bev_feat, depth = self.img_view_transformer(
            x, sensor2keyegos, intrins, post_rots, post_trans, bda)
        # (4) refine the BEV feature with a ResNet+FPN
        bev_feat = self.img_bev_encoder_neck(
            self.img_bev_encoder_backbone(bev_feat))
        return bev_feat, depth

    def forward(self, img_inputs):
        """Returns occ class logits (B, Dx, Dy, Dz, n_cls)."""
        bev_feat, _ = self.extract_img_feat(img_inputs)
        return self.occ_head(bev_feat)

    @torch.no_grad()
    def predict_occ(self, img_inputs):
        """Returns argmax label volume (B, Dx, Dy, Dz) on CPU as uint8."""
        logits = self.forward(img_inputs)
        return self.occ_head.get_occ(logits).to(torch.uint8).cpu()


def load_flashocc_checkpoint(model, ckpt_path, verbose=True):
    """Load the official flashocc-r50 checkpoint into our model.

    All module/parameter names were chosen to match the checkpoint exactly, so
    this is a plain ``load_state_dict``.  Returns (missing, unexpected).
    """
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f'[ckpt] loaded {len(sd)} tensors | '
              f'missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            print('  missing   :', missing[:10],
                  '...' if len(missing) > 10 else '')
        if unexpected:
            print('  unexpected:', unexpected[:10],
                  '...' if len(unexpected) > 10 else '')
    return missing, unexpected
