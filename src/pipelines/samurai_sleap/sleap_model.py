"""
Pure PyTorch SLEAP bottom-up model loader and inference.

Reconstructs the UNet architecture from the sleap-nn checkpoint
so we can run inference without the sleap-nn package.

Architecture (from checkpoint):
  Encoder: 4 blocks (3->64->128->256->512), each 2x Conv2d+ReLU, then MaxPool2d
  Middle:  512->1024->1024 (2 conv blocks)
  Decoder: 3 blocks (s16->s8->s4->s2), upsample + skip concat + 2x Conv2d+ReLU
  Heads:   ConfmapsHead 128->7 (1x1 conv, 7 keypoints at stride 2)
           PAFsHead     128->12 (1x1 conv, 6 edges x 2 directions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UNet architecture matching the SLEAP-NN checkpoint
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two 3x3 convolutions with ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        return x


class SLEAPUNet(nn.Module):
    """UNet matching the sleap-nn bottom-up checkpoint structure."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc0 = ConvBlock(3, 64)
        self.enc1 = ConvBlock(64, 128)
        self.enc2 = ConvBlock(128, 256)
        self.enc3 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Middle
        self.mid_expand = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.mid_contract = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Decoder (upsample + concat skip + refine)
        # sleap-nn concatenates: [skip, upsampled] (skip first)
        self.dec0_refine = ConvBlock(512 + 1024, 512)   # s16 -> s8
        self.dec1_refine = ConvBlock(256 + 512, 256)     # s8 -> s4
        self.dec2_refine = ConvBlock(128 + 256, 128)     # s4 -> s2

        # Heads (1x1 conv)
        self.confmaps_head = nn.Conv2d(128, 7, 1)
        self.pafs_head = nn.Conv2d(128, 12, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)             # s1, 64 ch
        e1 = self.enc1(self.pool(e0)) # s2, 128 ch
        e2 = self.enc2(self.pool(e1)) # s4, 256 ch
        e3 = self.enc3(self.pool(e2)) # s8, 512 ch

        # Middle
        m = self.pool(e3)             # s16
        m = self.mid_expand(m)        # s16, 1024 ch
        m = self.mid_contract(m)      # s16, 1024 ch

        # Decoder — skip connection FIRST, then upsampled features
        d0 = F.interpolate(m, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d0 = self.dec0_refine(torch.cat([e3, d0], dim=1))   # [skip, up] -> s8, 512 ch

        d1 = F.interpolate(d0, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1_refine(torch.cat([e2, d1], dim=1))   # [skip, up] -> s4, 256 ch

        d2 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2_refine(torch.cat([e1, d2], dim=1))   # [skip, up] -> s2, 128 ch

        confmaps = self.confmaps_head(d2)   # (B, 7, H/2, W/2)
        pafs = self.pafs_head(d2)           # (B, 12, H/2, W/2)
        return confmaps, pafs


# ---------------------------------------------------------------------------
# Checkpoint loading — map sleap-nn keys to our architecture
# ---------------------------------------------------------------------------

def _build_key_map() -> Dict[str, str]:
    """Map sleap-nn checkpoint keys -> our SLEAPUNet parameter names."""
    m = {}
    # Encoder blocks
    enc_channels = [(0, "enc0"), (1, "enc1"), (2, "enc2"), (3, "enc3")]
    for idx, name in enc_channels:
        prefix = f"model.backbone.encoders.0.encoder_stack.{idx}.blocks.stack0_enc{idx}"
        m[f"{prefix}_conv0.weight"] = f"{name}.conv0.weight"
        m[f"{prefix}_conv0.bias"] = f"{name}.conv0.bias"
        m[f"{prefix}_conv1.weight"] = f"{name}.conv1.weight"
        m[f"{prefix}_conv1.bias"] = f"{name}.conv1.bias"

    # Middle blocks
    m["model.backbone.middle_blocks.0.blocks.stack0_enc5_middle_expand_conv0.weight"] = "mid_expand.0.weight"
    m["model.backbone.middle_blocks.0.blocks.stack0_enc5_middle_expand_conv0.bias"] = "mid_expand.0.bias"
    m["model.backbone.middle_blocks.1.blocks.stack0_enc6_middle_contract_conv0.weight"] = "mid_contract.0.weight"
    m["model.backbone.middle_blocks.1.blocks.stack0_enc6_middle_contract_conv0.bias"] = "mid_contract.0.bias"

    # Decoder blocks
    dec_map = [
        (0, "s16_to_s8", "dec0_refine"),
        (1, "s8_to_s4", "dec1_refine"),
        (2, "s4_to_s2", "dec2_refine"),
    ]
    for idx, scale_name, our_name in dec_map:
        prefix = f"model.backbone.decoders.0.decoder_stack.{idx}.blocks.stack0_dec{idx}_{scale_name}_refine"
        m[f"{prefix}_conv0.weight"] = f"{our_name}.conv0.weight"
        m[f"{prefix}_conv0.bias"] = f"{our_name}.conv0.bias"
        m[f"{prefix}_conv1.weight"] = f"{our_name}.conv1.weight"
        m[f"{prefix}_conv1.bias"] = f"{our_name}.conv1.bias"

    # Head layers
    m["model.head_layers.0.MultiInstanceConfmapsHead.0.weight"] = "confmaps_head.weight"
    m["model.head_layers.0.MultiInstanceConfmapsHead.0.bias"] = "confmaps_head.bias"
    m["model.head_layers.1.PartAffinityFieldsHead.0.weight"] = "pafs_head.weight"
    m["model.head_layers.1.PartAffinityFieldsHead.0.bias"] = "pafs_head.bias"

    return m


def load_sleap_checkpoint(ckpt_path: str | Path, device: str = "cpu") -> SLEAPUNet:
    """Load the SLEAP checkpoint into our pure-PyTorch UNet."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    src_sd = ckpt["state_dict"]

    model = SLEAPUNet()
    key_map = _build_key_map()

    new_sd = {}
    for src_key, dst_key in key_map.items():
        if src_key not in src_sd:
            raise KeyError(f"Missing key in checkpoint: {src_key}")
        new_sd[dst_key] = src_sd[src_key]

    model.load_state_dict(new_sd)
    model.eval()
    model.to(device)
    logger.info("SLEAP UNet loaded from %s (%d parameters)", ckpt_path,
                sum(p.numel() for p in model.parameters()))
    return model


# ---------------------------------------------------------------------------
# Inference: confmaps -> peak detection -> instance grouping
# ---------------------------------------------------------------------------

def predict_keypoints(
    model: SLEAPUNet,
    frame_rgb: np.ndarray,
    keypoint_names: List[str],
    min_confidence: float = 0.3,
    device: str = "cpu",
) -> Tuple[np.ndarray, list]:
    """Run SLEAP inference on a single RGB frame.

    Returns:
        confmaps_np: (7, H/2, W/2) confidence maps
        peaks: list of (kpt_idx, x, y, conf) tuples in original image coordinates
    """
    h_in, w_in = frame_rgb.shape[:2]

    # Prepare input: (1, 3, H, W) float32 normalized
    img = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img = img.to(device)

    # Disable autocast — SLEAP runs inside SAM2's bfloat16 context but needs float32
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=False):
        confmaps, pafs = model(img)

    confmaps_np = confmaps.cpu().numpy().squeeze()  # (7, H/2, W/2)
    h_out, w_out = confmaps_np.shape[1], confmaps_np.shape[2]
    scale_x = w_in / w_out
    scale_y = h_in / h_out

    # Find peaks via local maximum filtering
    peaks = []
    for kpt_idx in range(min(confmaps_np.shape[0], len(keypoint_names))):
        cmap = confmaps_np[kpt_idx]
        local_max = maximum_filter(cmap, size=5)
        is_peak = (cmap == local_max) & (cmap > min_confidence)
        ys, xs = np.where(is_peak)
        confs = cmap[ys, xs]

        for py, px, pc in zip(ys, xs, confs):
            peaks.append((kpt_idx, float(px * scale_x), float(py * scale_y), float(pc)))

    logger.debug("SLEAP peak detection: %d peaks across %d channels (threshold=%.2f, confmap max=%.3f)",
                 len(peaks), confmaps_np.shape[0], min_confidence, confmaps_np.max())

    return confmaps_np, peaks
