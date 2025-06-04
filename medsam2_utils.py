"""Utility helpers for running MedSAM2 segmentation on 3D volumes."""

from __future__ import annotations

import numpy as np
import torch
from functools import lru_cache
from importlib import resources
import cv2
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_video_predictor_npz


def _get_config_path() -> str:
    """Return the absolute path to the default MedSAM2 config."""
    return str(resources.files("sam2").joinpath("configs", "sam2.1_hiera_t512.yaml"))


@lru_cache(maxsize=1)
def load_medsam2_model() -> torch.nn.Module:
    """Download weights (if needed) and load the MedSAM2 model.

    Returns
    -------
    torch.nn.Module
        The MedSAM2 video predictor model in evaluation mode.
    """

    ckpt_path = hf_hub_download("wanglab/MedSAM2", filename="MedSAM2_latest.pt")
    cfg_path = _get_config_path()
    model = build_sam2_video_predictor_npz(cfg_path, ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model


def _preprocess_ct(volume: np.ndarray, imsize: int = 512) -> np.ndarray:
    """Window/level the CT volume and resize for the model."""

    wl, ww = -750, 1500
    lower, upper = wl - ww / 2, wl + ww / 2
    vol = np.clip(volume, lower, upper)
    vol = (vol - vol.min()) / (vol.max() - vol.min()) * 255.0
    vol = vol.astype(np.uint8)

    d, h, w = vol.shape
    out = np.zeros((d, 3, imsize, imsize), dtype=np.float32)
    for i in range(d):
        img = cv2.resize(vol[i], (imsize, imsize))
        out[i] = np.stack([img, img, img], axis=0) / 255.0
    return out


def segment_volume(volume: np.ndarray) -> np.ndarray:
    """Run MedSAM2 on a CT volume.

    Parameters
    ----------
    volume : np.ndarray
        CT volume in Hounsfield units (D, H, W).

    Returns
    -------
    np.ndarray
        Binary mask volume of the same shape as ``volume``.
    """

    model = load_medsam2_model()
    device = next(model.parameters()).device

    vol_proc = _preprocess_ct(volume)
    tensor = torch.from_numpy(vol_proc).to(device)
    d, _, im_h, im_w = tensor.shape
    state = model.init_state(tensor, im_h, im_w)

    # use a center point on the middle slice as a simple prompt
    mid = d // 2
    points = np.array([[im_w // 2, im_h // 2]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    model.add_new_points_or_box(state, frame_idx=mid, obj_id=1, points=points, labels=labels)

    masks = np.zeros((d, im_h, im_w), dtype=np.uint8)
    for frame_idx, _, mask_logits in model.propagate_in_video(state):
        masks[frame_idx] = (mask_logits[0] > 0.0).cpu().numpy()[0]

    # resize masks back to original resolution
    d_o, h_o, w_o = volume.shape
    output = np.zeros((d_o, h_o, w_o), dtype=np.uint8)
    for i in range(d):
        output[i] = cv2.resize(masks[i].astype(np.uint8), (w_o, h_o), interpolation=cv2.INTER_NEAREST)

    return output

