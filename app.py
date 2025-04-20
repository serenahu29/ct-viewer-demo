import os
import pathlib
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ——— Paths —————————————————————————————————————————————
BASE_DIR = pathlib.Path(__file__).parent
CT_PATH  = BASE_DIR / "ct.nii.gz"
SEG_DIR  = BASE_DIR / "segmentations"

# ——— Load CT volume —————————————————————————————————————
ct_vol = nib.load(str(CT_PATH)).get_fdata()
_, _, nz = ct_vol.shape

# ——— Sidebar controls ———————————————————————————————————
st.title("🩻 Axial CT + Mask Viewer")

slice_idx = st.sidebar.slider("Axial slice (Z)", 0, nz - 1, nz // 2)

vmin = st.sidebar.slider(
    "Window Min",
    float(ct_vol.min()),
    float(ct_vol.max()),
    float(np.percentile(ct_vol, 1)),
)
vmax = st.sidebar.slider(
    "Window Max",
    float(ct_vol.min()),
    float(ct_vol.max()),
    float(np.percentile(ct_vol, 99)),
)

# List available masks by stripping the full ".nii.gz" suffix
seg_files  = sorted(SEG_DIR.glob("*.nii.gz"))
mask_names = [p.name[:-len(".nii.gz")] for p in seg_files]
overlay    = st.sidebar.multiselect("Overlay masks", mask_names)

st.sidebar.markdown(f"**Volume shape:** {ct_vol.shape}")

# ——— Prepare the axial slice —————————————————————————————
slice_raw  = ct_vol[:, :, slice_idx]
slice_norm = np.clip((slice_raw - vmin) / (vmax - vmin), 0, 1).T

# ——— Plot —————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(slice_norm, cmap="gray", origin="lower")

for name in overlay:
    mask_path = SEG_DIR / f"{name}.nii.gz"
    mask_vol  = nib.load(str(mask_path)).get_fdata()
    mask_slice = mask_vol[:, :, slice_idx].T
    ax.imshow(mask_slice, cmap="jet", alpha=0.4, origin="lower")

ax.set_title(f"Axial slice {slice_idx}")
ax.axis("off")

st.pyplot(fig, use_container_width=True)
