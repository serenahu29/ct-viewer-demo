import os
import pathlib
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ——— Setup —————————————————————————————————————————————
BASE_DIR = pathlib.Path(__file__).parent
CT_PATH  = BASE_DIR / "ct.nii.gz"
SEG_DIR  = BASE_DIR / "segmentations"

# ——— Sidebar Debug Info ——————————————————————————————
st.sidebar.markdown("## 🔍 Debug Info")
st.sidebar.write("CT exists:", CT_PATH.exists())
st.sidebar.write("Segmentation dir exists:", SEG_DIR.exists())
if SEG_DIR.exists():
    seg_files = sorted(SEG_DIR.glob("*.nii.gz"))
    st.sidebar.write("Masks found:", [p.name for p in seg_files])
else:
    seg_files = []

# ——— Correct mask name extraction ——————————————————————
# Strip the full ".nii.gz" so names are "stomach", "liver", etc.
mask_names = [p.name[:-len(".nii.gz")] for p in seg_files]
overlay    = st.sidebar.multiselect("Overlay masks", mask_names)

# ——— Load CT Volume ————————————————————————————————
try:
    ct_vol = nib.load(str(CT_PATH)).get_fdata()
    _, _, nz = ct_vol.shape
except Exception as e:
    st.error(f"❌ Failed to load CT volume:\n{e}")
    st.stop()

# ——— Sidebar Controls ——————————————————————————————
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

st.sidebar.markdown(f"**Volume shape:** {ct_vol.shape}")

# ——— Prepare Slice ————————————————————————————————
slice_raw  = ct_vol[:, :, slice_idx]
slice_norm = np.clip((slice_raw - vmin) / (vmax - vmin), 0, 1).T

# ——— Plotting with On‑Demand Mask Loading ———————————————————
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(slice_norm, cmap="gray", origin="lower")

for name in overlay:
    mask_path = SEG_DIR / f"{name}.nii.gz"
    try:
        mask_vol   = nib.load(str(mask_path)).get_fdata()
        mask_slice = mask_vol[:, :, slice_idx].T
        ax.imshow(mask_slice, cmap="jet", alpha=0.4, origin="lower")
    except Exception as e:
        st.error(f"❌ Failed to load/overlay mask '{name}': {e}")
        st.stop()

ax.set_title(f"Axial slice {slice_idx}")
ax.axis("off")

st.pyplot(fig, use_container_width=True)
