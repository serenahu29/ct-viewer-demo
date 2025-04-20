import os
import pathlib
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ——— Page config ——————————————————————————————————————
st.set_page_config(page_title="Axial CT Viewer", layout="wide")
st.title("🩻 Axial CT + Mask Viewer")

# ——— Paths —————————————————————————————————————————
BASE_DIR = pathlib.Path(__file__).parent
CT_PATH  = BASE_DIR / "ct.nii.gz"
MASK_DIR = BASE_DIR / "segmentations"

# ——— Load volumes (cached) —————————————————————————————
@st.cache_data
def load_data():
    ct_vol = nib.load(str(CT_PATH)).get_fdata()
    masks  = {}
    for fname in sorted(os.listdir(MASK_DIR)):
        if fname.endswith(".nii.gz"):
            key = pathlib.Path(fname).stem
            masks[key] = nib.load(str(MASK_DIR / fname)).get_fdata()
    return ct_vol, masks

ct_vol, masks = load_data()
_, _, nz = ct_vol.shape

# ——— Sidebar controls ———————————————————————————————————
slice_ax = st.sidebar.slider("Axial slice (Z)", 0, nz - 1, nz // 2)

st.sidebar.header("Window / Level")
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

overlay_names = st.sidebar.multiselect(
    "Overlay masks", options=list(masks.keys()), default=[]
)
st.sidebar.markdown(f"**Volume shape:** {ct_vol.shape}")

# ——— Extract and normalize the axial slice —————————————————————
slice_raw  = ct_vol[:, :, slice_ax]
slice_norm = np.clip((slice_raw - vmin) / (vmax - vmin), 0, 1).T

# ——— Plot the axial view —————————————————————————————————
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(slice_norm, cmap="gray", origin="lower")

for name in overlay_names:
    mask_slice = masks[name][:, :, slice_ax].T
    ax.imshow(
        mask_slice,
        cmap="jet",
        alpha=0.4,
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="none",
    )

ax.set_title(f"Axial slice {slice_ax}")
ax.axis("off")

st.pyplot(fig, use_container_width=True)
