import os
import pathlib
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas
from medsam2_utils import load_medsam2_model, segment_volume

# ——— Paths —————————————————————————————————————————————
BASE_DIR = pathlib.Path(__file__).parent
CT_PATH = BASE_DIR / "ct.nii.gz"
SEG_DIR = BASE_DIR / "segmentations"
ANNOT_DIR = BASE_DIR / "output_annotations"
ANNOT_DIR.mkdir(exist_ok=True)

# Cache heavy I/O operations for performance
@st.cache_data
def load_nifti(path):
    return nib.load(str(path)).get_fdata()

ct_vol = load_nifti(CT_PATH)
_, _, nz = ct_vol.shape

# ——— Session State Initialization ——————————————————————————
if 'slice_idx' not in st.session_state:
    st.session_state.slice_idx = nz // 2
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = str(np.random.randint(0, 1e6))
if 'last_saved_slice' not in st.session_state:
    st.session_state.last_saved_slice = None

# ——— UI Sidebar ————————————————————————————————————————
st.title("🩻 Axial CT + Annotation Viewer")

st.sidebar.markdown("""
### Annotation Guide
1. Switch modes: Free Draw, Polygon, Rectangle
2. Adjust brush size
3. Save or clear canvas
4. Use opacity slider to control overlay
""")

# Gather annotated slices dynamically
def get_annotated_slices():
    return sorted(
        int(f.stem.split('_')[-1])
        for f in ANNOT_DIR.glob("annotation_slice_*.npy")
    )

annotated_slices = get_annotated_slices()

# Navigation controls
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("← Prev") and annotated_slices:
        idx = annotated_slices.index(st.session_state.slice_idx) if st.session_state.slice_idx in annotated_slices else 0
        st.session_state.slice_idx = annotated_slices[(idx - 1) % len(annotated_slices)]
        st.session_state.canvas_key = str(np.random.randint(0, 1e6))
        st.rerun()
with col2:
    if st.button("Next →") and annotated_slices:
        idx = annotated_slices.index(st.session_state.slice_idx) if st.session_state.slice_idx in annotated_slices else 0
        st.session_state.slice_idx = annotated_slices[(idx + 1) % len(annotated_slices)]
        st.session_state.canvas_key = str(np.random.randint(0, 1e6))
        st.rerun()

# Quick-access slice buttons
if annotated_slices:
    st.sidebar.markdown("### Annotated Slices")
    for s in annotated_slices:
        if st.sidebar.button(f"Slice {s}", key=f"s{s}"):
            st.session_state.slice_idx = s
            st.session_state.canvas_key = str(np.random.randint(0, 1e6))
            st.rerun()

# Controls: mode, slice, windowing, overlays
mode = st.sidebar.radio("Mode", ["Free Draw", "Polygon", "Rectangle"] )
st.session_state.slice_idx = st.sidebar.slider("Slice Z", 0, nz - 1, st.session_state.slice_idx)
vmin = st.sidebar.slider("Window Min", float(ct_vol.min()), float(ct_vol.max()), float(np.percentile(ct_vol, 1)))
vmax = st.sidebar.slider("Window Max", float(ct_vol.min()), float(ct_vol.max()), float(np.percentile(ct_vol, 99)))
opac = st.sidebar.slider("Annotation Opacity", 0.0, 1.0, 0.5)

# Load masks list once
seg_names = sorted(p.stem for p in SEG_DIR.glob("*.nii.gz"))
overlays = st.sidebar.multiselect("Overlay masks", seg_names)
if st.sidebar.button("Run luna25medsam2"):
    with st.spinner("Running segmentation..."):
        mask_vol = segment_volume(ct_vol)
        out_path = SEG_DIR / "luna25medsam2.nii.gz"
        nib.save(nib.Nifti1Image(mask_vol.astype(np.uint8), np.eye(4)), str(out_path))
    st.success(f"Saved {out_path.name}")
    st.rerun()

# Prepare image slice
slice_raw = ct_vol[:, :, st.session_state.slice_idx]
slice_norm = np.clip((slice_raw - vmin) / (vmax - vmin), 0, 1)
slice_img = (slice_norm * 255).astype(np.uint8)
slice_pil = Image.fromarray(slice_img)

# Canvas setup
brush = st.sidebar.slider("Brush Size", 1, 50, 10)
draw_mode = {"Free Draw": "freedraw", "Polygon": "polygon", "Rectangle": "rect"}[mode]
canvas = st_canvas(
    fill_color="rgba(255,0,0,0.3)",
    stroke_width=brush,
    stroke_color="rgba(255,0,0,1)",
    background_image=slice_pil,
    drawing_mode=draw_mode,
    height=slice_img.shape[0],
    width=slice_img.shape[1],
    key=st.session_state.canvas_key,
)

# ——— Saving / Clearing —————————————————————————————————————
c1, c2 = st.columns(2)
with c1:
    if st.button("Save Annotation") and canvas.image_data is not None:
        # use alpha channel for robust mask extraction (handles all modes)
        alpha = canvas.image_data[:, :, 3]
        mask = (alpha > 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        path = ANNOT_DIR / f"annotation_slice_{st.session_state.slice_idx}.npy"
        np.save(path, mask)
        st.success(f"Saved {path.name}")
        st.session_state.last_saved_slice = st.session_state.slice_idx
        if st.session_state.slice_idx < nz - 1:
            st.session_state.slice_idx += 1
            st.session_state.canvas_key = str(np.random.randint(0, 1e6))
            st.rerun()
with c2:
    if st.button("Clear Canvas"):
        st.session_state.canvas_key = str(np.random.randint(0, 1e6))
        st.rerun()

# ——— Plotting —————————————————————————————————————————
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(slice_norm, cmap='gray', origin='upper')

# overlay saved mask
saved_path = ANNOT_DIR / f"annotation_slice_{st.session_state.slice_idx}.npy"
if saved_path.exists():
    saved = np.load(saved_path)
    overlay = np.zeros((*saved.shape, 4))
    overlay[saved > 0] = [1, 0, 0, opac]
    ax.imshow(overlay, origin='upper')

# overlay current canvas (green)
if canvas.image_data is not None:
    alpha = canvas.image_data[:, :, 3]
    curr = (alpha > 0).astype(np.uint8)
    overlay = np.zeros((*curr.shape, 4))
    overlay[curr > 0] = [0, 1, 0, opac]
    ax.imshow(overlay, origin='upper')

# overlay segmentation masks
for name in overlays:
    seg_vol = load_nifti(SEG_DIR / f"{name}.nii.gz")
    seg_slice = seg_vol[:, :, st.session_state.slice_idx]
    overlay = np.zeros((*seg_slice.shape, 4))
    overlay[seg_slice > 0] = [0, 0, 1, 0.3]
    ax.imshow(overlay, origin='upper')

# status
status = f"Slice {st.session_state.slice_idx}"
if st.session_state.slice_idx in annotated_slices:
    status += " (annotated)"
if st.session_state.slice_idx == st.session_state.last_saved_slice:
    status += " (last saved)"
ax.set_title(status)
ax.axis('off')

st.pyplot(fig, use_container_width=True)
