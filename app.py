import os
import pathlib
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import json
from PIL import Image
import io
import cv2
from streamlit_drawable_canvas import st_canvas

# ——— Paths —————————————————————————————————————————————
BASE_DIR = pathlib.Path(__file__).parent
CT_PATH  = BASE_DIR / "ct.nii.gz"
SEG_DIR  = BASE_DIR / "segmentations"
ANNOT_DIR = BASE_DIR / "output_annotations"
ANNOT_DIR.mkdir(exist_ok=True)

# ——— Load CT volume —————————————————————————————————————
ct_vol = nib.load(str(CT_PATH)).get_fdata()
_, _, nz = ct_vol.shape

# Initialize session state
if 'slice_idx' not in st.session_state:
    st.session_state.slice_idx = nz // 2
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = str(np.random.randint(0, 1000000))
if 'last_saved_slice' not in st.session_state:
    st.session_state.last_saved_slice = None

# ——— Sidebar controls ———————————————————————————————————
st.title("🩻 Axial CT + Annotation Viewer")

# Help text in sidebar
st.sidebar.markdown("""
### Annotation Guide
1. Choose annotation mode:
   - Free Draw: Draw freely on the image
   - Polygon: Click points to create a polygon
   - Rectangle: Click and drag to create a rectangle
2. Adjust brush size as needed
3. Click 'Save Annotation' to store your work
4. Use 'Clear Canvas' to start over
5. Adjust annotation opacity to see the CT scan better

**Tips:**
- Use larger brush sizes for rough outlines
- Use smaller brush sizes for fine details
- You can combine different annotation modes
- Saved annotations are shown in red
- Current drawing is shown in green
""")

# Get list of annotated slices
annotated_slices = sorted([
    int(f.name.split('_')[-1].split('.')[0])
    for f in ANNOT_DIR.glob("annotation_slice_*.npy")
])

# Navigation controls
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("← Previous Annotated"):
        if annotated_slices:
            current_idx = annotated_slices.index(st.session_state.slice_idx) if st.session_state.slice_idx in annotated_slices else 0
            new_idx = (current_idx - 1) % len(annotated_slices)
            st.session_state.slice_idx = annotated_slices[new_idx]
            st.session_state.canvas_key = str(np.random.randint(0, 1000000))
            st.rerun()
with col2:
    if st.button("Next Annotated →"):
        if annotated_slices:
            current_idx = annotated_slices.index(st.session_state.slice_idx) if st.session_state.slice_idx in annotated_slices else 0
            new_idx = (current_idx + 1) % len(annotated_slices)
            st.session_state.slice_idx = annotated_slices[new_idx]
            st.session_state.canvas_key = str(np.random.randint(0, 1000000))
            st.rerun()

# Show annotated slices in sidebar
if annotated_slices:
    st.sidebar.markdown("### Annotated Slices")
    for idx in annotated_slices:
        if st.sidebar.button(f"Slice {idx}", key=f"slice_{idx}"):
            st.session_state.slice_idx = idx
            st.session_state.canvas_key = str(np.random.randint(0, 1000000))
            st.rerun()

# Annotation mode selection
annotation_mode = st.sidebar.radio(
    "Annotation Mode",
    ["Free Draw", "Polygon", "Rectangle"],
    index=0
)

slice_idx = st.sidebar.slider("Axial slice (Z)", 0, nz - 1, st.session_state.slice_idx)
st.session_state.slice_idx = slice_idx  # Update session state

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

# Annotation opacity control
annotation_opacity = st.sidebar.slider(
    "Annotation Opacity",
    0.0, 1.0, 0.5,
    help="Adjust how visible the annotations are on the CT scan"
)

# List available masks by stripping the full ".nii.gz" suffix
seg_files  = sorted(SEG_DIR.glob("*.nii.gz"))
mask_names = [p.name[:-len(".nii.gz")] for p in seg_files]
overlay    = st.sidebar.multiselect("Overlay masks", mask_names)

st.sidebar.markdown(f"**Volume shape:** {ct_vol.shape}")

# ——— Prepare the axial slice —————————————————————————————
slice_raw  = ct_vol[:, :, slice_idx]
slice_norm = np.clip((slice_raw - vmin) / (vmax - vmin), 0, 1).T

# Convert to PIL Image for canvas
slice_pil = Image.fromarray((slice_norm * 255).astype(np.uint8))

# ——— Drawing Canvas —————————————————————————————————————
brush_size = st.slider("Brush Size", 1, 50, 10)

# Map annotation mode to canvas drawing mode
drawing_mode = {
    "Free Draw": "freedraw",
    "Polygon": "polygon",
    "Rectangle": "rect"
}[annotation_mode]

# Create a canvas with specific dimensions
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=brush_size,
    stroke_color="rgba(255, 0, 0, 1)",
    background_image=slice_pil,
    drawing_mode=drawing_mode,
    height=slice_norm.shape[0],
    width=slice_norm.shape[1],
    key=st.session_state.canvas_key,
)

# ——— Save Annotation —————————————————————————————————————
col1, col2 = st.columns(2)
with col1:
    if st.button("Save Annotation"):
        if canvas_result.json_data is not None:
            # Convert canvas to binary mask
            mask = np.zeros_like(slice_norm, dtype=np.uint8)
            
            # Process each object in the canvas
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "path":
                    # For free drawing, create a continuous line
                    points = obj["path"]
                    if len(points) >= 2:
                        points = np.array(points, dtype=np.int32)
                        cv2.polylines(mask, [points], False, 1, brush_size)
                elif obj["type"] in ["polygon", "rect"]:
                    # For polygon and rectangle, fill the area
                    if "points" in obj:
                        points = obj["points"]
                        if len(points) >= 3:
                            points = np.array(points, dtype=np.int32)
                            cv2.fillPoly(mask, [points], 1)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Save as .npy
            save_path = ANNOT_DIR / f"annotation_slice_{slice_idx}.npy"
            np.save(save_path, mask)
            st.success(f"Annotation saved to {save_path}")
            
            # Update last saved slice
            st.session_state.last_saved_slice = slice_idx
            
            # Move to next slice
            if slice_idx < nz - 1:
                st.session_state.slice_idx = slice_idx + 1
                st.session_state.canvas_key = str(np.random.randint(0, 1000000))
                st.rerun()

with col2:
    if st.button("Clear Canvas"):
        st.session_state.canvas_key = str(np.random.randint(0, 1000000))
        st.rerun()

# ——— Plot —————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(slice_norm, cmap="gray", origin="lower")

# Show saved annotation if exists
annot_path = ANNOT_DIR / f"annotation_slice_{slice_idx}.npy"
if annot_path.exists():
    saved_mask = np.load(annot_path)
    # Create a colored overlay for the annotation
    annotation_overlay = np.zeros((*saved_mask.shape, 4))
    annotation_overlay[saved_mask > 0] = [1, 0, 0, annotation_opacity]  # Red with adjustable opacity
    ax.imshow(annotation_overlay, origin="lower")

# Show current canvas drawing if any
if canvas_result.json_data is not None:
    current_mask = np.zeros_like(slice_norm, dtype=np.uint8)
    
    # Process each object in the canvas
    for obj in canvas_result.json_data["objects"]:
        if obj["type"] == "path":
            # For free drawing, create a continuous line
            points = obj["path"]
            if len(points) >= 2:
                points = np.array(points, dtype=np.int32)
                cv2.polylines(current_mask, [points], False, 1, brush_size)
        elif obj["type"] in ["polygon", "rect"]:
            # For polygon and rectangle, fill the area
            if "points" in obj:
                points = obj["points"]
                if len(points) >= 3:
                    points = np.array(points, dtype=np.int32)
                    cv2.fillPoly(current_mask, [points], 1)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
    
    # Show current drawing with a different color
    current_overlay = np.zeros((*current_mask.shape, 4))
    current_overlay[current_mask > 0] = [0, 1, 0, annotation_opacity]  # Green with adjustable opacity
    ax.imshow(current_overlay, origin="lower")

# Show overlay masks
for name in overlay:
    mask_path = SEG_DIR / f"{name}.nii.gz"
    mask_vol  = nib.load(str(mask_path)).get_fdata()
    mask_slice = mask_vol[:, :, slice_idx].T
    mask_overlay = np.zeros((*mask_slice.shape, 4))
    mask_overlay[mask_slice > 0] = [0, 0, 1, 0.3]  # Blue with fixed opacity
    ax.imshow(mask_overlay, origin="lower")

# Add status text
status_text = f"Axial slice {slice_idx}"
if slice_idx in annotated_slices:
    status_text += " (annotated)"
if slice_idx == st.session_state.last_saved_slice:
    status_text += " (last saved)"
ax.set_title(status_text)
ax.axis("off")

st.pyplot(fig, use_container_width=True)
