import os
import pathlib
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
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

st.sidebar.markdown("""
### Annotation Guide
1. Choose annotation mode:
   - Free Draw: Draw freely
   - Rectangle: Drag to create a rectangle
2. Adjust brush size
3. Click 'Save Annotation'
4. 'Clear Canvas' to reset
5. Adjust opacity to view underlying CT
""")

# Add Reset All button
if st.sidebar.button("Reset All Annotations", type="primary"):
    # Delete all annotation files
    for file in ANNOT_DIR.glob("annotation_slice_*.npy"):
        file.unlink()
    st.sidebar.success("All annotations have been cleared!")
    # Reset session state
    st.session_state.last_saved_slice = None
    st.rerun()

# List annotated slices
annotated_slices = sorted([
    int(p.name.split('_')[-1].split('.')[0])
    for p in ANNOT_DIR.glob("annotation_slice_*.npy")
])

# Navigation
col_pr, col_nx = st.sidebar.columns(2)
with col_pr:
    if st.button("← Previous"):  
        if annotated_slices:
            idx_list = annotated_slices
            ci = idx_list.index(st.session_state.slice_idx) if st.session_state.slice_idx in idx_list else 0
            st.session_state.slice_idx = idx_list[(ci-1) % len(idx_list)]
            st.session_state.canvas_key = str(np.random.randint(0, 1e6))
            st.rerun()
with col_nx:
    if st.button("Next →"):
        if annotated_slices:
            idx_list = annotated_slices
            ci = idx_list.index(st.session_state.slice_idx) if st.session_state.slice_idx in idx_list else 0
            st.session_state.slice_idx = idx_list[(ci+1) % len(idx_list)]
            st.session_state.canvas_key = str(np.random.randint(0, 1e6))
            st.rerun()

if annotated_slices:
    st.sidebar.markdown("### Annotated Slices")
    for s in annotated_slices:
        if st.sidebar.button(f"Slice {s}", key=f"slice_{s}"):
            st.session_state.slice_idx = s
            st.session_state.canvas_key = str(np.random.randint(0, 1e6))
            st.rerun()

# Mode, slice, window, overlays
annotation_mode = st.sidebar.radio("Mode", ["Free Draw", "Rectangle"], index=0)
st.session_state.slice_idx = st.sidebar.slider("Slice Z",0,nz-1,st.session_state.slice_idx)
vmin = st.sidebar.slider("Window Min", float(ct_vol.min()), float(ct_vol.max()), float(np.percentile(ct_vol,1)))
vmax = st.sidebar.slider("Window Max", float(ct_vol.min()), float(ct_vol.max()), float(np.percentile(ct_vol,99)))
opacity = st.sidebar.slider("Annotation Opacity",0.0,1.0,0.5)
seg_files = sorted(SEG_DIR.glob("*.nii.gz"))
overlays = [p.name[:-7] for p in seg_files]
over_choice = st.sidebar.multiselect("Overlay masks", overlays)

# Prepare slice image
slice_data = ct_vol[:,:,st.session_state.slice_idx]
slice_norm = np.clip((slice_data - vmin)/(vmax-vmin),0,1)
slice_img = (slice_norm*255).astype(np.uint8)
slice_pil = Image.fromarray(slice_img)

# Canvas
brush = st.sidebar.slider("Brush Size",1,50,10)
drawing_mode = {"Free Draw":"freedraw","Rectangle":"rect"}[annotation_mode]
canvas = st_canvas(
    fill_color="rgba(255,0,0,0.3)",
    stroke_width=brush,
    stroke_color="rgba(255,0,0,1)",
    background_image=slice_pil,
    drawing_mode=drawing_mode,
    height=slice_img.shape[0],
    width=slice_img.shape[1],
    key=st.session_state.canvas_key
)

# Save / Clear
c1,c2 = st.columns(2)
with c1:
    if st.button("Save Annotation") and canvas.json_data:
        mask = np.zeros(slice_img.shape, dtype=np.uint8)
        for obj in canvas.json_data["objects"]:
            t = obj.get("type")
            if t == "path":
                path_data = obj.get("path", [])
                if isinstance(path_data, list) and len(path_data) >= 2:
                    # Convert path points to numpy array, ensuring consistent dimensions
                    pts = []
                    for point in path_data:
                        if isinstance(point, (list, tuple)) and len(point) == 2:
                            pts.append([int(point[0]), int(point[1])])
                    if pts:
                        pts = np.array(pts, dtype=np.int32)
                        cv2.polylines(mask, [pts], False, 1, brush)
            elif t == "polygon":
                pts = np.array(obj.get("points",[]),dtype=np.int32)
                if pts.shape[0]>=3:
                    cv2.fillPoly(mask,[pts],1)
            elif t == "rect":
                x=int(obj["left"]); y=int(obj["top"])
                w=int(obj["width"]); h=int(obj["height"])
                cv2.rectangle(mask,(x,y),(x+w,y+h),1,-1)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        outp = ANNOT_DIR / f"annotation_slice_{st.session_state.slice_idx}.npy"
        np.save(outp,mask)
        st.success(f"Saved {outp.name}")
        st.session_state.last_saved_slice = st.session_state.slice_idx
        if st.session_state.slice_idx < nz-1:
            st.session_state.slice_idx +=1
            st.session_state.canvas_key = str(np.random.randint(0,1e6))
            st.rerun()
with c2:
    if st.button("Clear Canvas"):
        # Clear the canvas data
        if canvas.json_data and 'objects' in canvas.json_data:
            canvas.json_data['objects'] = []
        
        # Remove saved annotation file if it exists
        annotation_file = ANNOT_DIR / f"annotation_slice_{st.session_state.slice_idx}.npy"
        if annotation_file.exists():
            annotation_file.unlink()
            st.success(f"Cleared annotation for slice {st.session_state.slice_idx}")
        
        # Generate new canvas key to ensure clean state
        st.session_state.canvas_key = str(np.random.randint(0,1e6))
        st.rerun()

# Plotting
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(slice_norm, cmap='gray', origin='upper')

# saved annotation
ap = ANNOT_DIR/f"annotation_slice_{st.session_state.slice_idx}.npy"
if ap.exists():
    sm = np.load(ap)
    ao = np.zeros((*sm.shape,4))
    ao[sm>0] = [1,0,0,opacity]
    ax.imshow(ao,origin='upper')

# current drawing
if canvas.json_data and 'objects' in canvas.json_data:
    cm = np.zeros(slice_img.shape,dtype=np.uint8)
    for obj in canvas.json_data['objects']:
        try:
            t = obj.get('type')
            if t == 'path':
                path_data = obj.get('path', [])
                if isinstance(path_data, list) and len(path_data) >= 2:
                    # Convert path points to numpy array, ensuring consistent dimensions
                    pts = []
                    for point in path_data:
                        if isinstance(point, (list, tuple)) and len(point) == 2:
                            pts.append([int(point[0]), int(point[1])])
                    if pts:
                        pts = np.array(pts, dtype=np.int32)
                        cv2.polylines(cm, [pts], False, 1, brush)
            elif t == 'polygon':
                pts = np.array(obj.get('points',[]),dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(cm,[pts],1)
            elif t == 'rect':
                x = int(obj.get('left',0))
                y = int(obj.get('top',0))
                w = int(obj.get('width',0))
                h = int(obj.get('height',0))
                if w > 0 and h > 0:
                    cv2.rectangle(cm,(x,y),(x+w,y+h),1,-1)
        except Exception as e:
            st.warning(f"Error processing drawing object: {str(e)}")
            continue
    
    # Clean up the mask
    cm = cv2.morphologyEx(cm,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    
    # Create overlay
    co = np.zeros((*cm.shape,4))
    co[cm>0] = [0,1,0,opacity]  # Green for current drawing
    ax.imshow(co,origin='upper')

# overlays
for name in over_choice:
    try:
        mv = nib.load(str(SEG_DIR/f"{name}.nii.gz")).get_fdata()[:,:,st.session_state.slice_idx]
        mo = np.zeros((*mv.shape,4))
        mo[mv>0] = [0,0,1,0.3]  # Blue for overlays
        ax.imshow(mo,origin='upper')
    except Exception as e:
        st.warning(f"Error loading overlay {name}: {str(e)}")

# title/status
status = f"Slice {st.session_state.slice_idx}"
if st.session_state.slice_idx in annotated_slices:
    status += " (annotated)"
if st.session_state.slice_idx == st.session_state.last_saved_slice:
    status += " (last saved)"
ax.set_title(status)
ax.axis('off')
st.pyplot(fig,use_container_width=True)
