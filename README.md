# CT Viewer Demo

This repository contains a web‑based CT viewer application built with **Streamlit**, allowing interactive visualization of a 3D CT volume and optional per‑voxel mask overlays (organs and structures). It meets the requirements of Project I for the CCVL BodyMaps program.



## Features

- **Axial slice navigation**: scroll through CT slices with a slider.
- **Window/level controls**: adjust contrast and brightness in real time.
- **Mask overlays**: toggle any number of organ masks (e.g., liver, pancreas, spleen).
- **Pure Python**: no JavaScript required; everything runs in Streamlit.
- **One‑click deployment**: live demo available on Streamlit Cloud.


## Preview
### Inital view
<img width="1512" alt="demo" src="https://github.com/user-attachments/assets/22421e21-a3d3-4b6a-a58f-136ca12f3246" />

### Mask overlay
<img width="1512" alt="Screenshot 2025-04-20 at 2 02 26 AM" src="https://github.com/user-attachments/assets/583a49fe-e77e-4ecd-b99d-90eb0e7ce6ad" />



## Repository Structure

```
ct-viewer-demo/
├─ app.py               # Main Streamlit application
├─ requirements.txt     # Python dependencies
├─ ct.nii.gz            # CT volume file
├─ segmentations/       # Folder of organ mask .nii.gz files
│   ├─ liver.nii.gz
│   ├─ pancreas.nii.gz
│   └─ ...
└─ README.md            # This file
```

##  Setup Instructions

To run the viewer locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/serenahu29/ct-viewer-demo.git
   cd ct-viewer-demo
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
   The first run will download the MedSAM2 model weights (~150 MB) automatically.
3. Run the app
   ```bash
   streamlit run app.py
   ```
  
   


## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/serenahu29/ct-viewer-demo/main/app.py)


## Usage

- Use the **Axial slice** slider to navigate through the CT volume.
- Adjust **Window Min** and **Window Max** to fine‑tune brightness and contrast.
- Select one or more **Overlay masks** from the multiselect dropdown to see per‑voxel annotations.
- Click **Run luna25medsam2** in the sidebar to automatically segment the CT volume. The resulting mask is saved to `segmentations/luna25medsam2.nii.gz` and can be toggled like any other overlay.


## Deployment

This app is hosted on Streamlit Cloud.

## Future Work

Currently supports axial viewing. Future enhancements may include sagittal and coronal slicing.




