# CT Viewer Demo

This repository contains a web‑based CT viewer application built with **Streamlit**, allowing interactive visualization of a 3D CT volume and optional per‑voxel mask overlays (organs and structures). It meets the requirements of Project I for the CCVL BodyMaps program.

---

## 🚀 Features

- **Axial slice navigation**: scroll through CT slices with a slider.
- **Window/level controls**: adjust contrast and brightness in real time.
- **Mask overlays**: toggle any number of organ masks (e.g., liver, pancreas, spleen).
- **Pure Python**: no JavaScript required; everything runs in Streamlit.
- **One‑click deployment**: live demo available on Streamlit Cloud.

---

## 📂 Repository Structure

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

---

## ☁️ Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/serenahu29/ct-viewer-demo/main/app.py)

---

## 📝 Usage

- Use the **Axial slice** slider to navigate through the CT volume.
- Adjust **Window Min** and **Window Max** to fine‑tune brightness and contrast.
- Select one or more **Overlay masks** from the multiselect dropdown to see per‑voxel annotations.

---

## 🚩 Deployment

This app is hosted on Streamlit Cloud.

---



