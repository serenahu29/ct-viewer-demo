import nibabel as nib
import matplotlib.pyplot as plt
import os

# load data
ct = nib.load("ct.nii.gz").get_fdata()
mask = nib.load(os.path.join("segmentations","liver.nii.gz")).get_fdata()

z_mid = ct.shape[2] // 2
slicect = ct[:, :, z_mid].T
slicemask = mask[:, :, z_mid].T

# make one figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(slicect, cmap="gray", origin="lower")
ax1.set_title(f"CT (z={z_mid})")
ax1.axis("off")

ax2.imshow(slicect, cmap="gray", origin="lower")
ax2.imshow(slicemask, cmap="jet", alpha=0.4, origin="lower")
ax2.set_title("CT + Liver")
ax2.axis("off")

plt.tight_layout()
plt.show()

