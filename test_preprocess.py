
import os
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from preprocess import (
    apply_rgb_compo,
    apply_cloud_mask,
    digital_numbers_to_reflectance,
)

folder = "data/SENTINEL2_L2A/west=6.547121-south=47.910627-east=6.572061-north=47.93576/2016-01-01-2018-01-01"
date = "2016-09-22"

rasters = {}
for band in ("B02","B03","B04","B08","B11"):
    path = os.path.join(
        folder, band, f"SENTINEL2_L2A_{band}_{date}Z.tif"
    )
    rasters[band] = rasterio.open(path)

cloud_mask = apply_cloud_mask(
    rasters,
    preprocess=digital_numbers_to_reflectance,
)

rgb = apply_rgb_compo(
    rasters,
    preprocess=digital_numbers_to_reflectance,
    mask=cloud_mask,
)

print("Taille RGB :", rgb.shape)
print("Plage RGB :", rgb.min(), rgb.max())
print("Nombre de pixels nuageux :", cloud_mask.sum())

fig, (rgb_ax, mask_ax) = plt.subplots(1, 2, figsize=(12, 6))

rgb_ax.imshow(rgb)
rgb_ax.axis("off")
rgb_ax.set_title("Composition RGB")

mask_ax.imshow(
    cloud_mask,
    cmap=ListedColormap(["black", "red"]),
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
mask_ax.axis("off")
mask_ax.set_title("Masque nuage (True en rouge)")

fig.savefig("rgb_test1.png", dpi=500, bbox_inches="tight")
plt.close(fig)

for raster in rasters.values():
    raster.close()


