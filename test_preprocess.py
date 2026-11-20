
import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from preprocess import (
    apply_rgb_compo,
    apply_cloud_mask,
    compute_forest_mask,
    digital_numbers_to_reflectance,
)

folder = "data/SENTINEL2_L2A/west=6.547121-south=47.910627-east=6.572061-north=47.93576/2016-01-01-2018-01-01"
date = "2016-09-22"
bands = ("B02", "B03", "B04", "B08", "B11")

prefix = "SENTINEL2_L2A_B02_"
dates = sorted(
    filename[len(prefix):-5]
    for filename in os.listdir(os.path.join(folder, "B02"))
    if filename.startswith(prefix) and filename.endswith("Z.tif")
)
dates = [
    image_date
    for image_date in dates
    if image_date.startswith("2016-")
    and all(
        os.path.isfile(
            os.path.join(
                folder, band, f"SENTINEL2_L2A_{band}_{image_date}Z.tif"
            )
        )
        for band in bands
    )
]
if date not in dates:
    raise FileNotFoundError(f"No complete raster set found for {date}")

rasters_ts = {
    image_date: {
        band: rasterio.open(
            os.path.join(
                folder, band, f"SENTINEL2_L2A_{band}_{image_date}Z.tif"
            )
        )
        for band in bands
    }
    for image_date in dates
}
cloud_masks = np.stack([
    apply_cloud_mask(
        rasters_ts[image_date],
        preprocess=digital_numbers_to_reflectance,
    )
    for image_date in dates
])
forest_mask = compute_forest_mask(
    rasters_ts,
    cloud_masks,
    preprocess=digital_numbers_to_reflectance,
)

rasters = rasters_ts[date]
cloud_mask = cloud_masks[dates.index(date)]
rgb = apply_rgb_compo(
    rasters,
    preprocess=digital_numbers_to_reflectance,
    mask=cloud_mask,
)

print("Taille RGB :", rgb.shape)
print("Plage RGB :", rgb.min(), rgb.max())
print("Nombre de pixels nuageux :", cloud_mask.sum())
print("Nombre de pixels forestiers :", forest_mask.sum())

fig, (rgb_ax, forest_ax, cloud_ax) = plt.subplots(1, 3, figsize=(18, 6))

rgb_ax.imshow(rgb)
rgb_ax.axis("off")
rgb_ax.set_title(f"Composition RGB ({date})")

binary_cmap = ListedColormap(["black", "red"])
forest_ax.imshow(
    forest_mask,
    cmap=binary_cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
forest_ax.axis("off")
forest_ax.set_title("Masque forestier (forêt en rouge)")

cloud_ax.imshow(
    cloud_mask,
    cmap=binary_cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
cloud_ax.axis("off")
cloud_ax.set_title("Masque cloud (True en rouge)")

fig.savefig("rgb_test1.png", dpi=500, bbox_inches="tight")
plt.close(fig)

for rasters_by_date in rasters_ts.values():
    for raster in rasters_by_date.values():
        raster.close()


