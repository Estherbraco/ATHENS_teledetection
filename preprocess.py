from typing import Dict, Callable

import numpy as np
import rasterio

from utils import rgb_render


def digital_numbers_to_reflectance(raster: np.ndarray):
    """Convert a Sentinel-2 raster provided in digital numbers to reflectance
    for data acquired from 2016 to 2020. In this period, the formula for every
    S2 bands is : REFLECTANCE = DIGITAL_NUMBERS / 10000.
    """
    raster = raster / 10000
    raster = np.clip(raster, a_min=0, a_max=None)
    return raster


def apply_cloud_mask(
    rasters: Dict[str, rasterio.io.DatasetReader], 
    preprocess: Callable = lambda x: x, 
    seuil: float = 0.1,
    seuil_vegetation: float = 0.02,
    eps: float = 1e-4,
    ):
    """Create a mask for clouds, cloud shadows and non-vegetation pixels.

    Args:
        rasters: dictionary with keys B01, ..., B12 mapping to rasterio.io.DatasetReader datasets
        preprocess: optional argument, e.g. to convert Digital Numbers in reflectance
        seuil: reflectance threshold used to identify bright pixels
        seuil_vegetation: minimum NDVI considered vegetation
        eps: small value added to the NDVI denominator

    Returns:
        A boolean array where True indicates a cloud, cloud shadow,
        non-vegetation or invalid/SCL pixel.
    """
    bands = ("B02","B04","B08", "B11")
    values = []

    for band in bands:
        if band not in rasters:
            raise KeyError(f"Band {band} is required to create the cloud mask")
        values.append(np.asarray(preprocess(rasters[band].read(1))))

    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("All raster bands must have the same shape")

    b02, b04, b08, b11 = values
    mask =(((b08 - b04) / (b08 + b04)) > 0.2) & (b02 < 0.12) & (b08 > 0.15) & (b11 > 0.08)
    mask = ~mask
    return mask

def apply_rgb_compo(
    rasters: Dict[str, rasterio.io.DatasetReader], 
    preprocess: Callable = lambda x: x, 
    mask: np.ndarray = None,
    render: bool = True):
    """Make a RGB composition from multispectral data.

    Args:
        rasters: dictionary with keys B01, ..., B12 mapping to rasterio.io.DatasetReader datasets
        preprocess: optional argument, e.g. to convert Digital Numbers in reflectance
        mask: boolean array of invalid pixels (True = invalid)
        render: if True, return an uint8 array stretched for display;
            if False, return the raw composition.
    """
    bands = ("B02", "B03", "B04")
    values = []

    for band in bands:
        if band not in rasters:
            raise KeyError(f"Band {band} is required to create the RGB composition")
        values.append(np.asarray(preprocess(rasters[band].read(1))))

    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("All raster bands must have the same shape")

    composition = np.stack(values, axis=0)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError("The mask must have the same shape as the raster bands")

    if render:
        rendered, _, _ = rgb_render(
            composition,
            bands=[2, 1, 0],
            mask=mask,
        )
        return rendered

    return composition