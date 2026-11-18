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
    seuil: float = 0 
    ):
    """Cloud mask.
    
    Args:
        rasters: dictionary with keys B01, ..., B12 mapping to rasterio.io.DatasetReader datasets
        preprocess: optional argument, e.g. to convert Digital Numbers in reflectance
        seuil: reflectance threshold used to identify bright pixels

    Returns:
        A boolean array where True indicates a cloud pixel.
    """
    bands = ("B02", "B03", "B04", "B11")
    values = []

    for band in bands:
        if band not in rasters:
            raise KeyError(f"Band {band} is required to create the cloud mask")
        values.append(np.asarray(preprocess(rasters[band].read(1))))

    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("All raster bands must have the same shape")

    return np.logical_and.reduce([value > seuil for value in values])

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
    raise NotImplementedError