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
    """
    bands: Dict[str, np.ndarray] = {
        name: preprocess(ds.read(1)).astype(np.float32)
        for name, ds in rasters.items()
    }

    b02 = bands["B02"]  # Bleu
    b03 = bands["B03"]  # Vert
    b04 = bands["B04"]  # Rouge
    b11 = bands["B11"]  # SWIR 1

    # 1. Forte réflectance dans le visible
    visible_brightness = (b02 + b03 + b04) / 3.0

    # 2. Distinction nuages / neige : les nuages restent réfléchissants dans le SWIR (B11),
    # alors que la neige absorbe fortement dans le SWIR (NDSI élevé).
    ndsi = (b03 - b11) / (b03 + b11 + 1e-8)

    cloud_mask = (
        (visible_brightness > seuil)
        & (b11 > 0.5 * seuil)
        & (ndsi < 0.4)
    )

    # Application du masque sur l'ensemble des bandes
    masked_rasters: Dict[str, np.ndarray] = {
        name: np.where(cloud_mask, np.nan, arr)
        for name, arr in bands.items()
    }
    return cloud_mask

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