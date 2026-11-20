import warnings
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
    seuil_ndvi: float = 0.2,
    seuil_b02: float = 0.12,
    seuil_b08: float = 0.15,
    seuil_b11: float = 0.08,
    classes_scl: tuple = (0, 1, 3, 8, 9, 10, 11),
    eps: float = 1e-4,
    ):
    """Create a mask (one date) for clouds, cloud shadows and non-vegetation pixels.

    A pixel is VALID if all the following conditions hold:
        - NDVI > seuil_ndvi        (vegetation)
        - B02  < seuil_b02         (not a bright cloud / haze in the blue band)
        - if the SCL band is available: SCL not in classes_scl
        - otherwise (no SCL), shadow proxies:
            B08 >= seuil_b08 and B11 > seuil_b11
    SCL classes: 0 no data, 1 saturated, 3 cloud shadow, 8/9 medium/high
    probability clouds, 10 thin cirrus, 11 snow.

    Args:
        rasters: dictionary with keys B02, ..., B12 (and optionally SCL)
            mapping to rasterio.io.DatasetReader datasets
        preprocess: optional argument, e.g. to convert Digital Numbers in reflectance
        seuil_ndvi, seuil_b02, seuil_b08, seuil_b11: thresholds described above
        classes_scl: SCL classes considered invalid
        eps: small value added to the NDVI denominator

    Returns:
        A boolean array where True indicates an invalid pixel
        (cloud, cloud shadow, cirrus, non-vegetation, no data).
    """
    bands = ("B02", "B04", "B08", "B11")
    values = []

    for band in bands:
        if band not in rasters:
            raise KeyError(f"Band {band} is required to create the cloud mask")
        values.append(np.asarray(preprocess(rasters[band].read(1))))

    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("All raster bands must have the same shape")

    b02, b04, b08, b11 = values
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    valid = (ndvi > seuil_ndvi) & (b02 < seuil_b02)

    if "SCL" in rasters:
        # SCL detects shadows / clouds / cirrus
        scl = np.asarray(rasters["SCL"].read(1))
        if scl.shape != shape:
            raise ValueError("All raster bands must have the same shape")
        valid &= ~np.isin(scl, classes_scl)
    else:
        # No SCL: dark pixels in NIR / SWIR are considered as shadows.
        # (Not used with SCL: conifers are naturally dark in SWIR, B11 > 0.08
        # would remove ~40 % of their valid dates.)
        valid &= (b08 >= seuil_b08) & (b11 > seuil_b11)

    return ~valid


def compute_forest_mask(
    rasters_ts: Dict[str, Dict[str, rasterio.io.DatasetReader]],
    masks: np.ndarray,
    preprocess: Callable = lambda x: x,
    months: tuple = ("05", "06", "07", "08", "09"),
    seuil_b08: float = 0.33,
    seuil_b03: float = 0.06,
    ):
    """Static forest mask (one map for the whole time series).

    Meadows and crops have an NDVI as high as forests, so the NDVI cannot
    separate them. In summer, however, meadows are much brighter than
    (coniferous) forests in the NIR (B08) and in the green (B03).
    The criterion is computed on the median over the summer dates, so that
    a pixel is either forest on every date or on none.

    Args:
        rasters_ts: {date 'YYYY-MM-DD': {band: rasterio dataset}} (training period)
        masks: invalid pixels for these dates, shape (n_dates, H, W), same order as rasters_ts
        preprocess: e.g. digital_numbers_to_reflectance
        months: months used to compute the summer median
        seuil_b08, seuil_b03: a pixel is forest if median B08 < seuil_b08 and median B03 < seuil_b03

    Returns:
        A boolean array of shape (H, W), True = forest.
    """
    dates = list(rasters_ts)
    summer = [k for k, d in enumerate(dates) if d[5:7] in months]
    if not summer:
        raise ValueError("No summer date available to compute the forest mask")

    b08 = np.stack([np.where(masks[k], np.nan, preprocess(rasters_ts[dates[k]]["B08"].read(1)))
                    for k in summer])
    b03 = np.stack([np.where(masks[k], np.nan, preprocess(rasters_ts[dates[k]]["B03"].read(1)))
                    for k in summer])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # pixels never valid in summer
        med_b08 = np.nanmedian(b08, axis=0)
        med_b03 = np.nanmedian(b03, axis=0)
    # True = forest (NOT a "masked" pixel: to display what is excluded, use ~forest_mask)
    # NaN (never valid in summer) -> comparisons are False -> not forest
    forest_mask = (med_b08 < seuil_b08) & (med_b03 < seuil_b03)
    return forest_mask


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