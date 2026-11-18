from typing import Dict, Callable


def apply_ndvi(raster_dst: Dict, preprocess: Callable = lambda x: x, eps: float = 1e-4):
    """Normalized Difference Vegetation Index.
    Formula for Sentinel-2 bands: (B08-B04)/(B08+B04)

    Args:
        raster_dst: dictionary with keys B01, ..., B12 mapping to rasterio.io.DatasetReader datasets
        preprocess: function applied to each raster array before calculating NDVI
        eps: small value added to the denominator to avoid division by zero

    Returns:
        an array of NDVI values.
    """
    B04 = preprocess(raster_dst["B04"].read(1))
    B08 = preprocess(raster_dst["B08"].read(1))
    
    return (B08 - B04) / (B08 + B04 + eps)
