from typing import Dict, Callable

WAVELENGHTS = {
    "B02": 490,
    "B03": 560,
    "B04": 665,
    "B05": 705,
    "B06": 740,
    "B07": 783,
    "B08": 842,
    "B8A": 865,
    "B11": 1610,
    "B12": 2190
}

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


def apply_CRswir(raster_dst: Dict, preprocess: Callable = lambda x: x):
    """Continuum Removal in Short-Wave InfraRed
    Formula for Sentinel-2 bands: 

    Args:
        raster_dst: dictionary with keys B01, ..., B12 mapping to rasterio.io.DatasetReader datasets
        preprocess: function applied to each raster array before calculating CRswir

    Returns:
        an array of CRswir values.
    """
    rho_swir1 = preprocess(raster_dst["B11"].read(1))
    rho_swir2 = preprocess(raster_dst["B12"].read(1))
    rho_nir2 = preprocess(raster_dst["B8A"].read(1))
    lamb_swir1 = WAVELENGHTS["B11"]
    lamb_swir2 = WAVELENGHTS["B12"]
    lamb_nir2 = WAVELENGHTS["B8A"]

    rho_cont = rho_nir2 + (lamb_swir1 - lamb_nir2)*(rho_swir2 - rho_nir2)/(lamb_swir2 - lamb_nir2)

    return rho_swir1/rho_cont

