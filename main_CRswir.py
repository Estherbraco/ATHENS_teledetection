import os
import yaml
import json

import rasterio
import matplotlib.pyplot as plt

from download_from_cdse import query_id
from spectral_index import apply_CRswir
from preprocess import *
from harmonic_regression import *
from datetime import datetime

if __name__ == "__main__":
    with open('cdse_query.yaml', 'r') as file:
        query = yaml.safe_load(file)

    bands = query['bands'][:-1]
    
    print(bands)
    data_folder = os.path.join(query['data_folder'], query_id(query, temporal_id=False))
    years = sorted(os.listdir(data_folder))

    train_data_folder = os.path.join(data_folder, years[0])

    # Load Sentinel-2 data
    with open(os.path.join(train_data_folder, bands[0], 'job-results.json'), 'r') as f:
        job_results = json.load(f)

    img_files = list(job_results['assets'].keys())
    img_dates = [file.split('_')[-1].split('Z')[0] for file in img_files]

    # Make Sentinel-2 time series
    train_ts = {}
    for date in img_dates:
        train_ts[date] = dict((
            band, 
            rasterio.open(os.path.join(train_data_folder, band, f'SENTINEL2_L2A_{band}_{date}Z.tif'))) 
            for band in bands)
        train_ts[date]['CRswir'] = apply_CRswir(train_ts[date], preprocess=digital_numbers_to_reflectance)
    print(train_ts)

    dates = np.array(list(train_ts.keys()))
    n_dates = len(dates)
    dates_for_days = np.array(list(train_ts), dtype="datetime64[ns]")
    days = dates_to_days(dates_for_days)
    # Make a raster of size (n_days, height, width) of CRswir
    CRswir_array = np.array([train_ts[date]['CRswir'] for date in train_ts])
    height, width = CRswir_array.shape[1], CRswir_array.shape[2]
    px_coords = height // 3, width // 3

    # Plot the CRswir time series    
    fig, ax = plt.subplots()
    fontsize = 15
    CRswir_px = CRswir_array[:, px_coords[0], px_coords[1]]
    ax.scatter(days, CRswir_px, color="#8fc0a9")
    ax.set_ylabel("CRswir", fontsize=fontsize)
    #ax.set_ylim([0, 2])
    xticks = np.arange(len(dates))[::6]
    xticklabels = dates[xticks]
    ax.set_xticks(days[xticks], labels=xticklabels, rotation=45)
    ax.xaxis.grid(True, which="major", linestyle="dashed")
    plt.savefig("CRswir_timeseries.png", bbox_inches="tight")

    # Plot the time series
    #n_rows, n_cols = 2, 4
    #figsize = (150, 150)
    #dates = list(train_ts.keys())
    #n_dates = len(img_dates)
    #for i in range(n_dates // (n_rows * n_cols)):    
    #fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    #    for i, (ax, date) in enumerate(zip(axs.flat, dates[i*n_rows*n_cols:(i+1)*n_rows*n_cols])):
     #       ax.imshow(train_ts[date]['RGB'])
      #      ax.set_xticks([])
       #     ax.set_yticks([])
        #    ax.set_title(date)
       # plt.show()

cloud_mask_array = np.stack([
    apply_cloud_mask(rasters, preprocess=digital_numbers_to_reflectance)
    for rasters in train_ts.values()
])
mask_px = cloud_mask_array[:, px_coords[0], px_coords[1]] | ~np.isfinite(CRswir_px)
if np.count_nonzero(~mask_px) > 5:
    model = fit_harmonic_model(days, CRswir_px, mask_px)
    result = pred_harmonic_model(days, model)

    ax.plot(days, result["ts_hat"], color="green", label="Régression harmonique")
    ax.legend()
else:
    print("Pas assez d'observations valides pour ce pixel.")
    

ax.plot(days, result["ts_hat"], color="green", label="Régression harmonique")
plt.savefig("Regression_CRswir.png", bbox_inches="tight")
ax.legend()