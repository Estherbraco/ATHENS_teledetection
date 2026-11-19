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
    
    data_folder = os.path.join(query['data_folder'], query_id(query, temporal_id=False))
    years = sorted(os.listdir(data_folder))
    print(years)
    train_data_folder = os.path.join(data_folder, years[0])

    # Load Sentinel-2 data
    with open(os.path.join(train_data_folder, bands[0], 'job-results.json'), 'r') as f:
        job_results = json.load(f)

    img_files = list(job_results['assets'].keys())
    img_dates = [file.split('_')[-1].split('Z')[0] for file in img_files]

    # img_dates [date for date in img_dates if '2016' in date]

    # Make Sentinel-2 time series
    train_ts = {}
    for date in img_dates:
        train_ts[date] = dict((
            band, 
            rasterio.open(os.path.join(train_data_folder, band, f'SENTINEL2_L2A_{band}_{date}Z.tif'))) 
            for band in bands)
        train_ts[date]['CRswir'] = apply_CRswir(train_ts[date], preprocess=digital_numbers_to_reflectance)

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
    #        ax.imshow(train_ts[date]['RGB'])
    #        ax.set_xticks([])
    #        ax.set_yticks([])
    #        ax.set_title(date)
    #    plt.show()

# Masque (n_dates, H, W)
    cloud_mask_array = np.stack([
        apply_cloud_mask(train_ts[date], preprocess=digital_numbers_to_reflectance)
        for date in train_ts
    ])
    cloud_mask_array |= ~np.isfinite(CRswir_array)

    # Régression harmonique sur tous les pixels 2016-1018
    model = fit_harmonic_model(days, CRswir_array, cloud_mask_array)
    pred = pred_harmonic_model(days, model)
    np.savez("model_2016_CRswir.npz", **model)

    # Courbe sur le pixel  Rswir_px 2016-1018
    i, j = px_coords
    ax.plot(days, pred["ts_hat"][:, i, j], color="green", label="Régression harmonique")
    ax.fill_between(days, pred["bas"][:, i, j], pred["haut"][:, i, j],
                    color="green", alpha=0.2, label="IP 95 %")
    ax.legend()
    plt.savefig("Regression_CRswir.png", bbox_inches="tight")
    plt.show()

    # ==================================================================
    # TEST : mêmes calculs sur 2018, avec le modèle ajusté sur 2016
    # ==================================================================
    test_data_folder = os.path.join(data_folder, years[1])
    

    # Dates 2018 (même méthode que pour le train)
    with open(os.path.join(test_data_folder, bands[0], 'job-results.json'), 'r') as f:
        job_results_test = json.load(f)
    test_dates = [file.split('_')[-1].split('Z')[0] for file in job_results_test['assets'].keys()]
    test_dates = sorted(d for d in test_dates if d.startswith("2018"))

    # Série temporelle 2018
    test_ts = {}
    for date in test_dates:
        test_ts[date] = {
            band: rasterio.open(os.path.join(test_data_folder, band, f'SENTINEL2_L2A_{band}_{date}Z.tif'))
            for band in bands
        }
        test_ts[date]['CRswir'] = apply_CRswir(test_ts[date], preprocess=digital_numbers_to_reflectance)

    # Jours, cube CRswir et masque 2018
    days_test = dates_to_days(np.array(test_dates, dtype="datetime64[ns]"))
    CRswir_test = np.array([test_ts[date]['CRswir'] for date in test_ts])
    mask_test = np.stack([
        apply_cloud_mask(test_ts[date], preprocess=digital_numbers_to_reflectance)
        for date in test_ts
    ])
    mask_test |= ~np.isfinite(CRswir_test)

    # Prédiction 2018 avec le modèle 2016 (pas de nouvel ajustement !)
    pred_test = pred_harmonic_model(days_test, model)

    # Anomalies : observation valide au-dessus de l'intervalle de prédiction
    anomalie = (CRswir_test > pred_test["haut"]) & ~mask_test

    # Figure : même pixel qu'en 2016
    i, j = px_coords
    valid = ~mask_test[:, i, j]
    fig, ax = plt.subplots()
    ax.scatter(days_test[valid], CRswir_test[valid, i, j], color="#8fc0a9", label="obs. 2018")
    ax.scatter(days_test[anomalie[:, i, j]], CRswir_test[anomalie[:, i, j], i, j],
               color="red", label="anomalie")
    ax.plot(days_test, pred_test["ts_hat"][:, i, j], color="green", label="modèle 2016")
    ax.fill_between(days_test, pred_test["bas"][:, i, j], pred_test["haut"][:, i, j],
                    color="green", alpha=0.2, label="IP 95 %")
    ax.set_ylabel("CRswir")
    ax.legend()
    plt.savefig("Test_2018_CRswir.png", bbox_inches="tight")
    plt.show()

