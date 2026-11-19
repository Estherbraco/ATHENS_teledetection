import os
import yaml
import json

import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from download_from_cdse import query_id
from spectral_index import apply_CRswir
from preprocess import *
from harmonic_regression import *
from datetime import datetime

if __name__ == "__main__":
    with open('cdse_query.yaml', 'r') as file:
        query = yaml.safe_load(file)

    bands = query['bands']          # inclut SCL (utilisé par le masque nuages)
    TRAIN_YEAR = "2016"
    
    data_folder = os.path.join(query['data_folder'], query_id(query, temporal_id=False))
    years = sorted(os.listdir(data_folder))
    print(years)
    train_data_folder = os.path.join(data_folder, years[0])

    # Load Sentinel-2 data
    with open(os.path.join(train_data_folder, bands[0], 'job-results.json'), 'r') as f:
        job_results = json.load(f)

    img_files = list(job_results['assets'].keys())
    img_dates = [file.split('_')[-1].split('Z')[0] for file in img_files]

    # Entraînement sur 2016 uniquement (SCL disponible pour toutes les dates 2016)
    img_dates = sorted(d for d in img_dates if d.startswith(TRAIN_YEAR))
    print(f"{len(img_dates)} dates d'entraînement en {TRAIN_YEAR}")

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

    # Masque nuages / ombres / non-végétation (n_dates, H, W)
    cloud_mask_array = np.stack([
        apply_cloud_mask(train_ts[date], preprocess=digital_numbers_to_reflectance)
        for date in train_ts
    ])
    cloud_mask_array |= ~np.isfinite(CRswir_array)

    # Masque forêt FIXE (H, W) : médiane d'été 2016 de B08 et B03
    # (les prairies ont le même NDVI que la forêt mais sont plus claires en B08 / B03)
    foret = compute_forest_mask(train_ts, cloud_mask_array, preprocess=digital_numbers_to_reflectance)
    print(f"Forêt : {100 * foret.mean():.1f} % de la zone")

    # Régression harmonique sur tous les pixels (2016)
    model = fit_harmonic_model(days, CRswir_array, cloud_mask_array)
    pred = pred_harmonic_model(days, model)
    np.savez("model_2016_CRswir.npz", **model)

    # Courbe sur le pixel CRswir_px (2016)
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
    test_data_folder = os.path.join(data_folder, [y for y in years if y.startswith("2018")][0])

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

    # Anomalies : observation valide, en forêt, au-dessus de l'intervalle de prédiction
    anomalie = (CRswir_test > pred_test["haut"]) & ~mask_test & foret

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
    xticks_test = np.arange(len(test_dates))[::6]
    ax.set_xticks(days_test[xticks_test], labels=np.array(test_dates)[xticks_test], rotation=45)
    ax.xaxis.grid(True, which="major", linestyle="dashed")
    ax.legend()
    plt.savefig("Test_2018_CRswir.png", bbox_inches="tight")

    # ==================================================================
    # Confirmation : 3 anomalies consécutives sur les dates valides
    # ==================================================================
    N_CONSEC = 3
    n_test, H, W = anomalie.shape

    deperissement = np.zeros((H, W), dtype=bool)      
    date_detection = np.full((H, W), -1, dtype=int)   
    for i in range(H):
        for j in range(W):
            valides = np.where(~mask_test[:, i, j])[0]    
            compteur = 0
            for k in valides:
                if anomalie[k, i, j]:
                    compteur += 1
                    if compteur == 1:
                        debut = k                         
                    if compteur == N_CONSEC:
                        deperissement[i, j] = True
                        date_detection[i, j] = debut
                        break
                else:
                    compteur = 0                          

    print(f"Pixels dépérissants : {deperissement.sum()} "
          f"({100 * deperissement.mean():.1f} % de la zone)")

    # ==================================================================
    # Cartes
    # ==================================================================
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].imshow(deperissement, cmap="Reds")
    axs[0].set_title("Pixels dépérissants (2018)")
    axs[0].axis("off")

    carte_dates = np.where(deperissement, days_test[date_detection], np.nan)
    im = axs[1].imshow(carte_dates, cmap="viridis")
    axs[1].set_title("Date de détection")
    axs[1].axis("off")
    cbar = fig.colorbar(im, ax=axs[1], shrink=0.7)
    ticks = days_test[::max(1, len(days_test) // 6)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(np.array(test_dates)[::max(1, len(days_test) // 6)])

    plt.savefig("Deperissement_2018.png", bbox_inches="tight")
    plt.show()

    np.savez("detection_2018.npz", deperissement=deperissement,
             date_detection=date_detection, test_dates=np.array(test_dates), foret=foret)

    # ==================================================================
    # ÉTAPE 7 : sécheresse vs scolytes
    # ==================================================================
    # Score d'anomalie normalisé : de combien d'écarts-types l'obs. dépasse la prédiction
    z = (CRswir_test - pred_test["ts_hat"]) / model["std_hat"]      # (n, H, W)
    z = np.where(mask_test, np.nan, z)

    # --- Critère 1 : anomalie synchrone sur toute la zone (signature sécheresse)
    z_zone = np.nanmedian(z, axis=(1, 2))                           # (n,) une valeur par date
    frac_zone = anomalie.sum(axis=(1, 2)) / (~mask_test).sum(axis=(1, 2))

    # --- Critère 2 : persistance (scolytes = pas de retour à la normale)
    N_FIN = 3        # nb de dernières dates valides regardées
    classe7 = np.zeros((H, W), dtype=int)     # 0 sain, 1 scolytes, 2 sécheresse
    for i in range(H):
        for j in range(W):
            if not deperissement[i, j]:
                continue
            valides = np.where(~mask_test[:, i, j])[0]
            apres = valides[valides > date_detection[i, j]]
            fin = apres[-N_FIN:]
            if len(fin) < N_FIN:
                classe7[i, j] = 1                       # pas assez de recul : on garde scolytes
            elif anomalie[fin, i, j].any():
                classe7[i, j] = 1                       # toujours anormal en fin d'année -> scolytes
            else:
                classe7[i, j] = 2                       # revenu dans l'intervalle -> sécheresse

    # --- Critère 3 : détection après correction de l'anomalie régionale
    z_local = z - z_zone[:, None, None]                 # on retire l'effet "toute la zone"
    q = 2.0                                             # ~ borne de l'IP 95 %
    anomalie_locale = (z_local > q) & ~mask_test & foret
    deperissement_local = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            compteur = 0
            for k in np.where(~mask_test[:, i, j])[0]:
                compteur = compteur + 1 if anomalie_locale[k, i, j] else 0
                if compteur == N_CONSEC:
                    deperissement_local[i, j] = True
                    break

    n = deperissement.sum()
    print(f"[Étape 7] dépérissants : {n}")
    print(f"  critère 2 persistance -> scolytes {(classe7 == 1).sum()} | sécheresse {(classe7 == 2).sum()}")
    print(f"  critère 3 correction régionale -> restent {(deperissement & deperissement_local).sum()} "
          f"(les {(deperissement & ~deperissement_local).sum()} autres = effet zone / sécheresse)")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    axs[0].plot(days_test, z_zone, "o-", color="tab:red", label="z médian de la zone")
    axs[0].plot(days_test, 10 * frac_zone, "s--", color="tab:blue", label="10 × part de pixels en anomalie")
    axs[0].axhline(0, color="grey", lw=0.8)
    xt = np.arange(len(test_dates))[::6]
    axs[0].set_xticks(days_test[xt], labels=np.array(test_dates)[xt], rotation=45)
    axs[0].legend(); axs[0].set_title("Anomalie à l'échelle de la zone (2018)")
    cmap7 = ListedColormap(["white", "darkorange", "royalblue"])
    axs[1].imshow(classe7, cmap=cmap7, vmin=0, vmax=2, interpolation="nearest")
    axs[1].set_title("Persistance\nscolytes (orange) / sécheresse (bleu)"); axs[1].axis("off")
    axs[2].imshow(deperissement_local, cmap="Reds", interpolation="nearest")
    axs[2].set_title("Détection après correction régionale"); axs[2].axis("off")
    plt.savefig("Secheresse_vs_scolytes_2018.png", bbox_inches="tight")
    plt.show()
