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
    # ÉTAPE 7 : sécheresse vs scolytes — critère de NON-RETOUR à un CRswir normal
    # ==================================================================
    # Après la série d'anomalies qui a déclenché la détection, on regarde si le pixel
    # redevient normal (CRswir sous la borne haute de l'IP) pendant K_RETOUR dates
    # valides d'affilée :
    #   - oui                                  -> l'arbre récupère : sécheresse
    #   - non, avec au moins M_RECUL dates     -> pas de récupération : scolytes
    #   - non, moins de M_RECUL dates observées -> pas assez de recul : indéterminé
    K_RETOUR = 3      # dates normales d'affilée pour parler de "retour" (comme FORDEAD)
    M_RECUL = 6       # dates valides minimum après la détection pour conclure "pas de retour"
    normal = (CRswir_test <= pred_test["haut"]) & ~mask_test

    classe_retour = np.zeros((H, W), dtype=int)   # 0 sain, 1 scolytes, 2 sécheresse, 3 indéterminé
    date_retour = np.full((H, W), -1, dtype=int)  # indice de la 1re date du retour (-1 = aucun)
    for i in range(H):
        for j in range(W):
            if not deperissement[i, j]:
                continue
            valides = np.where(~mask_test[:, i, j])[0]
            # on saute les N_CONSEC dates valides de la série de détection
            pos = np.searchsorted(valides, date_detection[i, j])
            apres = valides[pos + N_CONSEC:]
            compteur = 0
            for k in apres:
                if normal[k, i, j]:
                    compteur += 1
                    if compteur == 1:
                        debut_retour = k
                    if compteur == K_RETOUR:
                        date_retour[i, j] = debut_retour
                        break
                else:
                    compteur = 0
            if date_retour[i, j] >= 0:
                classe_retour[i, j] = 2                 # retour à la normale -> sécheresse
            elif len(apres) >= M_RECUL:
                classe_retour[i, j] = 1                 # jamais revenu malgré le recul -> scolytes
            else:
                classe_retour[i, j] = 3                 # pas assez de recul -> indéterminé

    n = deperissement.sum()
    print(f"[Étape 7] dépérissants : {n}")
    for c, nom in [(1, "scolytes (pas de retour)"), (2, "sécheresse (retour à la normale)"),
                   (3, "indéterminé (pas assez de recul)")]:
        print(f"  {nom:35s}: {(classe_retour == c).sum():5d} ({100 * (classe_retour == c).sum() / n:.0f} %)")

    np.savez("secheresse_2018.npz", classe_retour=classe_retour, date_retour=date_retour,
             test_dates=np.array(test_dates))

    # Carte + un exemple de série par classe
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.8]})
    cmap7 = ListedColormap(["white", "darkorange", "royalblue", "lightgrey"])
    axs[0].imshow(classe_retour, cmap=cmap7, vmin=0, vmax=3, interpolation="nearest")
    axs[0].set_title("Non-retour à un CRswir normal\nscolytes (orange) / sécheresse (bleu) / indéterminé (gris)")
    axs[0].axis("off")
    for c, couleur, nom in [(1, "darkorange", "scolytes"), (2, "royalblue", "sécheresse")]:
        idx = np.argwhere(classe_retour == c)
        if len(idx) == 0:
            continue
        i, j = idx[len(idx) // 2]
        ok = ~mask_test[:, i, j]
        axs[1].plot(days_test[ok], CRswir_test[ok, i, j] - pred_test["haut"][ok, i, j], "o-",
                    color=couleur, label=f"{nom} ({i},{j})")
        axs[1].axvline(days_test[date_detection[i, j]], color=couleur, linestyle="--")
        if date_retour[i, j] >= 0:
            axs[1].axvline(days_test[date_retour[i, j]], color=couleur, linestyle=":")
    axs[1].axhline(0, color="grey", lw=0.8)
    axs[1].set_ylabel("CRswir − borne haute de l'IP")
    xt = np.arange(len(test_dates))[::6]
    axs[1].set_xticks(days_test[xt], labels=np.array(test_dates)[xt], rotation=45)
    axs[1].legend()
    axs[1].set_title("> 0 = anomalie ; tirets = détection ; pointillés = retour à la normale")
    plt.savefig("Secheresse_vs_scolytes_2018.png", bbox_inches="tight")
    plt.show()
