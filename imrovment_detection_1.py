"""Étape 6 : distinguer coupe rase et scolytes parmi les pixels dépérissants de 2018.

Entrée : detection_2018.npz (sauvegardé à la fin de main_CRswir.py)
Idée   : on regarde le NDVI 2018 SANS le masque de non-végétation
         - coupe rase : le NDVI s'effondre brutalement et reste bas (sol nu)
         - scolytes   : le NDVI reste assez haut, seul CRswir monte
"""
import os
import yaml

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from download_from_cdse import query_id
from spectral_index import apply_ndvi
from preprocess import digital_numbers_to_reflectance
from harmonic_regression import dates_to_days

# Seuils (à justifier / ajuster dans le rapport)
SEUIL_NDVI_FORET = 0.6      # NDVI avant détection au-dessus duquel le pixel était une forêt
SEUIL_NDVI_SOL_NU = 0.4     # méthode 1 : NDVI après détection en dessous duquel on considère le sol nu
SEUIL_CHUTE_REL = 0.2       # méthode 2 : chute de NDVI (avant - après) au-dessus de laquelle c'est une coupe


def masque_nuages(rasters, seuil=0.6):
    """Masque nuages / ombres SANS le critère de non-végétation (True = invalide)."""
    b = {k: digital_numbers_to_reflectance(rasters[k].read(1)) for k in ("B02", "B03", "B04", "B11")}
    masque = (b["B02"] > seuil) & (b["B03"] > seuil) & (b["B04"] > seuil) & (b["B11"] > seuil)
    if "SCL" in rasters:
        scl = rasters["SCL"].read(1)
        # 0 no data, 1 saturé, 3 ombre, 8/9 nuages, 10 cirrus, 11 neige
        masque |= np.isin(scl, (0, 1, 3, 8, 9, 10, 11))
    return masque


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Résultats de détection (étape 5)
    # ------------------------------------------------------------------
    det = np.load("detection_2018.npz")
    deperissement = det["deperissement"]          # (H, W) bool
    date_detection = det["date_detection"]        # (H, W) indice de date, -1 = aucune
    test_dates = list(det["test_dates"])          # même ordre que dans main_CRswir.py

    # ------------------------------------------------------------------
    # 2. NDVI 2018 + masque nuages (sans masque végétation)
    # ------------------------------------------------------------------
    with open("cdse_query.yaml") as f:
        query = yaml.safe_load(f)
    data_folder = os.path.join(query["data_folder"], query_id(query, temporal_id=False))
    test_folder = os.path.join(data_folder, [y for y in sorted(os.listdir(data_folder))
                                             if y.startswith("2018")][0])

    ndvi_list, mask_list = [], []
    for date in test_dates:
        rasters = {
            band: rasterio.open(os.path.join(test_folder, band, f"SENTINEL2_L2A_{band}_{date}Z.tif"))
            for band in ("B02", "B03", "B04", "B08", "B11", "SCL")
        }
        ndvi_list.append(apply_ndvi(rasters, preprocess=digital_numbers_to_reflectance))
        mask_list.append(masque_nuages(rasters))
        for r in rasters.values():
            r.close()

    ndvi = np.stack(ndvi_list)                    # (n, H, W)
    mask = np.stack(mask_list) | ~np.isfinite(ndvi)
    days = dates_to_days(np.array(test_dates, dtype="datetime64[ns]"))

    # ------------------------------------------------------------------
    # 3. Classification des pixels dépérissants
    #    0 = sain, 1 = scolytes, 2 = coupe rase
    #    Méthode 1 (classe)     : seuil absolu  -> NDVI après < SEUIL_NDVI_SOL_NU
    #    Méthode 2 (classe_rel) : chute relative -> NDVI avant - NDVI après > SEUIL_CHUTE_REL
    # ------------------------------------------------------------------
    classe = np.zeros(deperissement.shape, dtype=int)
    classe_rel = np.zeros(deperissement.shape, dtype=int)
    H, W = deperissement.shape

    for i in range(H):
        for j in range(W):
            if not deperissement[i, j]:
                continue
            valides = np.where(~mask[:, i, j])[0]
            serie = ndvi[valides, i, j]
            avant = valides < date_detection[i, j]
            apres = ~avant

            if avant.sum() == 0 or apres.sum() == 0:   # impossible de comparer
                classe[i, j] = 1
                classe_rel[i, j] = 1
                continue

            # Médianes : robustes aux quelques nuages mal masqués
            ndvi_avant = np.median(serie[avant])
            ndvi_apres = np.median(serie[apres])

            # Méthode 1 : seuil absolu
            if ndvi_avant > SEUIL_NDVI_FORET and ndvi_apres < SEUIL_NDVI_SOL_NU:
                classe[i, j] = 2                  # coupe rase
            else:
                classe[i, j] = 1                  # scolytes

            # Méthode 2 : chute relative
            if ndvi_avant > SEUIL_NDVI_FORET and (ndvi_avant - ndvi_apres) > SEUIL_CHUTE_REL:
                classe_rel[i, j] = 2              # coupe rase
            else:
                classe_rel[i, j] = 1              # scolytes

    n_tot = deperissement.sum()
    print(f"Pixels dépérissants : {n_tot}")
    for nom, c_arr in [("Méthode 1 (seuil absolu)", classe), ("Méthode 2 (chute relative)", classe_rel)]:
        print(nom)
        print(f"  scolytes   : {(c_arr == 1).sum()} ({100 * (c_arr == 1).sum() / n_tot:.1f} %)")
        print(f"  coupe rase : {(c_arr == 2).sum()} ({100 * (c_arr == 2).sum() / n_tot:.1f} %)")
    np.savez("classification_2018.npz", classe=classe, classe_rel=classe_rel)

    # ------------------------------------------------------------------
    # 4. Carte + exemples de séries NDVI
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(19, 5), gridspec_kw={"width_ratios": [1, 1, 1.6]})

    cmap = ListedColormap(["white", "darkorange", "purple"])
    axs[0].imshow(classe, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axs[0].set_title("Méthode 1 : seuil absolu\nscolytes (orange) / coupe rase (violet)")
    axs[0].axis("off")
    axs[1].imshow(classe_rel, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axs[1].set_title("Méthode 2 : chute relative\nscolytes (orange) / coupe rase (violet)")
    axs[1].axis("off")

    # Un exemple de chaque classe
    for c, couleur, nom in [(1, "darkorange", "scolytes"), (2, "purple", "coupe rase")]:
        idx = np.argwhere(classe_rel == c)
        if len(idx) == 0:
            continue
        i, j = idx[len(idx) // 2]
        ok = ~mask[:, i, j]
        axs[2].plot(days[ok], ndvi[ok, i, j], "o-", color=couleur, label=f"{nom} ({i},{j})")
        axs[2].axvline(days[date_detection[i, j]], color=couleur, linestyle="--")
    axs[2].set_ylabel("NDVI")
    axs[2].set_ylim(0, 1)
    xt = np.arange(len(test_dates))[::6]
    axs[2].set_xticks(days[xt], labels=np.array(test_dates)[xt], rotation=45)
    axs[2].legend()
    axs[2].set_title("NDVI 2018 (tirets = date de détection)")

    plt.savefig("Coupe_vs_scolytes_2018.png", bbox_inches="tight")
    plt.show()
