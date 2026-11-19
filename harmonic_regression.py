import numpy as np
from typing import Dict
from pandas import to_datetime
from scipy import stats

REF_DAY = to_datetime("2015-01-01").to_datetime64()
T = 365.25


# ---------------------------------------------------------------------------
# Version 1 pixel 
# ---------------------------------------------------------------------------

def _fit_un_pixel(
    ts_days: np.ndarray,
    ts_values: np.ndarray,
    ts_masks: np.ndarray):
    """
    Models periodic vegetation index for ONE pixel.

    Args:
        ts_days: array of days, encoded in integer value from a reference date, of shape (n_days,)
        ts_values: array of the time series values in floats, of shape (n_days,)
        ts_masks: array of invalid pixels, of shape (n_days,)

    Returns:
        dictionary with keys 'coeff', 'std_hat', 'XTX', 'df'
    """
    y = ts_values[~ts_masks]
    t = ts_days[~ts_masks]
    X = compute_harmonic_terms(t, T)
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeff
    df = X.shape[0] - X.shape[1]
    std_hat = np.sqrt(np.sum(residuals**2) / df)
    XTX = X.T @ X
    return {
        "coeff": coeff,
        "std_hat": std_hat,
        "XTX": XTX,
        "df": df
    }


def _pred_un_pixel(
    ts_days: np.ndarray,
    harmonic_model: Dict,
    confidence_level: float = 0.95):
    """
    Prediction of a vegetation index by a harmonic model, for ONE pixel.

    Returns:
        dictionary with keys 'ts_hat' (n_days,) and 'prediction_interval' (n_days, 2)
    """
    alpha = 1 - confidence_level
    X = compute_harmonic_terms(ts_days, T)
    ts_hat = X @ harmonic_model["coeff"]
    XTX_inv = np.linalg.inv(harmonic_model["XTX"])
    q = stats.t.ppf(1 - alpha / 2, df=harmonic_model["df"])
    h = np.sum((X @ XTX_inv) * X, axis=1)
    hw = q * harmonic_model["std_hat"] * np.sqrt(1 + h)
    prediction_interval = np.column_stack([ts_hat - hw, ts_hat + hw])
    return {"ts_hat": ts_hat, "prediction_interval": prediction_interval}


# ---------------------------------------------------------------------------
# Version image : boucle sur tous les pixels (on pourrait vectoriser mais c'est plus compliqué et on a pas 2 semaines de +)
# ---------------------------------------------------------------------------

def fit_harmonic_model(
    ts_days: np.ndarray,
    ts_values: np.ndarray,
    ts_masks: np.ndarray,
    min_obs: int = 10):
    """
    Fits the harmonic model on every pixel of an image time series.

    Args:
        ts_days: days since REF_DAY, of shape (n_days,)
        ts_values: vegetation index, of shape (n_days, H, W)
        ts_masks: invalid pixels (True = invalid), of shape (n_days, H, W)
        min_obs: minimum number of valid dates to fit a pixel

    Returns:
        dictionary containing, for every pixel:
            - 'coeff': (H, W, 5), NaN if the pixel was not fitted
            - 'std_hat': (H, W)
            - 'XTX': (H, W, 5, 5)
            - 'df': (H, W)
    """
    n_days, H, W = ts_values.shape
    n_terms = 5

    coeff = np.full((H, W, n_terms), np.nan)
    std_hat = np.full((H, W), np.nan)
    XTX = np.full((H, W, n_terms, n_terms), np.nan)
    df = np.zeros((H, W), dtype=int)

    for i in range(H):
        for j in range(W):
            masque = ts_masks[:, i, j] | ~np.isfinite(ts_values[:, i, j])
            if np.sum(~masque) < min_obs:
                continue
            m = _fit_un_pixel(ts_days, ts_values[:, i, j], masque)
            coeff[i, j] = m["coeff"]
            std_hat[i, j] = m["std_hat"]
            XTX[i, j] = m["XTX"]
            df[i, j] = m["df"]

    return {"coeff": coeff, "std_hat": std_hat, "XTX": XTX, "df": df}


def pred_harmonic_model(
    ts_days: np.ndarray,
    harmonic_model: Dict,
    confidence_level: float = 0.95):
    """
    Predicts the vegetation index on every pixel.

    Args:
        ts_days: days since REF_DAY, of shape (n_days,)
        harmonic_model: model given by fit_harmonic_model
        confidence_level: level of confidence for the prediction interval

    Returns:
        dictionary containing:
            - 'ts_hat': predictions, of shape (n_days, H, W)
            - 'bas': lower bound of the prediction interval, (n_days, H, W)
            - 'haut': upper bound of the prediction interval, (n_days, H, W)
    """
    H, W = harmonic_model["std_hat"].shape
    n = len(ts_days)

    ts_hat = np.full((n, H, W), np.nan)
    bas = np.full((n, H, W), np.nan)
    haut = np.full((n, H, W), np.nan)

    for i in range(H):
        for j in range(W):
            if np.isnan(harmonic_model["coeff"][i, j, 0]):
                continue
            m_px = {
                "coeff": harmonic_model["coeff"][i, j],
                "std_hat": harmonic_model["std_hat"][i, j],
                "XTX": harmonic_model["XTX"][i, j],
                "df": harmonic_model["df"][i, j],
            }
            r = _pred_un_pixel(ts_days, m_px, confidence_level)
            ts_hat[:, i, j] = r["ts_hat"]
            bas[:, i, j] = r["prediction_interval"][:, 0]
            haut[:, i, j] = r["prediction_interval"][:, 1]

    return {"ts_hat": ts_hat, "bas": bas, "haut": haut}


# ---------------------------------------------------------------------------
# fonctions utiles 
# ---------------------------------------------------------------------------

def dates_to_days(dates):
    """
    Converts datetime objects to days since reference date.

    Args:
        dates : array-like of datetime-like
        Input dates to convert.

    Returns:
        Array of integers representing days since REF_DAY (2015-01-01).
    """
    try:
        days = (dates - REF_DAY).astype("timedelta64[D]").astype(int)
    except ValueError as e:
        # .astype("timedelta64[D]") not supported anymore by pandas series
        # https://stackoverflow.com/a/76363956
        pd_msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
        if pd_msg in str(e):
            days = (dates - REF_DAY).days
        else:
            raise e

    return days


def compute_harmonic_terms(DateAsNumber: int, T: float = 365.25):
    """
    Computes harmonic terms for a given date.

    Args:
        DateAsNumber : number of days since reference date (2015-01-01).
        T: periodicity of the signal to model.

    Returns:
        Array of shape (..., 5) containing harmonic terms.
    """
    d = np.asarray(DateAsNumber, dtype=float)
    w = 2 * np.pi * d / T
    return np.stack([np.ones_like(d), np.cos(w), np.sin(w),
                     np.cos(2 * w), np.sin(2 * w)], axis=-1)