import numpy as np
from typing import Dict
from pandas import to_datetime
from scipy import stats

REF_DAY = to_datetime("2015-01-01").to_datetime64()
T=365.25

def fit_harmonic_model(
    ts_days: np.ndarray, 
    ts_values: np.ndarray,
    ts_masks: np.ndarray):
    """
    Models periodic vegetation index.

    Args:
        ts_days: array of days, encoded in integer value from a reference date, of shape (n_days,)
        ts_values: array of the time series values in floats, of shape (n_days,)
        ts_masks: array of invalid pixels, of shape (n_days,)

    Returns:
        dictionary containing:
            - a key 'coeff' mapping to the coefficients of the linear model
            - a key 'std_hat' mapping to the estimated standard deviation of the residual errors
            - a key 'XTX' mapping to the Gram matrix of the matrix of input variables
            - a key 'df' mapping to the degrees of freedom of the estimated model
    """
    y=ts_values[~ts_masks]
    t=ts_days[~ts_masks]
    X=compute_harmonic_terms(ts_days, T)
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeff
    df =X.shape[0] - X.shape[1]
    std_hat = np.sqrt(np.sum(residuals**2)/df)
    XTX = X.T @ X
    return {
        "coeff": coeff,
        "std_hat": std_hat,
        "XTX": XTX,
        "df": df
    }


def pred_harmonic_model(
    ts_days: np.ndarray,
    harmonic_model: Dict,
    confidence_level: float = 0.95,
    reshape_spatial_dims: bool = True
    ):
    """
    Prediction of a vegetation index by a harmonic model.

    Args:
        ts_days: array of days, encoded in integer value from a reference date, of shape (n_days,)
        harmonic_model: estimated model in a dictionary format, as given by fit_harmonic_model
        confidence_level: level of confidence for the confidence and prediction intervals

    Returns:
        dictionary containing:
            - a key 'ts_hat' mapping to the predictions
            - a key 'prediction_interval' mapping to the prediction interval of new observations
    """
    alpha = 1 - confidence_level  
    X = compute_harmonic_terms(ts_days, T)
    ts_hat = X @ harmonic_model["coeff"] 
    XTX_inv = np.linalg.inv(harmonic_model["XTX"])
    q = stats.t.ppf(1 - alpha/2, df=harmonic_model["df"])     
    h = np.sum((X @ XTX_inv) * X, axis=1)        
    hw = q * harmonic_model["std_hat"] * np.sqrt(1 + h) 
    prediction_interval = np.column_stack([ts_hat - hw, ts_hat + hw])
    return {"ts_hat": ts_hat, "prediction_interval": prediction_interval}
    raise NotImplementedError


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
        # https://github.com/pandas-dev/pandas/blob/8cf4ab491b06274f555d1ea305c172354208747a/doc/source/whatsnew/v2.0.0.rst#disallow-astype-conversion-to-non-supported-datetime64timedelta64-dtypes
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
        Array of shape (5,) containing harmonic terms.
    """
    return np.array([1, np.cos(2 * np.pi * DateAsNumber / T), np.sin(2 * np.pi * DateAsNumber / T),
                     np.cos(4 * np.pi * DateAsNumber / T), np.sin(4 * np.pi * DateAsNumber / T)])
    raise NotImplementedError