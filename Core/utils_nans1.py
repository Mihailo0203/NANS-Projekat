# Sadrži metrike koje se koriste u run_models.py

from __future__ import annotations
import numpy as np

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

def adjusted_r2(y_true, y_pred, p: int) -> float:
    """
    Prilagođeni R^2.
    p = broj atributa (feature-a), bez intercepta.
    """
    y_true = np.asarray(y_true, dtype=float)
    n = y_true.shape[0]
    r2_val = r2(y_true, y_pred)
    if np.isnan(r2_val) or n <= p + 1:
        return float("nan")
    return 1.0 - (1.0 - r2_val) * (n - 1) / (n - p - 1)

# Backwards-compatible nazivi koje projekat koristi:
def get_rmse(model, features, labels):
    y_pred = model.predict(features)
    return rmse(labels, y_pred)

def get_rsquared_adj(model, features, labels):
    p = features.shape[1] if hasattr(features, "shape") else len(features[0])
    y_pred = model.predict(features)
    return adjusted_r2(labels, y_pred, p=p)
