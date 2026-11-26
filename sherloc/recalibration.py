"""Out-of-fold post-hoc recalibration of a Cox risk score (SHERLOC V2).

SHERLOC's output sigmoid bounds the risk score eta in (0, 1), so the hazard ratio between two
patients is at most e and predicted survival is compressed. Recalibration rescales the score
without changing the ranking of patients:

    S(t | x) = exp(-H0(t) * exp(beta * eta)),

where beta is the slope of an unpenalised one-covariate Cox model fitted on *out-of-fold*
scores of the training patients (each scored by a model trained on the other folds) and H0 is
the Breslow estimator on the training patients with linear predictor beta * eta.
The same procedure applies to any model that outputs a risk score.

Typical use::

    oof = out_of_fold_scores(fit_fn, predict_fn, X_train, time_train, event_train, k=5)
    beta = calibration_slope(oof, time_train, event_train)
    surv_test = breslow_survival(beta * eta_train, time_train, event_train, beta * eta_test)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv


def calibration_slope(score: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Coefficient of an unpenalised Cox model with the score as only covariate."""
    m = CoxPHSurvivalAnalysis(alpha=0.0).fit(np.asarray(score, float).reshape(-1, 1),
                                             Surv.from_arrays(event=np.asarray(event).astype(bool),
                                                              time=np.asarray(time, float)))
    return float(m.coef_[0])


def breslow_survival(lp_train: np.ndarray, time_train: np.ndarray, event_train: np.ndarray,
                     lp_test: np.ndarray) -> pd.DataFrame:
    """Survival curves of test patients from the Breslow baseline hazard of the training patients.

    Returns a DataFrame indexed by time (0 and the distinct training event times), one column
    per test patient.
    """
    lp_tr, lp_te = np.asarray(lp_train, float), np.asarray(lp_test, float)
    t_tr, e_tr = np.asarray(time_train, float), np.asarray(event_train).astype(int)
    c = np.max(lp_tr)                                   # shift for numerical stability
    lp_tr, lp_te = np.clip(lp_tr - c, -700, 50), np.clip(lp_te - c, -700, 50)
    order = np.argsort(t_tr)
    t, e, r = t_tr[order], e_tr[order], np.exp(lp_tr[order])
    risk_set = np.cumsum(r[::-1])[::-1]
    ut = np.unique(t[e == 1])
    h = np.array([e[t == u].sum() / risk_set[np.searchsorted(t, u)] for u in ut])
    grid = np.unique(np.concatenate([[0.0], ut]))
    H = np.concatenate([[0.0], np.cumsum(h)])
    S = np.exp(-np.outer(H, np.exp(lp_te)))
    return pd.DataFrame(np.nan_to_num(S, nan=1.0), index=grid)


def out_of_fold_scores(fit: Callable, predict: Callable, X, time: np.ndarray, event: np.ndarray,
                       k: int = 5, seed: int = 0) -> np.ndarray:
    """Score each training patient with a model trained on the other k-1 folds.

    Args:
        fit: ``fit(X_subset, time_subset, event_subset) -> model`` (same settings as the full model).
        predict: ``predict(model, X_subset) -> risk scores``.
        X: training inputs, indexable by an integer array (numpy array, DataFrame via ``.iloc``,
           or a list/tuple of such objects for multi-input networks).
    """
    def take(obj, idx):
        if isinstance(obj, (list, tuple)):
            return type(obj)(take(o, idx) for o in obj)
        return obj.iloc[idx] if hasattr(obj, "iloc") else obj[idx]

    time, event = np.asarray(time, float), np.asarray(event).astype(int)
    oof = np.full(len(time), np.nan)
    for i_fit, i_out in StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(time)), event):
        m = fit(take(X, i_fit), time[i_fit], event[i_fit])
        oof[i_out] = np.asarray(predict(m, take(X, i_out)), float).ravel()
    return oof


def recalibrate(eta_train: np.ndarray, time_train: np.ndarray, event_train: np.ndarray,
                eta_test: np.ndarray, oof_train: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Slope on out-of-fold scores, then Breslow survival of the test patients. Returns (beta, S)."""
    beta = calibration_slope(oof_train, time_train, event_train)
    return beta, breslow_survival(beta * np.asarray(eta_train, float), time_train, event_train,
                                  beta * np.asarray(eta_test, float))
