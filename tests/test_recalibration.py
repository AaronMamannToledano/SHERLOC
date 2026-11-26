import numpy as np
import pytest

pytest.importorskip("sksurv")
from sherloc.recalibration import breslow_survival, calibration_slope, out_of_fold_scores, recalibrate  # noqa: E402


def _data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n)
    t = rng.exponential(np.exp(-0.8 * x))
    c = rng.exponential(2.0, n)
    return x, np.minimum(t, c), (t <= c).astype(int)


def test_slope_recovers_compression():
    x, t, e = _data()
    eta = 1 / (1 + np.exp(-0.2 * x))           # compressed, monotone score
    b = calibration_slope(eta, t, e)
    assert b > 5                                # stretching needed


def test_breslow_monotone_and_ranked():
    x, t, e = _data()
    S = breslow_survival(0.8 * x, t, e, np.array([-1.0, 0.0, 1.0]))
    assert np.all(np.diff(S.to_numpy(), axis=0) <= 1e-12)
    assert np.all(S.iloc[-1].to_numpy()[:-1] >= S.iloc[-1].to_numpy()[1:])


def test_oof_and_recalibrate_run():
    x, t, e = _data()
    fit = lambda X, tt, ee: calibration_slope(X, tt, ee)
    pred = lambda m, X: m * X
    oof = out_of_fold_scores(fit, pred, x, t, e, k=5)
    assert np.isfinite(oof).all()
    beta, S = recalibrate(x, t, e, x[:5], oof)
    assert S.shape[1] == 5 and np.isfinite(beta)
