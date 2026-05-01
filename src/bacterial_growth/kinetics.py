"""
Growth curve models for predictive microbiology.

Implements the four canonical models from the predictive microbiology literature:
  - Baranyi-Roberts (1994) — mechanistic, biologically interpretable
  - Modified Gompertz (Zwietering 1990) — sigmoidal, widely adopted
  - Logistic / Verhulst — symmetric sigmoidal baseline
  - Buchanan tri-linear — piecewise linear, simple and robust

All models work in log CFU/mL space. Fitting uses scipy.optimize with multiple
random restarts to avoid local minima.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import pearsonr

__all__ = [
    "GrowthKinetics",
    "fit_growth_curve",
    "baranyi_roberts",
    "modified_gompertz",
    "logistic_growth",
    "buchanan_trilinear",
]


@dataclass
class GrowthKinetics:
    """Fitted kinetic parameters for a single growth curve."""

    model_name: str
    mu_max: float           # Maximum specific growth rate (log CFU / h)
    lag_time: float         # Lag phase duration (h)
    log_n0: float           # Initial population (log CFU / mL)
    log_nmax: float         # Carrying capacity (log CFU / mL)
    rmse: float             # Root mean squared error of fit
    r_squared: float        # Coefficient of determination
    generation_time: float = field(init=False)   # ln(2) / mu_max (h)
    delta_log_n: float = field(init=False)       # log_nmax - log_n0 (growth range)

    def __post_init__(self) -> None:
        self.generation_time = np.log(2) / max(self.mu_max, 1e-9)
        self.delta_log_n = self.log_nmax - self.log_n0

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "mu_max": self.mu_max,
            "lag_time": self.lag_time,
            "log_n0": self.log_n0,
            "log_nmax": self.log_nmax,
            "generation_time": self.generation_time,
            "delta_log_n": self.delta_log_n,
            "rmse": self.rmse,
            "r_squared": self.r_squared,
        }


# ---------------------------------------------------------------------------
# Model functions (vectorised, work in log CFU/mL)
# ---------------------------------------------------------------------------

def baranyi_roberts(t: np.ndarray, mu_max: float, lag: float,
                    log_n0: float, log_nmax: float) -> np.ndarray:
    """Baranyi-Roberts model (closed-form approximation, Baranyi & Roberts 1994).

    This is the most mechanistically motivated model — it tracks the
    physiological state of cells as they adapt from lag to exponential growth.
    """
    t = np.asarray(t, dtype=float)
    # A(t) = adjustment function for lag phase
    At = t + (1.0 / mu_max) * np.log(np.exp(-mu_max * t) + np.exp(-mu_max * lag)
                                       - np.exp(-mu_max * (t + lag)) + 1e-300)
    delta = log_nmax - log_n0
    log_nt = log_nmax - np.log(1.0 + (np.exp(delta) - 1.0) * np.exp(-mu_max * At) + 1e-300)
    return log_nt


def modified_gompertz(t: np.ndarray, mu_max: float, lag: float,
                      log_n0: float, log_nmax: float) -> np.ndarray:
    """Modified Gompertz model (Zwietering et al. 1990).

    Asymmetric sigmoidal — inflection occurs before the midpoint.
    """
    t = np.asarray(t, dtype=float)
    A = log_nmax - log_n0
    exponent = np.exp(1.0) * mu_max / A * (lag - t) + 1.0
    return log_n0 + A * np.exp(-np.exp(exponent))


def logistic_growth(t: np.ndarray, mu_max: float, lag: float,
                    log_n0: float, log_nmax: float) -> np.ndarray:
    """Symmetric logistic (Verhulst) model."""
    t = np.asarray(t, dtype=float)
    A = log_nmax - log_n0
    t_inflect = lag + A / (2.0 * mu_max)
    return log_n0 + A / (1.0 + np.exp(-4.0 * mu_max / A * (t - t_inflect)))


def buchanan_trilinear(t: np.ndarray, mu_max: float, lag: float,
                       log_n0: float, log_nmax: float) -> np.ndarray:
    """Buchanan tri-linear model (Buchanan et al. 1997).

    Piecewise: flat lag → linear exponential → flat stationary.
    """
    t = np.asarray(t, dtype=float)
    t_stat = lag + (log_nmax - log_n0) / mu_max
    log_nt = np.where(
        t <= lag,
        log_n0,
        np.where(t <= t_stat, log_n0 + mu_max * (t - lag), log_nmax),
    )
    return log_nt.astype(float)


# ---------------------------------------------------------------------------
# Fitting engine
# ---------------------------------------------------------------------------

_MODELS = {
    "baranyi_roberts": baranyi_roberts,
    "modified_gompertz": modified_gompertz,
    "logistic": logistic_growth,
    "buchanan": buchanan_trilinear,
}


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-300)


def _fit_single_model(model_fn, t, log_n, n_restarts: int = 8) -> Optional[tuple]:
    """Fit one model with multiple random restarts; return (params, rmse, r2) or None."""
    # Heuristic initial guesses
    mu_guess = (log_n.max() - log_n.min()) / (t.max() - t.min() + 1e-9)
    lag_guess = t.max() * 0.15
    n0_guess = log_n[0]
    nmax_guess = log_n.max()

    best = None
    rng = np.random.default_rng(0)
    for _ in range(n_restarts):
        p0 = [
            max(rng.uniform(0.5, 2.0) * mu_guess, 1e-4),
            rng.uniform(0.0, 0.5) * t.max() if _ > 0 else lag_guess,
            n0_guess + rng.normal(0, 0.2),
            nmax_guess + rng.normal(0, 0.2),
        ]
        bounds = ([1e-6, 0.0, -2.0, 0.0], [20.0, t.max() * 0.9, 15.0, 20.0])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(model_fn, t, log_n, p0=p0, bounds=bounds,
                                    maxfev=5000)
            pred = model_fn(t, *popt)
            rmse = np.sqrt(np.mean((log_n - pred) ** 2))
            r2 = _r_squared(log_n, pred)
            if best is None or rmse < best[1]:
                best = (popt, rmse, r2)
        except (RuntimeError, ValueError):
            continue
    return best


def fit_growth_curve(
    t: np.ndarray,
    log_n: np.ndarray,
    models: tuple[str, ...] = ("baranyi_roberts", "modified_gompertz", "logistic", "buchanan"),
    n_restarts: int = 8,
) -> GrowthKinetics:
    """Fit multiple growth models and return the best-fitting one.

    Parameters
    ----------
    t : array of time points (hours)
    log_n : array of log CFU/mL measurements
    models : which models to try
    n_restarts : random restarts per model to escape local minima

    Returns
    -------
    GrowthKinetics with parameters from the model with lowest RMSE
    """
    t = np.asarray(t, dtype=float)
    log_n = np.asarray(log_n, dtype=float)
    if len(t) < 4:
        raise ValueError("Need at least 4 time points to fit a growth curve.")

    best_kin: Optional[GrowthKinetics] = None
    for name in models:
        fn = _MODELS[name]
        result = _fit_single_model(fn, t, log_n, n_restarts=n_restarts)
        if result is None:
            continue
        (mu_max, lag, log_n0, log_nmax), rmse, r2 = result
        kin = GrowthKinetics(
            model_name=name,
            mu_max=abs(mu_max),
            lag_time=abs(lag),
            log_n0=log_n0,
            log_nmax=log_nmax,
            rmse=rmse,
            r_squared=r2,
        )
        if best_kin is None or rmse < best_kin.rmse:
            best_kin = kin

    if best_kin is None:
        raise RuntimeError("All growth models failed to converge. Check input data.")
    return best_kin


def predict_growth_curve(
    kinetics: GrowthKinetics,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluate the fitted growth model over an arbitrary time grid."""
    fn = _MODELS[kinetics.model_name]
    return fn(
        np.asarray(t, dtype=float),
        kinetics.mu_max,
        kinetics.lag_time,
        kinetics.log_n0,
        kinetics.log_nmax,
    )
