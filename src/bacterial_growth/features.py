"""
Feature extraction from raw growth-curve time series.

Two complementary feature sets:
  - Kinetic features: fit a growth model and extract µmax, lag, etc.
  - Statistical features: model-free statistics (slopes, AUC, percentiles)

Both implement sklearn's fit/transform interface for use in Pipelines.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .kinetics import GrowthKinetics, fit_growth_curve

__all__ = ["KineticFeatureExtractor", "StatisticalFeatureExtractor", "extract_all_features"]


class KineticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Fit growth models to each curve and return kinetic parameters as features.

    Parameters
    ----------
    models : growth models to try; best-fitting is kept
    """

    def __init__(self, models: tuple = ("baranyi_roberts", "modified_gompertz", "logistic")):
        self.models = models
        self.fitted_kinetics_: list[GrowthKinetics] = []

    def fit(self, curves: list[tuple[np.ndarray, np.ndarray]], y=None):
        return self

    def transform(self, curves: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """
        Parameters
        ----------
        curves : list of (t, log_n) tuples

        Returns
        -------
        DataFrame with one row per curve
        """
        records = []
        self.fitted_kinetics_ = []
        for t, log_n in curves:
            try:
                kin = fit_growth_curve(np.asarray(t), np.asarray(log_n), models=self.models)
                self.fitted_kinetics_.append(kin)
                records.append(kin.as_dict())
            except RuntimeError:
                self.fitted_kinetics_.append(None)
                records.append({k: np.nan for k in [
                    "mu_max", "lag_time", "log_n0", "log_nmax",
                    "generation_time", "delta_log_n", "rmse", "r_squared",
                ]})
        df = pd.DataFrame(records).drop(columns=["model"], errors="ignore")
        return df


class StatisticalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Model-free statistical features from raw growth curves.

    These are computed directly from the time-series without any model fitting
    and complement the kinetic features as a robustness hedge.
    """

    def fit(self, curves: list[tuple[np.ndarray, np.ndarray]], y=None):
        return self

    def transform(self, curves: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        records = []
        for t, log_n in curves:
            t = np.asarray(t, dtype=float)
            log_n = np.asarray(log_n, dtype=float)
            records.append(self._extract(t, log_n))
        return pd.DataFrame(records)

    @staticmethod
    def _extract(t: np.ndarray, log_n: np.ndarray) -> dict:
        n = len(log_n)
        delta_t = t[1] - t[0] if n > 1 else 1.0

        # Instantaneous rates
        rates = np.diff(log_n) / (np.diff(t) + 1e-9)

        # AUC via trapezoidal rule
        auc = float(np.trapz(log_n, t))
        auc_rate = float(np.trapz(np.maximum(rates, 0), t[:-1]))

        # Phase detection heuristic
        max_rate_idx = int(np.argmax(rates)) if len(rates) > 0 else 0

        return {
            # Basic summary
            "stat_mean_log_n": float(np.mean(log_n)),
            "stat_std_log_n": float(np.std(log_n)),
            "stat_max_log_n": float(np.max(log_n)),
            "stat_min_log_n": float(np.min(log_n)),
            "stat_range_log_n": float(np.max(log_n) - np.min(log_n)),
            # Growth dynamics
            "stat_max_rate": float(np.max(rates)) if len(rates) else np.nan,
            "stat_mean_rate": float(np.mean(rates)) if len(rates) else np.nan,
            "stat_auc": auc,
            "stat_auc_positive_rate": auc_rate,
            # Time-indexed features
            "stat_time_to_max": float(t[np.argmax(log_n)]),
            "stat_time_max_rate": float(t[max_rate_idx]) if len(rates) > 0 else np.nan,
            # Shape / curvature
            "stat_skewness": _skewness(log_n),
            "stat_kurtosis": _kurtosis(log_n),
            "stat_rate_std": float(np.std(rates)) if len(rates) else np.nan,
            # Early vs late growth split
            "stat_early_auc": float(np.trapz(log_n[:n // 3], t[:n // 3])) if n > 3 else np.nan,
            "stat_late_auc": float(np.trapz(log_n[2 * n // 3:], t[2 * n // 3:])) if n > 3 else np.nan,
            "stat_early_late_ratio": float(
                np.mean(log_n[:n // 4]) / (np.mean(log_n[3 * n // 4:]) + 1e-9)
            ) if n > 4 else np.nan,
            # Percentiles
            "stat_p25": float(np.percentile(log_n, 25)),
            "stat_p75": float(np.percentile(log_n, 75)),
            "stat_iqr": float(np.percentile(log_n, 75) - np.percentile(log_n, 25)),
        }


def extract_all_features(
    curves: list[tuple[np.ndarray, np.ndarray]],
    use_kinetic: bool = True,
    use_statistical: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: extract both kinetic and statistical features.

    Parameters
    ----------
    curves : list of (t, log_n) tuples

    Returns
    -------
    Combined feature DataFrame
    """
    frames = []
    if use_kinetic:
        frames.append(KineticFeatureExtractor().transform(curves))
    if use_statistical:
        frames.append(StatisticalFeatureExtractor().transform(curves))
    if not frames:
        raise ValueError("At least one of use_kinetic or use_statistical must be True.")
    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skewness(x: np.ndarray) -> float:
    mu, sigma = np.mean(x), np.std(x)
    if sigma < 1e-9:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    mu, sigma = np.mean(x), np.std(x)
    if sigma < 1e-9:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)
