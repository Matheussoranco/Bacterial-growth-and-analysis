"""
Environmental response modelling: predict growth kinetics from conditions.

Uses Gaussian Process Regression to learn the mapping:
    (temperature, pH, ...) → (µmax, lag_time, log_nmax)

The Cardinal Parameter Model (CPM) is used as an interpretable analytical
reference alongside the data-driven GP predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SPECIES_PROFILES, _effective_mu_max, _gamma_ph, _gamma_temperature

__all__ = ["GrowthPredictor", "CardinalParameterModel"]


class GrowthPredictor:
    """Data-driven growth kinetics predictor using Gaussian Process Regression.

    Predicts µmax, lag_time, and log_nmax from environmental inputs using GPR.
    Uncertainty estimates (σ) are returned alongside point predictions.

    Parameters
    ----------
    targets : kinetic parameters to predict
    """

    _TARGETS = ("mu_max", "lag_time", "log_nmax")

    def __init__(self, targets: tuple[str, ...] = _TARGETS):
        self.targets = targets
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42,
        )
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", MultiOutputRegressor(gpr)),
        ])
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, kinetics_df: pd.DataFrame, condition_cols: list[str]) -> "GrowthPredictor":
        """
        Parameters
        ----------
        kinetics_df : output of data.generate_growth_curves (kinetics_df)
        condition_cols : columns to use as features (e.g. ["temperature", "pH"])
        """
        self._feature_names = condition_cols
        X = kinetics_df[condition_cols].values
        Y = kinetics_df[list(self.targets)].values
        # GPR scales poorly with N; subsample if large
        if len(X) > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), 500, replace=False)
            X, Y = X[idx], Y[idx]
        self._pipeline.fit(X, Y)
        self._is_fitted = True
        return self

    def predict(
        self, conditions: pd.DataFrame, return_std: bool = True
    ) -> pd.DataFrame:
        """Predict kinetics for new environmental conditions.

        Returns
        -------
        DataFrame with columns [target, target_std] for each target.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        X = conditions[self._feature_names].values
        gpr_model = self._pipeline.named_steps["gpr"]
        scaler = self._pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X)

        results = {}
        for i, (estimator, target) in enumerate(zip(gpr_model.estimators_, self.targets)):
            y_pred, y_std = estimator.predict(X_scaled, return_std=True)
            results[target] = y_pred
            results[f"{target}_std"] = y_std

        return pd.DataFrame(results, index=conditions.index)


class CardinalParameterModel:
    """Analytical Cardinal Parameter Model (Rosso et al. 1995).

    Predicts µmax as a product of temperature and pH gamma functions.
    No fitting required — uses tabulated species profiles from SPECIES_PROFILES.

    µ(T, pH) = µmax_opt · γ(T) · γ(pH)
    """

    def __init__(self, species: str):
        if species not in SPECIES_PROFILES:
            raise ValueError(f"Unknown species '{species}'. "
                             f"Available: {list(SPECIES_PROFILES.keys())}")
        self.species = species
        self._p = SPECIES_PROFILES[species]

    def mu_max(self, T: float | np.ndarray, pH: float | np.ndarray) -> np.ndarray:
        T = np.atleast_1d(np.asarray(T, dtype=float))
        pH = np.atleast_1d(np.asarray(pH, dtype=float))
        result = np.zeros((len(T), len(pH)))
        for i, t in enumerate(T):
            for j, ph in enumerate(pH):
                result[i, j] = _effective_mu_max(self.species, t, ph)
        return result.squeeze()

    def growth_boundary(
        self,
        T_range: tuple[float, float] = (0, 55),
        pH_range: tuple[float, float] = (3, 10),
        threshold: float = 0.01,
        resolution: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the growth boundary in (T, pH) space.

        Returns
        -------
        T_grid, pH_grid, mu_grid — for use with matplotlib contourf.
        """
        T_vals = np.linspace(*T_range, resolution)
        pH_vals = np.linspace(*pH_range, resolution)
        T_grid, pH_grid = np.meshgrid(T_vals, pH_vals)
        mu_grid = np.vectorize(
            lambda t, ph: _effective_mu_max(self.species, t, ph)
        )(T_grid, pH_grid)
        return T_grid, pH_grid, mu_grid

    @property
    def cardinal_temperatures(self) -> dict:
        return {k: self._p[k] for k in ("T_min", "T_opt", "T_max")}

    @property
    def cardinal_pH(self) -> dict:
        return {k: self._p[k] for k in ("pH_min", "pH_opt", "pH_max")}
