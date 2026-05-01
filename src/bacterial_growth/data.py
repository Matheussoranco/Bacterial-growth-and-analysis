"""
Synthetic data generators for bacterial growth analysis.

Two datasets are provided:
  1. growth_curves  — time-series log CFU/mL observations for each culture
  2. classification — static feature vectors with species labels

The species profiles are loosely calibrated to published literature values for
the four reference organisms used throughout the package.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .kinetics import GrowthKinetics, fit_growth_curve

__all__ = [
    "SPECIES_PROFILES",
    "generate_growth_curves",
    "generate_classification_dataset",
]


# Literature-calibrated cardinal parameter profiles
# Keys: (T_min, T_opt, T_max, pH_min, pH_opt, pH_max, mu_max_opt, lag_opt, log_nmax)
SPECIES_PROFILES: dict[str, dict] = {
    "E. coli": dict(
        T_min=7.0, T_opt=37.0, T_max=46.0,
        pH_min=4.4, pH_opt=7.0, pH_max=9.0,
        mu_max_opt=1.2, lag_opt=1.0, log_nmax=9.0,
        gram="negative", oxygen="facultative",
        shape="rod", motility=True,
    ),
    "S. aureus": dict(
        T_min=7.0, T_opt=37.0, T_max=48.0,
        pH_min=4.0, pH_opt=7.0, pH_max=9.6,
        mu_max_opt=0.9, lag_opt=1.5, log_nmax=9.0,
        gram="positive", oxygen="facultative",
        shape="coccus", motility=False,
    ),
    "P. aeruginosa": dict(
        T_min=4.0, T_opt=37.0, T_max=42.0,
        pH_min=5.0, pH_opt=7.0, pH_max=8.5,
        mu_max_opt=0.8, lag_opt=2.0, log_nmax=8.5,
        gram="negative", oxygen="aerobic",
        shape="rod", motility=True,
    ),
    "B. subtilis": dict(
        T_min=10.0, T_opt=30.0, T_max=55.0,
        pH_min=5.5, pH_opt=7.0, pH_max=8.5,
        mu_max_opt=1.0, lag_opt=0.8, log_nmax=9.5,
        gram="positive", oxygen="facultative",
        shape="rod", motility=True,
    ),
}


# ---------------------------------------------------------------------------
# Cardinal Parameter Model (Rosso et al. 1995)
# ---------------------------------------------------------------------------

def _gamma_temperature(T: float, T_min: float, T_opt: float, T_max: float) -> float:
    """Square-root cardinal model for temperature (Ratkowsky-style)."""
    if T <= T_min or T >= T_max:
        return 0.0
    num = (T - T_min) ** 2 * (T - T_max)
    denom = (T_opt - T_min) * ((T_opt - T_min) * (T - T_opt)
                                - (T_opt - T_max) * (T_opt + T_min - 2 * T))
    return abs(num / (denom + 1e-300))


def _gamma_ph(pH: float, pH_min: float, pH_opt: float, pH_max: float) -> float:
    """Cardinal model for pH (Rosso 1995)."""
    if pH <= pH_min or pH >= pH_max:
        return 0.0
    num = (pH - pH_min) * (pH - pH_max)
    denom = num - (pH - pH_opt) ** 2
    return abs(num / (denom + 1e-300))


def _effective_mu_max(species: str, T: float, pH: float) -> float:
    p = SPECIES_PROFILES[species]
    gamma_T = _gamma_temperature(T, p["T_min"], p["T_opt"], p["T_max"])
    gamma_pH = _gamma_ph(pH, p["pH_min"], p["pH_opt"], p["pH_max"])
    return p["mu_max_opt"] * gamma_T * gamma_pH


# ---------------------------------------------------------------------------
# Growth curve generator
# ---------------------------------------------------------------------------

def generate_growth_curves(
    n_curves: int = 200,
    n_timepoints: int = 24,
    t_max: float = 24.0,
    noise_std: float = 0.15,
    species: Optional[list[str]] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic growth curves + fitted kinetics.

    Parameters
    ----------
    n_curves : number of cultures to simulate
    n_timepoints : observations per culture
    t_max : experiment duration (hours)
    noise_std : measurement noise (log CFU/mL)
    species : species to include (default: all four)
    random_state : reproducibility seed

    Returns
    -------
    curves_df : DataFrame with columns [curve_id, species, T, pH, time, log_n]
    kinetics_df : DataFrame with one row per curve, kinetic parameters + features
    """
    rng = np.random.default_rng(random_state)
    if species is None:
        species = list(SPECIES_PROFILES.keys())

    t_grid = np.linspace(0, t_max, n_timepoints)
    curve_records = []
    kinetics_records = []

    for i in range(n_curves):
        sp = species[i % len(species)]
        p = SPECIES_PROFILES[sp]

        # Sample environmental conditions
        T = rng.uniform(p["T_min"] + 3, p["T_max"] - 3)
        pH = rng.uniform(p["pH_min"] + 0.5, p["pH_max"] - 0.5)

        # Compute effective kinetics via Cardinal Parameter Model
        mu = _effective_mu_max(sp, T, pH)
        mu = max(mu * rng.lognormal(0, 0.15), 0.05)
        lag = p["lag_opt"] * rng.lognormal(0, 0.2)
        log_n0 = rng.uniform(2.0, 4.0)
        log_nmax = p["log_nmax"] + rng.normal(0, 0.3)

        # True curve (Baranyi-Roberts)
        from .kinetics import baranyi_roberts
        log_n_true = baranyi_roberts(t_grid, mu, lag, log_n0, log_nmax)
        log_n_obs = log_n_true + rng.normal(0, noise_std, size=n_timepoints)
        log_n_obs = np.clip(log_n_obs, 0, 12)

        for t_val, n_val in zip(t_grid, log_n_obs):
            curve_records.append(dict(
                curve_id=i, species=sp, temperature=T, pH=pH,
                time=t_val, log_n=n_val,
            ))

        # Fit the noisy curve to extract kinetics
        try:
            kin = fit_growth_curve(t_grid, log_n_obs)
            row = dict(curve_id=i, species=sp, temperature=T, pH=pH)
            row.update(kin.as_dict())
        except RuntimeError:
            row = dict(curve_id=i, species=sp, temperature=T, pH=pH,
                       model="failed", mu_max=np.nan, lag_time=np.nan,
                       log_n0=np.nan, log_nmax=np.nan,
                       generation_time=np.nan, delta_log_n=np.nan,
                       rmse=np.nan, r_squared=np.nan)
        kinetics_records.append(row)

    curves_df = pd.DataFrame(curve_records)
    kinetics_df = pd.DataFrame(kinetics_records).dropna()
    return curves_df, kinetics_df


# ---------------------------------------------------------------------------
# Static classification dataset generator
# ---------------------------------------------------------------------------

def generate_classification_dataset(
    n_samples: int = 2000,
    species: Optional[list[str]] = None,
    class_weights: Optional[dict[str, float]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a static feature dataset for species classification.

    Features include phenotypic, biochemical, and antibiotic resistance markers
    plus cardinal growth parameter observations at standard conditions.

    Returns
    -------
    DataFrame with feature columns and a 'species' label column.
    """
    rng = np.random.default_rng(random_state)
    if species is None:
        species = list(SPECIES_PROFILES.keys())

    # Default balanced weights
    weights = {sp: 1.0 / len(species) for sp in species}
    if class_weights:
        weights.update(class_weights)
    total_w = sum(weights.values())
    probs = [weights[sp] / total_w for sp in species]

    labels = rng.choice(species, size=n_samples, p=probs)
    records = []

    for sp in labels:
        p = SPECIES_PROFILES[sp]
        # Phenotypic features (with measurement noise)
        record = {
            "species": sp,
            # Growth kinetics at optimal conditions (with noise)
            "mu_max_37C_pH7": p["mu_max_opt"] * rng.lognormal(0, 0.15),
            "lag_time_h": p["lag_opt"] * rng.lognormal(0, 0.2),
            "log_nmax": p["log_nmax"] + rng.normal(0, 0.25),
            "generation_time_h": np.log(2) / max(p["mu_max_opt"] * rng.lognormal(0, 0.1), 1e-3),
            # Cardinal temperatures
            "T_min": p["T_min"] + rng.normal(0, 1.0),
            "T_opt": p["T_opt"] + rng.normal(0, 1.5),
            "T_max": p["T_max"] + rng.normal(0, 1.0),
            # Cardinal pH
            "pH_min": p["pH_min"] + rng.normal(0, 0.2),
            "pH_opt": p["pH_opt"] + rng.normal(0, 0.2),
            "pH_max": p["pH_max"] + rng.normal(0, 0.2),
            # Biochemical markers (0–1 continuous, or Bernoulli for binary)
            "catalase_activity": rng.beta(8, 2) if p["gram"] == "positive" else rng.beta(6, 3),
            "oxidase_activity": rng.beta(8, 2) if sp == "P. aeruginosa" else rng.beta(2, 8),
            "indole_production": rng.beta(7, 3) if sp == "E. coli" else rng.beta(2, 8),
            "motility_score": rng.beta(7, 2) if p["motility"] else rng.beta(2, 8),
            "biofilm_index": rng.beta(8, 2) if sp == "P. aeruginosa" else rng.beta(3, 6),
            "hemolysis_score": rng.beta(7, 3) if sp == "S. aureus" else rng.beta(2, 7),
            "spore_forming": float(sp == "B. subtilis") + rng.normal(0, 0.05),
            # Antibiotic resistance scores (0 = sensitive, 5 = resistant)
            "resistance_beta_lactam": _resistance_score(sp, "beta_lactam", rng),
            "resistance_fluoroquinolone": _resistance_score(sp, "fluoroquinolone", rng),
            "resistance_aminoglycoside": _resistance_score(sp, "aminoglycoside", rng),
            "resistance_macrolide": _resistance_score(sp, "macrolide", rng),
            # Morphological metrics
            "colony_diameter_mm": _colony_size(sp, rng),
            "gram_stain_score": 1.0 if p["gram"] == "positive" else 0.0,
            # Metabolic activity (OD600 proxy)
            "metabolic_activity_24h": rng.beta(6, 2) if p["mu_max_opt"] > 0.9 else rng.beta(4, 4),
        }
        # Add one-hot encoded categorical features
        for ox_cat in ["aerobic", "anaerobic", "facultative"]:
            record[f"oxygen_{ox_cat}"] = float(p["oxygen"] == ox_cat)
        for sh in ["rod", "coccus", "spiral"]:
            record[f"shape_{sh}"] = float(p["shape"] == sh)
        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_RESISTANCE_PROFILES = {
    ("E. coli", "beta_lactam"): (0.6, 0.3),
    ("E. coli", "fluoroquinolone"): (0.4, 0.3),
    ("E. coli", "aminoglycoside"): (0.3, 0.3),
    ("E. coli", "macrolide"): (0.8, 0.2),
    ("S. aureus", "beta_lactam"): (0.7, 0.3),
    ("S. aureus", "fluoroquinolone"): (0.5, 0.3),
    ("S. aureus", "aminoglycoside"): (0.4, 0.3),
    ("S. aureus", "macrolide"): (0.3, 0.3),
    ("P. aeruginosa", "beta_lactam"): (0.8, 0.2),
    ("P. aeruginosa", "fluoroquinolone"): (0.7, 0.25),
    ("P. aeruginosa", "aminoglycoside"): (0.6, 0.25),
    ("P. aeruginosa", "macrolide"): (0.9, 0.15),
    ("B. subtilis", "beta_lactam"): (0.2, 0.2),
    ("B. subtilis", "fluoroquinolone"): (0.2, 0.2),
    ("B. subtilis", "aminoglycoside"): (0.3, 0.25),
    ("B. subtilis", "macrolide"): (0.1, 0.15),
}


def _resistance_score(species: str, antibiotic: str, rng: np.random.Generator) -> float:
    mu, sigma = _RESISTANCE_PROFILES.get((species, antibiotic), (0.4, 0.3))
    return float(np.clip(rng.normal(mu, sigma), 0.0, 1.0))


def _colony_size(species: str, rng: np.random.Generator) -> float:
    sizes = {
        "E. coli": (2.0, 0.5),
        "S. aureus": (1.5, 0.4),
        "P. aeruginosa": (3.0, 0.7),
        "B. subtilis": (4.0, 0.8),
    }
    mu, sigma = sizes.get(species, (2.0, 0.5))
    return float(np.clip(rng.normal(mu, sigma), 0.3, 8.0))
