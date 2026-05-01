"""
bacterial_growth — SOTA predictive microbiology toolkit.

Modules
-------
kinetics        : growth curve models (Baranyi-Roberts, Gompertz, Logistic, Buchanan)
data            : synthetic data generation with Cardinal Parameter Model
features        : feature extraction from growth time series
classifier      : species classifier with Optuna HPO + stacking ensemble + SHAP
growth_predictor: GPR-based environmental response modelling
visualization   : comprehensive plotting suite
pipeline        : end-to-end orchestration
"""

from .kinetics import GrowthKinetics, fit_growth_curve
from .data import SPECIES_PROFILES, generate_growth_curves, generate_classification_dataset
from .classifier import BacterialCultureClassifier
from .growth_predictor import GrowthPredictor, CardinalParameterModel
from .pipeline import BacterialAnalysisPipeline

__version__ = "2.0.0"
__all__ = [
    "GrowthKinetics",
    "fit_growth_curve",
    "SPECIES_PROFILES",
    "generate_growth_curves",
    "generate_classification_dataset",
    "BacterialCultureClassifier",
    "GrowthPredictor",
    "CardinalParameterModel",
    "BacterialAnalysisPipeline",
]
