# Bacterial Growth Analysis

A SOTA predictive microbiology toolkit combining mechanistic growth models, Gaussian Process Regression, and a Bayesian-optimised stacking ensemble classifier for bacterial species identification.

---

## Overview

This project addresses two complementary problems in computational microbiology:

1. **Growth kinetics estimation** — given a time-series of bacterial population counts, fit a mechanistic model and extract biologically interpretable kinetic parameters (µmax, lag time, carrying capacity).
2. **Species classification** — given phenotypic, biochemical, and antibiotic resistance features, classify bacterial cultures into species with calibrated uncertainty.

Both problems are approached with a full pipeline: synthetic data generation via the **Cardinal Parameter Model**, mechanistic curve fitting, Bayesian hyperparameter optimisation with **Optuna**, a **stacking ensemble** classifier, and **conformal prediction** for coverage-guaranteed uncertainty sets.

---

## Architecture

```
Bacterial-growth-and-analysis/
├── src/bacterial_growth/
│   ├── kinetics.py          # Growth curve models + multi-restart fitting
│   ├── data.py              # Synthetic data generation (CPM-calibrated)
│   ├── features.py          # Kinetic and statistical feature extractors
│   ├── classifier.py        # Stacking ensemble + Optuna HPO + conformal prediction
│   ├── growth_predictor.py  # GPR environmental response model + CPM
│   ├── visualization.py     # Comprehensive plotting suite
│   └── pipeline.py          # End-to-end orchestration
├── Bacterial_Growth_Simulation.py   # CLI entry point
├── requirements.txt
└── pyproject.toml
```

---

## Scientific Background

### Growth Curve Models

All four canonical models from the predictive microbiology literature are implemented. Each works in **log CFU/mL** space and is fitted using `scipy.optimize.curve_fit` with multiple random restarts to escape local minima.

| Model | Reference | Notes |
|---|---|---|
| **Baranyi-Roberts** | Baranyi & Roberts (1994) | Mechanistic; models physiological state variable q(t) |
| **Modified Gompertz** | Zwietering et al. (1990) | Asymmetric sigmoidal; inflection before midpoint |
| **Logistic** | Verhulst (1838) | Symmetric sigmoidal; simple baseline |
| **Buchanan Tri-linear** | Buchanan et al. (1997) | Piecewise linear; robust for noisy data |

The **Baranyi-Roberts** closed-form approximation is:

```
A(t) = t + (1/µmax) · ln(e^(-µmax·t) + e^(-µmax·λ) - e^(-µmax·(t+λ)))

log N(t) = Nmax - ln(1 + (e^(Nmax-N0) - 1) · e^(-µmax·A(t)))
```

where λ is the lag duration, µmax the maximum specific growth rate, N0 the inoculum, and Nmax the carrying capacity.

### Cardinal Parameter Model (CPM)

Environmental conditions modulate µmax multiplicatively (Rosso et al. 1995):

```
µ(T, pH) = µmax_opt · γ(T) · γ(pH)
```

where each gamma function is zero outside cardinal boundaries (Tmin/Topt/Tmax, pHmin/pHopt/pHmax) and equals 1 at the optimum. This gives a biologically interpretable growth boundary in (T, pH) space.

### Extracted Kinetic Parameters

| Parameter | Symbol | Unit | Description |
|---|---|---|---|
| Max growth rate | µmax | /h | Steepest slope of log N vs time |
| Lag duration | λ | h | Adaptation phase before exponential growth |
| Initial population | N0 | log CFU/mL | Inoculum density |
| Carrying capacity | Nmax | log CFU/mL | Stationary phase maximum |
| Generation time | tgen = ln2/µmax | h | Mean cell doubling time |
| Growth range | ΔN = Nmax − N0 | log CFU/mL | Total growth achieved |

---

## Machine Learning Pipeline

### Feature Engineering

Two complementary feature sets are extracted from each growth curve:

- **Kinetic features** (8 features): fitted model parameters — µmax, λ, N0, Nmax, tgen, ΔN, RMSE, R².
- **Statistical features** (18 features): model-free — AUC, max/mean rate, IQ range, early/late AUC ratio, skewness, kurtosis.

Static classification features include: growth kinetics at standard conditions, cardinal temperatures, phenotypic markers, biochemical activity scores, antibiotic resistance profiles, morphological measurements.

### Preprocessing Pipeline

```
Raw data
  → StandardScaler          (zero mean, unit variance)
  → SMOTE                   (synthetic minority over-sampling)
  → SelectFromModel         (Random Forest mean-importance threshold)
  → PCA (95% variance)      (dimensionality reduction)
```

### Base Learners + Stacking Ensemble

Five base learners are trained with **Optuna** Bayesian hyperparameter optimisation (Tree-structured Parzen Estimator):

| Model | Search space |
|---|---|
| Random Forest | n_estimators, max_depth, min_samples_split/leaf, max_features |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample, α, λ |
| LightGBM | n_estimators, num_leaves, learning_rate, min_child_samples, α, λ |
| MLP | n_layers, units per layer, α, learning_rate, batch_size |
| SVM | C, kernel, γ, degree |

A **StackingClassifier** combines these using a **LogisticRegression** meta-learner with 3-fold out-of-fold predictions as meta-features. The final ensemble is **isotonic calibrated** (Platt scaling variant) for reliable probability outputs.

### Uncertainty Quantification — Conformal Prediction

Coverage-guaranteed prediction sets are computed using **split conformal prediction** (Venn-Abers variant):

1. Compute non-conformity scores α̂ᵢ = 1 − P̂(yᵢ | xᵢ) on calibration data.
2. At test time, include class k in the prediction set if 1 − P̂(k | x) ≤ quantile(α̂, coverage).

This guarantees **marginal coverage**: the true species is in the prediction set with at least the specified probability (e.g. 90%), regardless of the model's internal calibration.

### Environmental Response — Gaussian Process Regression

A **Gaussian Process Regressor** (Matérn-5/2 kernel + white noise) learns the mapping from (temperature, pH, ...) to (µmax, λ, Nmax). GPR provides predictive mean and standard deviation, enabling uncertainty-aware growth kinetics prediction under novel environmental conditions.

---

## Installation

```bash
git clone <repo-url>
cd Bacterial-growth-and-analysis

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# Optional: install as editable package
pip install -e .
```

---

## Usage

### Quick Demo (no heavy training)

```bash
python Bacterial_Growth_Simulation.py --mode demo
```

Runs three lightweight demos:
- Growth curve fitting with model selection
- Cardinal Parameter Model predictions
- Classifier training and conformal prediction sets

### Full Pipeline

```bash
python Bacterial_Growth_Simulation.py --mode pipeline \
    --output-dir results/ \
    --n-curves 300 \
    --n-samples 3000 \
    --optuna-trials 40
```

Produces in `results/`:

| Output | Description |
|---|---|
| `growth_curves_comparison.png` | Side-by-side growth curves per species |
| `growth_curve_<species>.png` | Individual curve with model overlay and annotations |
| `kinetics_mu_max.png` | Violin plots of µmax distribution per species |
| `kinetics_lag_time.png` | Violin plots of lag time distribution |
| `boundary_<species>.png` | CPM growth boundary in (T, pH) space |
| `confusion_matrix.png` | Normalised confusion matrix heatmap |
| `roc_curves.png` | One-vs-Rest ROC curves with per-class AUC |
| `calibration_<species>.png` | Reliability diagrams per species |
| `conformal_coverage.png` | Empirical coverage per class at 90% target |
| `classifier.joblib` | Serialised trained classifier |
| `kinetics.csv` | Fitted kinetic parameters for all curves |
| `classification_dataset.csv` | Full classification feature dataset |
| `metrics.json` | Evaluation metrics (F1, AUC) |

### Python API

```python
import sys; sys.path.insert(0, "src")
from bacterial_growth import fit_growth_curve, BacterialCultureClassifier
from bacterial_growth import CardinalParameterModel, generate_classification_dataset
import numpy as np

# --- Growth curve fitting ---
t = np.linspace(0, 24, 25)
log_n = ...   # your measurements (log CFU/mL)
kin = fit_growth_curve(t, log_n)
print(kin.mu_max, kin.lag_time, kin.generation_time)

# --- Cardinal Parameter Model ---
cpm = CardinalParameterModel("E. coli")
mu = cpm.mu_max(T=15.0, pH=5.5)

# --- Species classification ---
df = generate_classification_dataset(n_samples=2000)
X, y = df.drop(columns=["species"]), df["species"].values

clf = BacterialCultureClassifier(n_trials=40)
clf.fit(X[:1600], y[:1600])
clf.evaluate(X[1600:], y[1600:])

# Conformal prediction sets (90% coverage guarantee)
sets = clf.predict_set(X[1600:1605], coverage=0.90)

# SHAP explanations
shap_vals = clf.explain(X[:100])

# Save / load
clf.save("my_classifier.joblib")
clf2 = BacterialCultureClassifier.load("my_classifier.joblib")
```

---

## Reference Species

| Species | Gram | Oxygen | T_opt (°C) | pH_opt | µmax_opt (/h) |
|---|---|---|---|---|---|
| *E. coli* | − | Facultative | 37 | 7.0 | 1.20 |
| *S. aureus* | + | Facultative | 37 | 7.0 | 0.90 |
| *P. aeruginosa* | − | Aerobic | 37 | 7.0 | 0.80 |
| *B. subtilis* | + | Facultative | 30 | 7.0 | 1.00 |

Cardinal parameters sourced from ComBase and published predictive microbiology literature.

---

## Key References

- Baranyi, J. & Roberts, T.A. (1994). A dynamic approach to predicting bacterial growth in food. *IJFM* 23:277–294.
- Zwietering, M.H. et al. (1990). Modeling of the bacterial growth curve. *Appl. Environ. Microbiol.* 56:1875–1881.
- Rosso, L. et al. (1995). An unexpected correlation between cardinal temperatures of microbial growth. *J. Theor. Biol.* 176:169–176.
- Buchanan, R.L. et al. (1997). When is simple good enough: a comparison of the Gompertz, Baranyi, and three-phase linear models. *IJFM* 38:121–132.
- Venn-Abers / Split Conformal: Angelopoulos & Bates (2023). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.*
- Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions (SHAP). *NeurIPS*.

---

## License

MIT — see [LICENSE](LICENSE).
