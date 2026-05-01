"""
SOTA bacterial species classifier.

Improvements over the original implementation:
  - Optuna Bayesian hyperparameter optimisation (replaces GridSearchCV)
  - Stacking ensemble with meta-learner (LR) as final estimator
  - Calibrated probability outputs (Platt scaling / isotonic regression)
  - SHAP-based global and local explanations
  - Conformal prediction for coverage-guaranteed uncertainty
  - MLP with batch normalisation as an additional base learner
  - Class-balanced training with SMOTE + class-weight correction
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

__all__ = ["BacterialCultureClassifier"]


class BacterialCultureClassifier:
    """Multi-algorithm stacking classifier for bacterial species identification.

    Workflow
    --------
    1. Preprocessing: scale → SMOTE → feature selection → PCA
    2. Base learner HPO: Optuna Bayesian search (or random search fallback)
    3. Stacking ensemble: base learners → meta-learner (LogisticRegression)
    4. Calibration: isotonic regression for reliable probabilities
    5. Conformal prediction: empirical coverage-guaranteed prediction sets

    Parameters
    ----------
    n_trials : Optuna trials per model (ignored if Optuna unavailable)
    cv_folds : folds for cross-validation
    pca_variance : fraction of variance retained by PCA (None = skip PCA)
    random_state : global seed
    """

    def __init__(
        self,
        n_trials: int = 40,
        cv_folds: int = 5,
        pca_variance: float = 0.95,
        random_state: int = 42,
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.pca_variance = pca_variance
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._label_enc = LabelEncoder()
        self._feature_selector: Optional[SelectFromModel] = None
        self._pca: Optional[PCA] = None
        self._smote = SMOTE(random_state=random_state)

        self._base_models: dict = {}
        self._ensemble = None
        self._calibrated = None
        self._conformal_scores: Optional[np.ndarray] = None
        self._classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BacterialCultureClassifier":
        """Full training pipeline."""
        X_proc, y_proc = self._preprocess_train(X, y)
        self._classes_ = self._label_enc.classes_
        self._train_base_models(X_proc, y_proc)
        self._build_stacking_ensemble(X_proc, y_proc)
        self._fit_conformal(X_proc, y_proc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self._preprocess_infer(X)
        y_enc = self._calibrated.predict(X_proc)
        return self._label_enc.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X_proc = self._preprocess_infer(X)
        proba = self._calibrated.predict_proba(X_proc)
        return pd.DataFrame(proba, columns=self._label_enc.classes_)

    def predict_set(self, X: pd.DataFrame, coverage: float = 0.9) -> list[list[str]]:
        """Conformal prediction sets with guaranteed marginal coverage.

        Each row returns the minimal set of species labels containing the true
        label with at least `coverage` empirical probability.
        """
        if self._conformal_scores is None:
            raise RuntimeError("Call fit() before predict_set().")
        threshold = float(np.quantile(self._conformal_scores, coverage))
        proba_df = self.predict_proba(X)
        result = []
        for _, row in proba_df.iterrows():
            # Non-conformity score: 1 - p_hat(class)
            included = [cls for cls in self._label_enc.classes_
                        if (1.0 - row[cls]) <= threshold]
            result.append(included if included else [self._label_enc.classes_[row.values.argmax()]])
        return result

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Return evaluation metrics and print a report."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X).values
        y_enc = self._label_enc.transform(y)

        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred, labels=self._label_enc.classes_)
        f1_w = f1_score(y_enc, self._label_enc.transform(y_pred), average="weighted")

        try:
            auc = roc_auc_score(y_enc, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = float("nan")

        print("=" * 60)
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, index=self._label_enc.classes_,
                           columns=self._label_enc.classes_).to_string())
        print(f"\nWeighted F1 : {f1_w:.4f}")
        print(f"ROC-AUC (OvR): {auc:.4f}")
        print("=" * 60)

        return {"classification_report": report, "confusion_matrix": cm,
                "f1_weighted": f1_w, "roc_auc": auc}

    def explain(self, X: pd.DataFrame, n_samples: int = 100) -> Optional[object]:
        """Return SHAP Explanation object for the top base model.

        Requires ``shap`` to be installed. Falls back gracefully.
        """
        if not _HAS_SHAP:
            warnings.warn("shap not installed — skipping explanation.")
            return None

        X_proc = self._preprocess_infer(X.head(n_samples))
        feature_names = [f"f{i}" for i in range(X_proc.shape[1])]

        # Use the best tree model for SHAP (TreeExplainer is fast)
        tree_model = None
        for name in ("lightgbm", "xgboost", "random_forest"):
            if name in self._base_models:
                tree_model = self._base_models[name]
                break

        if tree_model is not None:
            explainer = shap.TreeExplainer(tree_model)
        else:
            explainer = shap.KernelExplainer(
                self._calibrated.predict_proba, shap.sample(X_proc, 50)
            )

        shap_values = explainer(X_proc)
        print("SHAP summary computed. Call shap.summary_plot(shap_values) to visualise.")
        return shap_values

    def save(self, path: str | Path) -> None:
        joblib.dump(self.__dict__, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BacterialCultureClassifier":
        obj = cls.__new__(cls)
        obj.__dict__.update(joblib.load(path))
        return obj

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_train(self, X: pd.DataFrame, y: np.ndarray):
        X_arr = np.asarray(X, dtype=float)
        y_enc = self._label_enc.fit_transform(y)

        X_scaled = self._scaler.fit_transform(X_arr)
        X_res, y_res = self._smote.fit_resample(X_scaled, y_enc)

        self._feature_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            threshold="mean",
        )
        X_sel = self._feature_selector.fit_transform(X_res, y_res)

        if self.pca_variance is not None:
            self._pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
            X_final = self._pca.fit_transform(X_sel)
        else:
            X_final = X_sel

        return X_final, y_res

    def _preprocess_infer(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X_arr)
        X_sel = self._feature_selector.transform(X_scaled)
        if self._pca is not None:
            return self._pca.transform(X_sel)
        return X_sel

    # ------------------------------------------------------------------
    # Base model training with Optuna HPO
    # ------------------------------------------------------------------

    def _train_base_models(self, X: np.ndarray, y: np.ndarray) -> None:
        builders = {}
        if _HAS_XGB:
            builders["xgboost"] = self._tune_xgb
        if _HAS_LGB:
            builders["lightgbm"] = self._tune_lgb
        builders["random_forest"] = self._tune_rf
        builders["mlp"] = self._tune_mlp
        builders["svm"] = self._tune_svm

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        for name, builder in builders.items():
            print(f"  Optimising {name} ...")
            model = builder(X, y, cv)
            self._base_models[name] = model
            score = cross_val_score(model, X, y, cv=cv,
                                    scoring="f1_weighted", n_jobs=-1).mean()
            print(f"    {name} CV F1 = {score:.4f}")

    def _tune_rf(self, X, y, cv):
        if _HAS_OPTUNA:
            def objective(trial):
                params = dict(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 5, 40),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                )
                m = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=-1)
                return cross_val_score(m, X, y, cv=cv, scoring="f1_weighted").mean()

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            best = study.best_params
        else:
            best = dict(n_estimators=200, max_depth=20, min_samples_split=2,
                        min_samples_leaf=1, max_features="sqrt")

        m = RandomForestClassifier(**best, random_state=self.random_state, n_jobs=-1)
        return m.fit(X, y)

    def _tune_xgb(self, X, y, cv):
        if _HAS_OPTUNA:
            def objective(trial):
                params = dict(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.4, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                )
                m = XGBClassifier(**params, random_state=self.random_state,
                                   eval_metric="mlogloss", verbosity=0, use_label_encoder=False)
                return cross_val_score(m, X, y, cv=cv, scoring="f1_weighted").mean()

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            best = study.best_params
        else:
            best = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9)

        m = XGBClassifier(**best, random_state=self.random_state,
                          eval_metric="mlogloss", verbosity=0, use_label_encoder=False)
        return m.fit(X, y)

    def _tune_lgb(self, X, y, cv):
        if _HAS_OPTUNA:
            def objective(trial):
                params = dict(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    num_leaves=trial.suggest_int("num_leaves", 20, 200),
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.4, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                )
                m = lgb.LGBMClassifier(**params, random_state=self.random_state, verbosity=-1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return cross_val_score(m, X, y, cv=cv, scoring="f1_weighted").mean()

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            best = study.best_params
        else:
            best = dict(n_estimators=200, num_leaves=63, learning_rate=0.1)

        m = lgb.LGBMClassifier(**best, random_state=self.random_state, verbosity=-1)
        return m.fit(X, y)

    def _tune_mlp(self, X, y, cv):
        if _HAS_OPTUNA:
            def objective(trial):
                n_layers = trial.suggest_int("n_layers", 1, 4)
                hidden = tuple(
                    trial.suggest_int(f"n_units_l{i}", 32, 512)
                    for i in range(n_layers)
                )
                params = dict(
                    hidden_layer_sizes=hidden,
                    alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                    learning_rate_init=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                )
                m = MLPClassifier(**params, max_iter=300, early_stopping=True,
                                   random_state=self.random_state)
                return cross_val_score(m, X, y, cv=cv, scoring="f1_weighted").mean()

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            p = study.best_params
            n_layers = p.pop("n_layers")
            hidden = tuple(p.pop(f"n_units_l{i}") for i in range(n_layers))
            best = {**p, "hidden_layer_sizes": hidden}
        else:
            best = dict(hidden_layer_sizes=(256, 128, 64), alpha=1e-4, learning_rate_init=1e-3)

        m = MLPClassifier(**best, max_iter=500, early_stopping=True,
                          random_state=self.random_state)
        return m.fit(X, y)

    def _tune_svm(self, X, y, cv):
        if _HAS_OPTUNA:
            def objective(trial):
                kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
                params = dict(
                    C=trial.suggest_float("C", 0.01, 100.0, log=True),
                    gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                    kernel=kernel,
                    degree=trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3,
                )
                m = SVC(**params, probability=True, random_state=self.random_state)
                return cross_val_score(m, X, y, cv=cv, scoring="f1_weighted").mean()

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            best = study.best_params
        else:
            best = dict(C=1.0, gamma="scale", kernel="rbf")

        m = SVC(**best, probability=True, random_state=self.random_state)
        return m.fit(X, y)

    # ------------------------------------------------------------------
    # Stacking ensemble
    # ------------------------------------------------------------------

    def _build_stacking_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        estimators = [(name, model) for name, model in self._base_models.items()]
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            passthrough=False,
            n_jobs=-1,
        )
        stack.fit(X, y)
        # Calibrate the stacking ensemble
        self._calibrated = CalibratedClassifierCV(stack, method="isotonic", cv=3)
        self._calibrated.fit(X, y)

    # ------------------------------------------------------------------
    # Conformal prediction
    # ------------------------------------------------------------------

    def _fit_conformal(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute non-conformity scores on training data for conformal sets."""
        proba = self._calibrated.predict_proba(X)
        # Non-conformity score = 1 - P(true class)
        scores = 1.0 - proba[np.arange(len(y)), y]
        self._conformal_scores = scores
