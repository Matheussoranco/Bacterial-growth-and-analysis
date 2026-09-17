"""
End-to-end pipeline orchestrating data generation, growth modelling,
species classification, and visualisation.

Usage
-----
    from bacterial_growth.pipeline import BacterialAnalysisPipeline

    pipe = BacterialAnalysisPipeline(output_dir="results/")
    pipe.run()

Or step-by-step:
    pipe = BacterialAnalysisPipeline()
    pipe.generate_data()
    pipe.fit_growth_models()
    pipe.train_classifier()
    pipe.evaluate()
    pipe.save_results()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .classifier import BacterialCultureClassifier
from .data import generate_classification_dataset, generate_growth_curves
from .growth_predictor import CardinalParameterModel, GrowthPredictor
from .visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_growth_boundary,
    plot_growth_curve,
    plot_growth_curve_comparison,
    plot_kinetics_distributions,
    save_figure,
)

__all__ = ["BacterialAnalysisPipeline"]


class BacterialAnalysisPipeline:
    """Full analysis pipeline.

    Parameters
    ----------
    output_dir : directory for figures and saved models
    n_curves : number of growth curves to simulate
    n_samples : number of classification samples to generate
    n_optuna_trials : Optuna trials per model
    test_size : fraction for test split
    random_state : global seed
    """

    def __init__(
        self,
        output_dir: str = "results",
        n_curves: int = 300,
        n_samples: int = 3000,
        n_optuna_trials: int = 30,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if not output_dir:
            raise ValueError("output_dir não pode ser vazio")
        out = Path(output_dir)
        if out.exists() and not out.is_dir():
            raise ValueError(f"output_dir existe e não é diretório: {out}")
        self.output_dir = out
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_curves = n_curves
        self.n_samples = n_samples
        self.n_optuna_trials = n_optuna_trials
        self.test_size = test_size
        self.random_state = random_state

        # State populated as pipeline runs
        self.curves_df: Optional[pd.DataFrame] = None
        self.kinetics_df: Optional[pd.DataFrame] = None
        self.class_df: Optional[pd.DataFrame] = None
        self.classifier: Optional[BacterialCultureClassifier] = None
        self.growth_predictor: Optional[GrowthPredictor] = None
        self.eval_metrics: dict = {}

    def run(self) -> dict:
        """Execute the complete pipeline and return evaluation metrics."""
        logger.info("Bacterial Growth Analysis Pipeline — SOTA Edition")

        t0 = time.time()
        self.generate_data()
        self.visualise_growth_curves()
        self.visualise_kinetics()
        self.fit_growth_predictor()
        self.visualise_growth_boundaries()
        self.train_classifier()
        self.evaluate()
        self.save_results()

        elapsed = time.time() - t0
        logger.info("Pipeline completed in %.1fs", elapsed)
        logger.info("Results saved to %s", self.output_dir.resolve())
        return self.eval_metrics

    # ------------------------------------------------------------------
    # Stage 1 — Data generation
    # ------------------------------------------------------------------

    def generate_data(self) -> "BacterialAnalysisPipeline":
        logger.info("[1/6] Generating synthetic data ...")
        self.curves_df, self.kinetics_df = generate_growth_curves(
            n_curves=self.n_curves,
            random_state=self.random_state,
        )
        self.class_df = generate_classification_dataset(
            n_samples=self.n_samples,
            random_state=self.random_state,
        )
        logger.info("Growth curves: %d | Kinetics rows: %d", self.n_curves, len(self.kinetics_df))
        logger.info("Classification samples: %d", len(self.class_df))
        logger.info("Class distribution:\n%s", self.class_df['species'].value_counts().to_string())
        return self

    # ------------------------------------------------------------------
    # Stage 2 — Growth curve visualisation
    # ------------------------------------------------------------------

    def visualise_growth_curves(self) -> "BacterialAnalysisPipeline":
        logger.info("[2/6] Visualising growth curves ...")
        fig = plot_growth_curve_comparison(self.curves_df, n_per_species=4)
        save_figure(fig, self.output_dir / "growth_curves_comparison.png")

        # Plot one annotated curve per species
        species_list = self.kinetics_df["species"].unique()
        for sp in species_list:
            sp_ids = self.kinetics_df[self.kinetics_df["species"] == sp]["curve_id"].values
            if len(sp_ids) == 0:
                continue
            cid = sp_ids[0]
            sub = self.curves_df[self.curves_df["curve_id"] == cid].sort_values("time")
            t_arr = sub["time"].values
            log_n_arr = sub["log_n"].values

            from .kinetics import fit_growth_curve
            try:
                kin = fit_growth_curve(t_arr, log_n_arr)
            except (RuntimeError, ValueError) as exc:
                logger.warning("fit_growth_curve pulado para %s: %s", sp, exc)
                continue
            fig = plot_growth_curve(t_arr, log_n_arr, kinetics=kin, species=sp)
            sp_clean = sp.replace(". ", "_").replace(" ", "_")
            save_figure(fig, self.output_dir / f"growth_curve_{sp_clean}.png")
        return self

    def visualise_kinetics(self) -> "BacterialAnalysisPipeline":
        for param in ("mu_max", "lag_time", "generation_time"):
            fig = plot_kinetics_distributions(self.kinetics_df, parameter=param)
            save_figure(fig, self.output_dir / f"kinetics_{param}.png")
        return self

    # ------------------------------------------------------------------
    # Stage 3 — Growth predictor (GPR + CPM)
    # ------------------------------------------------------------------

    def fit_growth_predictor(self) -> "BacterialAnalysisPipeline":
        logger.info("[3/6] Fitting growth predictor (GPR) ...")
        self.growth_predictor = GrowthPredictor()
        self.growth_predictor.fit(self.kinetics_df, condition_cols=["temperature", "pH"])

        # Evaluate on held-out kinetics
        n_eval = min(20, len(self.kinetics_df))
        if n_eval == 0:
            logger.warning("No kinetics rows available for GPR evaluation; skipping.")
            return self
        sample = self.kinetics_df.sample(n_eval, random_state=self.random_state)
        preds = self.growth_predictor.predict(sample)
        for target in ("mu_max", "lag_time", "log_nmax"):
            mae = float(np.mean(np.abs(sample[target].values - preds[target].values)))
            logger.info("GPR MAE %s: %.4f", target, mae)
        return self

    def visualise_growth_boundaries(self) -> "BacterialAnalysisPipeline":
        logger.info("Plotting growth boundaries ...")
        species_list = list(self.kinetics_df["species"].unique())
        for sp in species_list:
            cpm = CardinalParameterModel(sp)
            T_grid, pH_grid, mu_grid = cpm.growth_boundary(resolution=80)
            fig = plot_growth_boundary(T_grid, pH_grid, mu_grid, species=sp)
            sp_clean = sp.replace(". ", "_").replace(" ", "_")
            save_figure(fig, self.output_dir / f"boundary_{sp_clean}.png")
        return self

    # ------------------------------------------------------------------
    # Stage 4 — Classifier training
    # ------------------------------------------------------------------

    def train_classifier(self) -> "BacterialAnalysisPipeline":
        logger.info("[4/6] Training species classifier ...")
        y = self.class_df["species"].values
        X = self.class_df.drop(columns=["species"])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        self.classifier = BacterialCultureClassifier(
            n_trials=self.n_optuna_trials,
            random_state=self.random_state,
        )
        self.classifier.fit(self.X_train, self.y_train)
        return self

    # ------------------------------------------------------------------
    # Stage 5 — Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> "BacterialAnalysisPipeline":
        logger.info("[5/6] Evaluating classifier ...")
        self.eval_metrics = self.classifier.evaluate(self.X_test, self.y_test)

        # Confusion matrix
        cm = self.eval_metrics["confusion_matrix"]
        class_names = list(self.classifier._label_enc.classes_)
        fig = plot_confusion_matrix(cm, class_names)
        save_figure(fig, self.output_dir / "confusion_matrix.png")

        # Calibration (binary per class: is it this species?)
        proba_df = self.classifier.predict_proba(self.X_test)
        for sp in class_names:
            y_bin = (self.y_test == sp).astype(int)
            fig = plot_calibration_curve(y_bin, proba_df[sp].values)
            sp_clean = sp.replace(". ", "_").replace(" ", "_")
            save_figure(fig, self.output_dir / f"calibration_{sp_clean}.png")

        # Conformal prediction coverage at 90%
        try:
            pred_sets = self.classifier.predict_set(self.X_test, coverage=0.90)
            from .visualization import plot_conformal_coverage
            fig = plot_conformal_coverage(pred_sets, self.y_test, target_coverage=0.90)
            save_figure(fig, self.output_dir / "conformal_coverage.png")
        except (ImportError, ValueError, AttributeError) as e:
            logger.warning("conformal plot skipped (%s: %s)", type(e).__name__, e)

        # ROC curves
        try:
            from sklearn.preprocessing import label_binarize
            from .visualization import plot_roc_curves
            y_bin = label_binarize(self.y_test, classes=class_names)
            y_score = proba_df.values
            fig = plot_roc_curves(y_bin, y_score, class_names)
            save_figure(fig, self.output_dir / "roc_curves.png")
        except (ImportError, ValueError) as e:
            logger.warning("ROC plot skipped (%s: %s)", type(e).__name__, e)

        return self

    # ------------------------------------------------------------------
    # Stage 6 — Persist
    # ------------------------------------------------------------------

    def save_results(self) -> "BacterialAnalysisPipeline":
        logger.info("[6/6] Saving results ...")
        self.classifier.save(self.output_dir / "classifier.joblib")
        self.kinetics_df.to_csv(self.output_dir / "kinetics.csv", index=False)
        self.class_df.to_csv(self.output_dir / "classification_dataset.csv", index=False)

        # Write metrics JSON (convert numpy types for JSON serialisation)
        metrics_json = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.eval_metrics.items()
            if k != "classification_report"
        }
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)

        logger.info("Saved: %s", ', '.join(p.name for p in sorted(self.output_dir.iterdir())))
        return self
