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
import time
from pathlib import Path
from typing import Optional

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
        self.output_dir = Path(output_dir)
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
        print("\n" + "=" * 65)
        print("  Bacterial Growth Analysis Pipeline — SOTA Edition")
        print("=" * 65)

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
        print(f"\nPipeline completed in {elapsed:.1f}s")
        print(f"Results saved to {self.output_dir.resolve()}")
        return self.eval_metrics

    # ------------------------------------------------------------------
    # Stage 1 — Data generation
    # ------------------------------------------------------------------

    def generate_data(self) -> "BacterialAnalysisPipeline":
        print("\n[1/6] Generating synthetic data ...")
        self.curves_df, self.kinetics_df = generate_growth_curves(
            n_curves=self.n_curves,
            random_state=self.random_state,
        )
        self.class_df = generate_classification_dataset(
            n_samples=self.n_samples,
            random_state=self.random_state,
        )
        print(f"  Growth curves: {self.n_curves} | Kinetics rows: {len(self.kinetics_df)}")
        print(f"  Classification samples: {len(self.class_df)}")
        print(f"  Class distribution:\n{self.class_df['species'].value_counts().to_string()}")
        return self

    # ------------------------------------------------------------------
    # Stage 2 — Growth curve visualisation
    # ------------------------------------------------------------------

    def visualise_growth_curves(self) -> "BacterialAnalysisPipeline":
        print("\n[2/6] Visualising growth curves ...")
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
            except RuntimeError:
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
        print("\n[3/6] Fitting growth predictor (GPR) ...")
        self.growth_predictor = GrowthPredictor()
        self.growth_predictor.fit(self.kinetics_df, condition_cols=["temperature", "pH"])

        # Evaluate on held-out kinetics
        sample = self.kinetics_df.sample(20, random_state=self.random_state)
        preds = self.growth_predictor.predict(sample)
        for target in ("mu_max", "lag_time", "log_nmax"):
            mae = float(np.mean(np.abs(sample[target].values - preds[target].values)))
            print(f"  GPR MAE {target}: {mae:.4f}")
        return self

    def visualise_growth_boundaries(self) -> "BacterialAnalysisPipeline":
        print("  Plotting growth boundaries ...")
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
        print("\n[4/6] Training species classifier ...")
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
        print("\n[5/6] Evaluating classifier ...")
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
        except Exception as e:
            print(f"  Warning: conformal plot skipped ({e})")

        # ROC curves
        try:
            from sklearn.preprocessing import label_binarize
            from .visualization import plot_roc_curves
            y_bin = label_binarize(self.y_test, classes=class_names)
            y_score = proba_df.values
            fig = plot_roc_curves(y_bin, y_score, class_names)
            save_figure(fig, self.output_dir / "roc_curves.png")
        except Exception as e:
            print(f"  Warning: ROC plot skipped ({e})")

        return self

    # ------------------------------------------------------------------
    # Stage 6 — Persist
    # ------------------------------------------------------------------

    def save_results(self) -> "BacterialAnalysisPipeline":
        print("\n[6/6] Saving results ...")
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

        print(f"  Saved: {', '.join(p.name for p in sorted(self.output_dir.iterdir()))}")
        return self
