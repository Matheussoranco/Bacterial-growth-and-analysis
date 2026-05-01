"""
Bacterial Growth Simulation — entry-point script.

Run the full SOTA pipeline end-to-end:
    python Bacterial_Growth_Simulation.py --mode pipeline

Quick component demos (no heavy training):
    python Bacterial_Growth_Simulation.py --mode demo

Or import individual components directly:
    from src.bacterial_growth import BacterialCultureClassifier, fit_growth_curve
"""

import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bacterial_growth.pipeline import BacterialAnalysisPipeline
from bacterial_growth.kinetics import fit_growth_curve
from bacterial_growth.data import generate_classification_dataset
from bacterial_growth.classifier import BacterialCultureClassifier
from bacterial_growth.growth_predictor import CardinalParameterModel

import numpy as np
from sklearn.model_selection import train_test_split


def demo_kinetics() -> None:
    """Demonstrate growth curve fitting on a single synthetic culture."""
    print("\n--- Growth Curve Fitting Demo ---")
    from bacterial_growth.kinetics import baranyi_roberts

    t = np.linspace(0, 24, 25)
    log_n = baranyi_roberts(t, 0.8, 2.5, 3.0, 9.0) + np.random.default_rng(0).normal(0, 0.1, 25)

    kin = fit_growth_curve(t, log_n)
    print(f"  Best model     : {kin.model_name}")
    print(f"  µmax           : {kin.mu_max:.4f} /h (true: 0.8000)")
    print(f"  Lag time       : {kin.lag_time:.4f} h  (true: 2.5000)")
    print(f"  Generation time: {kin.generation_time:.4f} h")
    print(f"  R²             : {kin.r_squared:.4f}")


def demo_cardinal_model() -> None:
    """Illustrate Cardinal Parameter Model predictions."""
    print("\n--- Cardinal Parameter Model Demo ---")
    cpm = CardinalParameterModel("E. coli")
    print(f"  E. coli cardinal T : {cpm.cardinal_temperatures}")
    print(f"  E. coli cardinal pH: {cpm.cardinal_pH}")
    mu_37_7 = cpm.mu_max(37.0, 7.0)
    mu_10_6 = cpm.mu_max(10.0, 6.0)
    print(f"  µmax at 37°C / pH 7.0 = {float(mu_37_7):.4f} /h")
    print(f"  µmax at 10°C / pH 6.0 = {float(mu_10_6):.4f} /h  (cold stress)")


def demo_classifier() -> None:
    """Train and evaluate the stacking classifier on generated data."""
    print("\n--- Species Classifier Demo ---")
    df = generate_classification_dataset(n_samples=800, random_state=42)
    X = df.drop(columns=["species"])
    y = df["species"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = BacterialCultureClassifier(n_trials=10, random_state=42)
    clf.fit(X_train, y_train)
    clf.evaluate(X_test, y_test)

    print("\n  Conformal prediction sets (90% coverage) for first 5 test samples:")
    pred_sets = clf.predict_set(X_test.head(5), coverage=0.90)
    preds = clf.predict(X_test.head(5))
    for i, (ps, pred, truth) in enumerate(zip(pred_sets, preds, y_test[:5])):
        print(f"    Sample {i+1}: truth={truth!r:15s}  point={pred!r:15s}  set={ps}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bacterial Growth Analysis")
    parser.add_argument(
        "--mode",
        choices=["demo", "pipeline"],
        default="demo",
        help="'demo' runs quick component demos; 'pipeline' runs the full pipeline",
    )
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for pipeline mode")
    parser.add_argument("--n-curves", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--optuna-trials", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "demo":
        demo_kinetics()
        demo_cardinal_model()
        demo_classifier()
    else:
        pipe = BacterialAnalysisPipeline(
            output_dir=args.output_dir,
            n_curves=args.n_curves,
            n_samples=args.n_samples,
            n_optuna_trials=args.optuna_trials,
        )
        metrics = pipe.run()
        print(f"\nFinal weighted F1 : {metrics.get('f1_weighted', 'N/A'):.4f}")
        print(f"Final ROC-AUC     : {metrics.get('roc_auc', 'N/A'):.4f}")
