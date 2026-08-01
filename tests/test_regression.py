"""
Focused regression tests for bugs found in a codebase audit (see git history).

There was no test suite prior to this file. These tests target the two
concrete bugs that were fixed, not general coverage:

  1. features.py used ``np.trapz``, which was removed in NumPy 2.0 (renamed
     to ``np.trapezoid``) -> AttributeError at runtime on any modern NumPy.
  2. classifier.BacterialCultureClassifier fit conformal non-conformity
     scores on the same data used to fit/calibrate the ensemble
     (train/calibration leakage) instead of a held-out split.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bacterial_growth.features import StatisticalFeatureExtractor, extract_all_features
from bacterial_growth.data import generate_classification_dataset
from bacterial_growth.classifier import BacterialCultureClassifier


def test_statistical_feature_extractor_does_not_crash_on_trapz():
    """Regression test for np.trapz -> np.trapezoid (NumPy 2.0 removal)."""
    t = np.linspace(0, 24, 25)
    log_n = np.linspace(3, 9, 25) + np.random.default_rng(0).normal(0, 0.1, 25)

    df = StatisticalFeatureExtractor().transform([(t, log_n)])

    assert len(df) == 1
    assert np.isfinite(df.loc[0, "stat_auc"])
    assert np.isfinite(df.loc[0, "stat_early_auc"])
    assert np.isfinite(df.loc[0, "stat_late_auc"])
    assert np.isfinite(df.loc[0, "stat_auc_positive_rate"])


def test_extract_all_features_combines_kinetic_and_statistical():
    t = np.linspace(0, 24, 25)
    log_n = np.linspace(3, 9, 25) + np.random.default_rng(1).normal(0, 0.1, 25)
    df = extract_all_features([(t, log_n)])
    # 8 kinetic + 20 statistical features
    assert df.shape == (1, 28)


def test_conformal_scores_come_from_held_out_split():
    """Regression test: conformal non-conformity scores must be computed on a
    calibration split the ensemble never trained/calibrated on, not on the
    same data passed to fit(). A leaked (in-sample) implementation would use
    the full input as calibration data.
    """
    df = generate_classification_dataset(n_samples=200, random_state=0)
    X = df.drop(columns=["species"])
    y = df["species"].values

    clf = BacterialCultureClassifier(n_trials=1, cv_folds=2, random_state=0)
    clf.fit(X, y)

    n_cal = len(clf._conformal_scores)
    # ~15% of 200 held out for calibration -- strictly less than the full
    # input, proving a split happened rather than reusing the training data.
    assert n_cal < len(X)
    assert 15 <= n_cal <= 45  # ballpark around test_size=0.15 of 200

    # Scores are valid non-conformity values in [0, 1].
    assert np.all(clf._conformal_scores >= 0.0)
    assert np.all(clf._conformal_scores <= 1.0)
