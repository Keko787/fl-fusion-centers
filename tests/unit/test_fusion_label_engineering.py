"""Phase A.2 — label engineering tests for Communities and Crime.

Pins the math the rest of Phase A depends on:

* ``threat_class`` is the argmax of normalized family rates; the
  ``other`` fallback fires when all three family rates are zero.
* ``escalation_score`` is min-max scaled to [0, 1] and respects the
  weights (violent > property > other).
* ``engineer_labels`` drops raw rate columns when asked and keeps them
  when told not to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLabelEngineering import (
    DEFAULT_ESCALATION_WEIGHTS,
    derive_escalation_score,
    derive_threat_class,
    engineer_labels,
)


def _make_row(murd=0, rapes=0, robb=0, assault=0,
              burgl=0, larc=0, autoth=0, arson=0):
    return {
        "murdPerPop": murd, "rapesPerPop": rapes,
        "robbbPerPop": robb, "assaultPerPop": assault,
        "burglPerPop": burgl, "larcPerPop": larc,
        "autoTheftPerPop": autoth, "arsonsPerPop": arson,
    }


def test_threat_class_per_family_zscore_assigns_correctly():
    """Per-family z-score normalization (Phase E §9.10 fix): each
    community gets the family it is most extreme in *relative to the
    dataset average*, not the family with highest raw count.

    Construct 3 communities where each is clearly extreme in a
    different family while the others are at moderate / typical rates.
    """
    rows = [
        # 0: violent-extreme (high violent, moderate property + arson)
        _make_row(murd=1000, rapes=500, robb=500, assault=500,
                  burgl=100, larc=100, autoth=100, arson=10),
        # 1: property-extreme (high property, moderate violent + arson)
        _make_row(murd=50, rapes=50, robb=50, assault=50,
                  burgl=2000, larc=3000, autoth=1000, arson=10),
        # 2: arson-extreme (high arson, moderate everything else)
        _make_row(murd=50, rapes=50, robb=50, assault=50,
                  burgl=100, larc=100, autoth=100, arson=500),
    ]
    df = pd.DataFrame(rows)
    classes = derive_threat_class(df)
    assert classes.iloc[0] == 0  # violent
    assert classes.iloc[1] == 1  # property
    assert classes.iloc[2] == 2  # other (arson)


def test_threat_class_z_score_beats_raw_magnitude():
    """The key correctness property: a community with HIGH violent
    crime but ABSOLUTELY HIGHER property crime should still be labeled
    VIOLENT, because it's extreme on violent relative to the dataset.
    Pre-fix (per-row argmax of raw rates), this row would have been
    labeled property — the bug that motivated this rewrite."""
    rows = [
        # Average community — sets the population baseline.
        _make_row(murd=50, rapes=50, robb=50, assault=50,
                  burgl=2000, larc=3000, autoth=1000, arson=10),
        _make_row(murd=50, rapes=50, robb=50, assault=50,
                  burgl=2000, larc=3000, autoth=1000, arson=10),
        # Spike-violent community: violent is 10x average, property is
        # exactly average (still absolutely higher than violent in raw).
        _make_row(murd=500, rapes=500, robb=500, assault=500,
                  burgl=2000, larc=3000, autoth=1000, arson=10),
    ]
    df = pd.DataFrame(rows)
    classes = derive_threat_class(df)
    assert classes.iloc[2] == 0, (
        "Spike-violent community labeled wrong — z-score should beat "
        "raw magnitude here"
    )


def test_threat_class_all_zero_falls_back_to_other():
    """Rows with zero crime in every family deterministically map to
    class 2 (other) regardless of dataset-level z-score statistics."""
    rows = [
        _make_row(murd=100, rapes=50, robb=100, assault=100,
                  burgl=500, larc=600, autoth=200, arson=20),
        _make_row(),  # all zeros
    ]
    df = pd.DataFrame(rows)
    classes = derive_threat_class(df)
    assert classes.iloc[1] == 2


def test_threat_class_produces_balanced_distribution_on_real_like_data():
    """Real UCI data has property rates ~10x violent and ~100x arson by
    raw magnitude; the pre-fix per-row argmax collapsed everything to
    class 1 (property). The z-score fix produces a non-degenerate
    distribution across all three classes."""
    rng = np.random.default_rng(0)
    n = 200
    rows = []
    for _ in range(n):
        # Realistic per-100k rates with random per-community emphasis.
        rows.append(_make_row(
            murd=rng.uniform(0, 50),
            rapes=rng.uniform(0, 80),
            robb=rng.uniform(0, 250),
            assault=rng.uniform(0, 500),
            burgl=rng.uniform(500, 2000),
            larc=rng.uniform(1000, 3500),
            autoth=rng.uniform(200, 1000),
            arson=rng.uniform(0, 100),
        ))
    df = pd.DataFrame(rows)
    classes = derive_threat_class(df)
    distribution = classes.value_counts().to_dict()
    # All three classes should be represented.
    assert all(c in distribution for c in (0, 1, 2)), (
        f"Single-class collapse — labels are {distribution}"
    )
    # No class should claim >90% of the dataset (would suggest a bias).
    assert max(distribution.values()) < 0.9 * n


def test_escalation_score_in_unit_interval():
    df = pd.DataFrame([
        _make_row(murd=10, assault=20),
        _make_row(burgl=100, larc=200),
        _make_row(arson=5),
        _make_row(),
    ])
    scores = derive_escalation_score(df)
    assert (scores >= 0).all() and (scores <= 1).all()
    # At least one row hits each extreme (highest-severity row is 1.0,
    # zero-rate row is 0.0).
    assert scores.min() == pytest.approx(0.0, abs=1e-6)
    assert scores.max() == pytest.approx(1.0, abs=1e-6)


def test_escalation_respects_weights():
    # Two rows with identical raw-rate sums but split across families.
    # Violent-heavy row should rank higher than property-heavy.
    df = pd.DataFrame([
        _make_row(murd=1000),     # violent_rate=1000, escalation = 0.6*1000
        _make_row(burgl=1000),    # property_rate=1000, escalation = 0.3*1000
    ])
    scores = derive_escalation_score(df, DEFAULT_ESCALATION_WEIGHTS)
    assert scores.iloc[0] > scores.iloc[1]


def test_engineer_labels_drops_rate_columns_by_default():
    df = pd.DataFrame([_make_row(murd=10, burgl=20, arson=3)])
    out = engineer_labels(df)
    assert "threat_class" in out.columns
    assert "escalation_score" in out.columns
    assert "murdPerPop" not in out.columns
    assert "burglPerPop" not in out.columns


def test_engineer_labels_keeps_rates_when_asked():
    df = pd.DataFrame([_make_row(murd=10)])
    out = engineer_labels(df, drop_rate_columns=False)
    assert "murdPerPop" in out.columns
    assert "threat_class" in out.columns


def test_engineer_labels_degenerate_dataset_constant_escalation():
    df = pd.DataFrame([_make_row(murd=10), _make_row(murd=10)])
    out = engineer_labels(df)
    assert (out["escalation_score"] == 0.5).all()
