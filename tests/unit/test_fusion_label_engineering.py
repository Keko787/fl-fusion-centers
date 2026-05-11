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


def test_threat_class_violent_dominant():
    df = pd.DataFrame([_make_row(murd=100, rapes=50, robb=200, assault=400)])
    assert derive_threat_class(df).iloc[0] == 0  # violent


def test_threat_class_property_dominant():
    df = pd.DataFrame([_make_row(burgl=500, larc=800, autoth=300, murd=5)])
    assert derive_threat_class(df).iloc[0] == 1  # property


def test_threat_class_other_dominant():
    df = pd.DataFrame([_make_row(arson=200, murd=5, burgl=10)])
    assert derive_threat_class(df).iloc[0] == 2  # other


def test_threat_class_all_zero_falls_back_to_other():
    df = pd.DataFrame([_make_row()])
    assert derive_threat_class(df).iloc[0] == 2


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
