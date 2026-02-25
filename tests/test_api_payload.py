"""Validate Pydantic request schema in serve.py accepts list-of-dicts. No network calls."""

import pytest
from pydantic import ValidationError

from src.serve import PredictRequest


def test_accepts_list_of_dicts():
    payload = {
        "data": [
            {"feature_a": 1.0, "feature_b": 2.0, "category": "A"},
            {"feature_a": 3.0, "feature_b": 4.0, "category": "B"},
        ],
    }
    req = PredictRequest(**payload)
    assert len(req.data) == 2
    assert req.data[0]["feature_a"] == 1.0
    assert req.data[1]["category"] == "B"


def test_accepts_mixed_numeric_and_string_values():
    payload = {"data": [{"x": 1, "y": "foo", "z": 2.5}]}
    req = PredictRequest(**payload)
    assert req.data[0]["x"] == 1
    assert req.data[0]["y"] == "foo"
    assert req.data[0]["z"] == 2.5


def test_rejects_empty_data():
    with pytest.raises(ValidationError):
        PredictRequest(data=[])


def test_rejects_missing_data_key():
    with pytest.raises(ValidationError):
        PredictRequest(**{})
