import pandas as pd
import pytest

from app import engineer_features


def test_engineer_features_adds_expected_booking_signals():
    raw = pd.DataFrame({
        "stays_in_weekend_nights": [2],
        "stays_in_week_nights": [3],
        "adults": [2],
        "children": [1],
        "babies": [0],
        "previous_cancellations": [1],
        "adr": [150.0],
    })
    result = engineer_features(raw)
    assert result.loc[0, "total_nights"] == 5
    assert result.loc[0, "total_guests"] == 3
    assert result.loc[0, "has_children"] == 1
    assert result.loc[0, "has_babies"] == 0
    assert result.loc[0, "has_previous_cancellations"] == 1
    assert result.loc[0, "adr_per_guest"] == 50.0


def test_engineer_features_rejects_missing_fields():
    with pytest.raises(ValueError, match="Missing required booking fields"):
        engineer_features(pd.DataFrame({"adr": [100.0]}))
