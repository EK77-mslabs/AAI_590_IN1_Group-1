import pandas as pd

from ml_system.features.engineering import create_features


def test_feature_engineering():
    df = pd.DataFrame(
        {
            "max_power": ["88.50bhp@6000rpm", "40.36bhp@6000rpm"],
            "max_torque": ["113Nm@4400rpm", "60Nm@3500rpm"],
            "airbags": [2, 6],
            "is_esc": [1, 0],
            "is_tpms": [0, 1],
            "is_brake_assist": [1, 1],
        }
    )

    df_engineered = create_features(df)

    assert "power" in df_engineered.columns
    assert "torque" in df_engineered.columns
    assert "safety_score" in df_engineered.columns
    assert df_engineered["power"].iloc[0] == 88.5
    assert df_engineered["torque"].iloc[1] == 60.0
