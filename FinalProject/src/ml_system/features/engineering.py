import re

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering logic. We extract numeric values from strings
    like max_power and max_torque.
    """
    df_engineered = df.copy()

    # Torque extraction
    df_engineered["torque"] = df_engineered["max_torque"].apply(
        lambda x: float(re.findall(r"\d+\.?\d*", x)[0])
    )
    df_engineered["torque_rpm"] = df_engineered["max_torque"].apply(
        lambda x: float(re.findall(r"\d+\.?\d*", x)[1])
    )

    # Power extraction
    df_engineered["power"] = df_engineered["max_power"].apply(
        lambda x: float(re.findall(r"\d+\.?\d*", x)[0])
    )
    df_engineered["power_rpm"] = df_engineered["max_power"].apply(
        lambda x: float(re.findall(r"\d+\.?\d*", x)[1])
    )

    # Add safety score
    safety_cols = ["airbags", "is_esc", "is_tpms", "is_brake_assist"]

    df_engineered[safety_cols] = df_engineered[safety_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    # Fill any NaNs with 0 and convert to int
    df_engineered[safety_cols] = df_engineered[safety_cols].fillna(0)
    df_engineered[safety_cols] = df_engineered[safety_cols].astype(int)
    df_engineered["safety_score"] = df_engineered[safety_cols].sum(axis=1)

    return df_engineered


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    from ml_system.data.loader import load_config, load_data

    print("Testing preprocessor.py individually...")
    config = load_config()
    df_test = load_data(config["data"]["train_path"])
    print("Testing engineering.py individually...")
    print("Original DataFrame:\n", df_test.info())
    df_out = create_features(df_test)
    print("Engineered DataFrame:\n", df_out.info())
