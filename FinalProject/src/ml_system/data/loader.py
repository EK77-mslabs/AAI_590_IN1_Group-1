import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    print("Testing loader.py individually...")
    try:
        config_file = str(Path(project_root) / "config" / "config.yaml")
        config_data = load_config(config_file)
        print("Successfully loaded config keys:", config_data.keys())
    except Exception as e:
        print("Failed to load config:", e)

    try:
        train_df = load_data(config_data["data"]["train_path"])
        print(train_df.head())
        print("Successfully loaded train.csv")
    except Exception as e:
        print("Failed to load train.csv:", e)

    try:
        test_df = load_data(config_data["data"]["test_path"])
        print(test_df.head())
        print("Successfully loaded test.csv")
    except Exception as e:
        print("Failed to load test.csv:", e)
