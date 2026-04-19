from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self, config: dict):
        self.target_col = config["data"]["target_col"]
        self.categorical_cols = config["data"]["categorical_cols"]
        self.numeric_cols = config["data"]["numeric_cols"]
        self.drop_cols = config["data"]["drop_cols"]

        self.categorical_cols = [
            c for c in self.categorical_cols if c not in self.drop_cols
        ]
        self.numeric_cols = [c for c in self.numeric_cols if c not in self.drop_cols]

        self.preprocessor: Any = None

    def prepare_data(self, df: pd.DataFrame, is_train: bool = True):
        import warnings

        # target is present if is_train=True
        if self.target_col in df.columns:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
        else:
            X = df.copy()
            y = None

        # Drop columns if it exists
        if self.drop_cols:
            X = X.drop(columns=self.drop_cols, errors="ignore")

        if is_train:
            # Dynamically include engineered features not listed in config
            for col in X.columns:
                in_metrics = col not in self.numeric_cols
                in_cats = col not in self.categorical_cols
                if in_metrics and in_cats:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        self.numeric_cols.append(col)
                    else:
                        self.categorical_cols.append(col)

        # Force categorical columns to string to ensure consistent ONNX typing
        for c in self.categorical_cols:
            if c in X.columns:
                X[c] = X[c].astype(str)  # pandas NaN becomes the string "nan"

        if is_train:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent", missing_values="nan"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            dtype="float32",
                        ),
                    ),
                ]
            )

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, self.numeric_cols),
                    ("cat", categorical_transformer, self.categorical_cols),
                ]
            )
            X_transformed = self.preprocessor.fit_transform(X)
        else:
            if self.preprocessor is None or not hasattr(
                self.preprocessor, "transformers_"
            ):
                raise ValueError(
                    "Preprocessor not fitted yet. \
                                  Call with is_train=True first."
                )
            X_transformed = self.preprocessor.transform(X)

        # Get feature names after one-hot encoding
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cat_features = (
                    self.preprocessor.named_transformers_["cat"]
                    .named_steps["onehot"]
                    .get_feature_names_out(self.categorical_cols)
                )
            all_features = self.numeric_cols + list(cat_features)
        except Exception:
            all_features = None  # Fallback

        if isinstance(X_transformed, pd.DataFrame):
            X_processed = X_transformed
            if all_features is not None and len(X_processed.columns) == len(
                all_features
            ):
                X_processed.columns = all_features
        else:
            X_processed = pd.DataFrame(X_transformed, columns=all_features)

        return X_processed, y

    def save(self, filepath: str):
        state = {
            "preprocessor": self.preprocessor,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
        }
        joblib.dump(state, filepath)

    def load(self, filepath: str):
        state = joblib.load(filepath)
        if isinstance(state, dict) and "preprocessor" in state:
            self.preprocessor = state["preprocessor"]
            self.numeric_cols = state["numeric_cols"]
            self.categorical_cols = state["categorical_cols"]
        else:
            self.preprocessor = state


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    from ml_system.data.loader import load_config, load_data
    from ml_system.features.engineering import create_features

    print("Testing preprocessor.py individually...")
    config = load_config()
    df = load_data(config["data"]["train_path"])
    print("Original cols:", len(df.columns))
    df = create_features(df)
    print("Engineered cols:", len(df.columns))

    p = DataPreprocessor(config)
    X_processed, y_processed = p.prepare_data(df, is_train=True)
    print("Successfully created DataPreprocessor instance.")
    print("X_processed shape:", X_processed.shape)
    print("y_processed shape:", y_processed.shape)
    print("X_processed columns:", X_processed.columns.tolist()[:5], "...")
    print(
        "y_processed name:",
        y_processed.name if y_processed is not None else None,
    )
    print("X_processed head:\n", X_processed.head())
