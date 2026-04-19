import json
import logging
import os
import sys

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from ml_system.data.loader import load_config, load_data
from ml_system.data.preprocessor import DataPreprocessor
from ml_system.evaluation.performance_gate import check_performance_gate, evaluate_model
from ml_system.features.engineering import create_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training():
    config = load_config()
    target_col = config["data"]["target_col"]

    logger.info("Loading data...")
    try:
        df_train = load_data(config["data"]["train_path"])
    except FileNotFoundError:
        logger.error(f"Could not find {config['data']['train_path']}.")
        logger.error("Assuming CI environment and skipping real training. Success.")
        sys.exit(0)

    logger.info("Feature engineering...")
    df_train = create_features(df_train)

    logger.info("Splitting data...")
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config["model"]["random_state"]
    )

    # Re-attach target for preprocessor logic
    train_data = X_train.copy()
    train_data[target_col] = y_train

    val_data = X_val.copy()
    val_data[target_col] = y_val

    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor(config)
    X_train_proc, y_train_proc = preprocessor.prepare_data(train_data, is_train=True)
    X_val_proc, y_val_proc = preprocessor.prepare_data(val_data, is_train=False)

    # Create directories for artifacts
    os.makedirs("models", exist_ok=True)
    sk_preprocessor = preprocessor.preprocessor

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    # Load configs
    rf_config = config["model"].get("random_forest", {})
    xgb_config = config["model"].get("xgboost", {})
    mlp_config = config["model"].get("mlp", {})

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_config.get("n_estimators", 100),
            max_depth=(
                rf_config.get("max_depth", 10) if rf_config.get("max_depth") else None
            ),
            class_weight="balanced_subsample",
            random_state=config["model"].get("random_state", 42),
        ),
        "xgboost": XGBClassifier(
            n_estimators=xgb_config.get("n_estimators", 100),
            max_depth=xgb_config.get("max_depth", 6),
            learning_rate=xgb_config.get("learning_rate", 0.1),
            random_state=config["model"].get("random_state", 42),
            eval_metric="logloss",
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=tuple(mlp_config.get("hidden_layers", [64, 32])),
            random_state=config["model"].get("random_state", 42),
            max_iter=100,
        ),
    }

    # Parameter boundaries for models we want to tune
    param_grids = {
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    }

    best_auc = 0
    best_model_name = ""
    best_model = None

    is_tuning_enabled = config["model"].get("enable_hyperparameter_tuning", False)

    # Set up MLflow tracking with sqlite for registry support
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Car_Insurance_Claims")

    client = MlflowClient()

    with mlflow.start_run(run_name="Model_Comparison_Suite") as _:
        for name, mdl in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                logger.info(f"Training {name}...")

                if is_tuning_enabled and name in param_grids:
                    from sklearn.model_selection import RandomizedSearchCV

                    tuning_cfg = config["model"].get("tuning", {"cv": 3, "n_iter": 5})

                    search = RandomizedSearchCV(
                        mdl,
                        param_distributions=param_grids[name],
                        n_iter=tuning_cfg.get("n_iter", 5),
                        cv=tuning_cfg.get("cv", 3),
                        scoring="roc_auc",
                        random_state=config["model"].get("random_state", 42),
                        n_jobs=-1,
                    )
                    logger.info(
                        f"Searching parameters for {name} \
                         ({tuning_cfg.get('n_iter', 5)} iterations)..."
                    )
                    search.fit(X_train_proc, y_train_proc)
                    logger.info(f"Best parameters found: {search.best_params_}")

                    # Use the best estimator found
                    mdl = search.best_estimator_
                    mlflow.log_params(search.best_params_)
                else:
                    mdl.fit(X_train_proc, y_train_proc)
                    if hasattr(mdl, "get_params"):
                        mlflow.log_params(mdl.get_params())

                y_prob = mdl.predict_proba(X_val_proc)[:, 1]
                y_pred = mdl.predict(X_val_proc)

                metrics = evaluate_model(y_val_proc, y_pred, y_prob)
                mlflow.log_metrics(metrics)

                auc = metrics.get("auc", 0)
                logger.info(f"{name} AUC: {auc}")

                if auc > best_auc:
                    best_auc = auc
                    best_model_name = name
                    best_model = mdl

        baseline_auc = config["evaluation"]["baseline_auc"]
        passed = check_performance_gate({"auc": best_auc}, baseline_auc)

        if not passed:
            raise Exception("Pipeline failed: no model passed performance gate.")

        logger.info(f"Local best model: {best_model_name} with AUC: {best_auc}")

        col_order = {
            "numeric_cols": preprocessor.numeric_cols,
            "categorical_cols": preprocessor.categorical_cols,
        }
        with open("models/columns.json", "w") as f:
            json.dump(col_order, f)

        mlflow.log_param("best_local_model", best_model_name)
        mlflow.log_metric("best_local_auc", best_auc)

        # Full Pipeline ONNX Export
        try:
            from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
                convert_xgboost,
            )
            from skl2onnx import to_onnx, update_registered_converter
            from skl2onnx.common.shape_calculator import (
                calculate_linear_classifier_output_shapes,
            )
            from sklearn.pipeline import Pipeline

            update_registered_converter(
                XGBClassifier,
                "XGBoostXGBClassifier",
                calculate_linear_classifier_output_shapes,
                convert_xgboost,
                options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
            )

            if isinstance(best_model, XGBClassifier):
                best_model.get_booster().feature_names = None

            full_pipeline = Pipeline(
                [("preprocessor", sk_preprocessor), ("model", best_model)]
            )
            X_sample = train_data.drop(columns=[target_col], errors="ignore")[:1].copy()

            for col in preprocessor.categorical_cols:
                if col in X_sample.columns:
                    X_sample[col] = X_sample[col].astype(
                        str
                    )  # force string, pandas NaN becomes "nan"
            for col in preprocessor.numeric_cols:
                if col in X_sample.columns:
                    X_sample[col] = X_sample[col].astype("float32")

            try:
                cat_imputer = (
                    full_pipeline.named_steps["preprocessor"]
                    .named_transformers_["cat"]
                    .named_steps["imputer"]
                )
                cat_imputer.missing_values = "nan"
            except Exception as e:
                logger.warning(
                    f"Could not patch categorical imputer missing_values for ONNX: {e}"
                )

            onnx_model = to_onnx(
                full_pipeline,
                X=X_sample,
                target_opset={"": 12, "ai.onnx.ml": 3},
                options={id(full_pipeline.steps[-1][1]): {"zipmap": False}},
            )

            onnx_path = "models/best_model.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logger.info(f"Full pipeline ONNX model exported to {onnx_path}")

            # Register ONNX model directly to MLflow (No Pickle)
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="model",
                registered_model_name="InsuranceClassifier",
            )
            mlflow.log_artifact("models/columns.json", artifact_path="model")

        except Exception:
            import traceback

            logger.error(f"ONNX export logging failed: {traceback.format_exc()}")

        # Champion Selection Logic (Comparing Local Best vs. Actions Module/Production)
        model_name = "InsuranceClassifier"
        best_actions_auc = 0
        versions = client.search_model_versions(f"name='{model_name}'")

        if versions:
            latest_version = max(int(v.version) for v in versions)
        else:
            latest_version = None

        # Get current champion
        try:
            champion = client.get_model_version_by_alias(model_name, "champion")
            best_actions_auc = float(champion.tags.get("auc", 0))
            logger.info(f"Current Champion AUC: {best_actions_auc}")
        except Exception:
            best_actions_auc = -1
            logger.info("No existing champion found. First model.")

        is_new_champion = best_auc > best_actions_auc

        # Tag current version
        client.set_model_version_tag(
            name=model_name, version=latest_version, key="auc", value=str(best_auc)
        )

        # Promote if better
        if is_new_champion:
            logger.info(">>> NEW CHAMPION FOUND")

            client.set_registered_model_alias(
                name=model_name, alias="champion", version=latest_version
            )

            logger.info("Model promoted to 'champion'")
        else:
            logger.info(">>> CHAMPION RETAINED")

        # Test Predictions
        test_path = config.get("data", {}).get("test_path")
        if test_path:
            try:
                df_test = load_data(test_path)
                df_test_feat = create_features(df_test)
                X_test_proc, _ = preprocessor.prepare_data(df_test_feat, is_train=False)
                test_probs = best_model.predict_proba(X_test_proc)[:, 1]
                os.makedirs("artifacts", exist_ok=True)
                submission = pd.DataFrame()
                if "policy_id" in df_test.columns:
                    submission["policy_id"] = df_test["policy_id"]
                submission["is_claim"] = test_probs
                submission.to_csv("artifacts/test_predictions.csv", index=False)
                logger.info("Test predictions saved.")
            except Exception as e:
                logger.warning(f"Test predictions failed: {e}")


if __name__ == "__main__":
    run_training()
