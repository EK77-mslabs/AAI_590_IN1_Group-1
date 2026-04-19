import json
import logging
from typing import Any, Dict, List

# import third-party libraries
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# import from src
from ml_system.data.loader import load_config

app = FastAPI(title="ML Insurance Claim Prediction API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
config = load_config()
onnx_session = None
sklearn_preprocessor = None
column_order = None


@app.on_event("startup")
def load_artifacts():
    global onnx_session, sklearn_preprocessor, column_order
    try:
        import onnxruntime as ort

        onnx_session = ort.InferenceSession("models/best_model.onnx")

        # Load the sklearn preprocessor (ColumnTransformer) for feature transformation
        # REMOVED: pre_state = joblib.load("models/preprocessor.pkl")
        # Preprocessing is now handled exclusively inside the ONNX pipeline.
        sklearn_preprocessor = None

        with open("models/columns.json", "r") as f:
            column_order = json.load(f)

        logger.info("ONNX full-pipeline model and column config loaded successfully.")
    except Exception as e:
        logger.warning(
            f"Could not load artifacts. \
            This is expected during CI or first run. Error: {e}"
        )


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Insurance Claim Prediction API is \
             active. Visit /docs for the interactive Swagger UI.",
    }


class PredictRequest(BaseModel):
    features: List[Dict[str, Any]]


@app.post("/predict")
def predict(request: PredictRequest):
    if onnx_session is None:
        raise HTTPException(status_code=503, detail="ONNX Model not loaded yet.")

    df = pd.DataFrame(request.features)
    try:
        from ml_system.features.engineering import create_features

        df_engineered = create_features(df)

        # Drop columns that were dropped during training
        drop_cols = config["data"].get("drop_cols", [])
        df_engineered = df_engineered.drop(columns=drop_cols, errors="ignore")

        # Build dictionary from dataframe dynamically for
        # ONNX Pipeline matching exactly `get_inputs()`
        onnx_inputs = {}
        for onnx_input in onnx_session.get_inputs():
            name = onnx_input.name

            # If a feature was somehow dropped, fill it with default
            # so ONNX does not break
            if name not in df_engineered.columns:
                df_engineered[name] = 0.0

            col_data = df_engineered[name].values.reshape(-1, 1)

            if onnx_input.type == "tensor(float)":
                col_data = col_data.astype(np.float32)
            elif onnx_input.type == "tensor(int64)":
                col_data = col_data.astype(np.int64)
            elif onnx_input.type == "tensor(string)":
                col_data = col_data.astype(str)
            else:
                col_data = col_data.astype(np.float32)  # fallback

            onnx_inputs[name] = col_data

        results = onnx_session.run(None, onnx_inputs)

        # results[0] = predicted labels, results[1] = probabilities
        predictions = results[0].flatten().tolist()
        probabilities = [float(p[1]) for p in results[1]] if len(results) > 1 else []

        # Map binary outcome to human-readable labels
        label_map = {0: "No Claim", 1: "Claim"}
        human_readable = [label_map.get(int(p), str(p)) for p in predictions]

        return {
            "predictions": human_readable,
            "raw_predictions": predictions,
            "probabilities": probabilities,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
