# Production Grade ML System for Insurance Claims

This project creates a production-grade Machine Learning system, complete with CI/CD, Pytest integration, Dockerization, and a robust FastAPI inference service powered by **Astral `uv`**.

## Project Structure

- `src/data/`: Data loading and preprocessing pipelines.
- `src/features/`: Complex feature engineering (handled automatically at inference).
- `src/models/`: Model definitions (Scikit-learn RandomForest & PyTorch FFN).
- `src/pipelines/`: The `train.py` script for end-to-end training and serialization.
- `src/evaluation/`: Performance gate logic evaluating AUC against baselines.
- `src/api/`: Ultra-fast FastAPI real-time model serving.
- `tests/`: End-to-end unit and behavioral testing running via Pytest.
- `config/`: Centralized YAML configurations avoiding hardcoding.

## Setup Instructions

This core project uses the lightning-fast `uv` package manager natively instead of classic `pip`, stripping out massive GB-heavy GPU dependencies by locking down onto `pytorch-cpu`.

1. **Install virtual environment and all dev dependencies:**
   ```bash
   make install
   ```

## Development & Training

1. **Run Full Pytest Suite:**
   ```bash
   make test
   ```
   *Guarantees zero regressions in data engineering pipelines and ML structures.*

2. **Train the Models Natively:**
   ```bash
   make train
   ```
   *Executes `src/pipelines/train.py`, processes datasets, passes the AUC performance gate, and serializes artifacts (`.pkl` and `.bin`) into the local `models/` directory instantly.*

## Native Docker Deployment

We containerize the system to emulate a perfect production backend. `localstack` and massive AWS dependencies have been aggressively stripped.

1. **Spin up the Backend Service natively inside Docker:**
   ```bash
   docker compose up --build
   ```

2. **Access the Application UI:**
   Navigate straight to **[http://localhost:8000/docs](http://localhost:8000/docs)** in your browser!

   This auto-generated Swagger UI gives you full interactive capability to test the `POST /predict` API, automatically mapping complex binary outputs to human-readable statuses (e.g., `Claim` or `No Claim`).

## Linting and Formatting

Maintain perfect code quality via:
```bash
make format
make lint
```
