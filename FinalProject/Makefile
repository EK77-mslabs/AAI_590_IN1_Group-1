.PHONY: install format lint test train serve docker-train docker-serve clean

# Standard installations
install:
	uv sync --all-extras
	uv run pre-commit install

format:
	uv run black src tests
	uv run isort src tests

lint:
	uv run flake8 src tests
	uv run mypy src tests

test:
	uv run pytest tests/

train:
	uv run python src/ml_system/pipelines/train.py

serve:
	uv run uvicorn ml_system.api.main:app --host 0.0.0.0 --port 8000

docker-train:
	docker compose -f docker/docker-compose.yaml run --rm --build train

docker-serve:
	docker compose -f docker/docker-compose.yaml up --build inference

clean:
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf build
	rm -rf dist
