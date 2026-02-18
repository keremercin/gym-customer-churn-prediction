.PHONY: install gen-data train card run-api test lint

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

gen-data:
	python scripts/generate_sample_data.py

train:
	python scripts/train_model.py

card:
	python scripts/make_model_card.py

run-api:
	uvicorn gym_churn.api.main:app --reload --port 8700

test:
	pytest

lint:
	ruff check src tests scripts
