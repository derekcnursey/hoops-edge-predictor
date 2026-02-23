.PHONY: install test lint build-features train tune predict-today predict-season validate

install:
	poetry install

test:
	poetry run pytest tests/ -v

lint:
	poetry run python -m py_compile src/config.py
	poetry run python -m py_compile src/s3_reader.py
	poetry run python -m py_compile src/four_factors.py
	poetry run python -m py_compile src/rolling_averages.py
	poetry run python -m py_compile src/features.py
	poetry run python -m py_compile src/dataset.py
	poetry run python -m py_compile src/architecture.py
	poetry run python -m py_compile src/trainer.py
	poetry run python -m py_compile src/infer.py
	poetry run python -m py_compile src/tuner.py
	poetry run python -m py_compile src/cli.py

build-features:
	poetry run python -m src.cli build-features --season $(SEASON)

train:
	poetry run python -m src.cli train --seasons $(SEASONS)

tune:
	poetry run python -m src.cli tune --seasons $(SEASONS) --trials $(or $(TRIALS),50)

predict-today:
	poetry run python -m src.cli predict-today --season $(SEASON)

predict-season:
	poetry run python -m src.cli predict-season --season $(SEASON)

validate:
	poetry run python -m src.cli validate-features --season $(SEASON)
