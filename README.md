# hoops-edge-predictor

College basketball game predictions using a PyTorch MLP model. Reads from the `hoops-edge` S3 lakehouse (produced by cbbd_etl) to generate spread predictions and win probabilities.

## Setup

```bash
poetry install
```

Requires AWS credentials configured with read access to `s3://hoops-edge/`.

## Usage

### 1. Build Features

```bash
python -m src.cli build-features --season 2026
```

### 2. Train Models

```bash
python -m src.cli train --seasons 2015-2025
```

### 3. Hyperparameter Tuning

```bash
python -m src.cli tune --seasons 2015-2025 --trials 50
```

### 4. Predict Today's Games

```bash
python -m src.cli predict-today --season 2026
```

### 5. Predict Full Season

```bash
python -m src.cli predict-season --season 2026
```

### 6. Validate Features

```bash
python -m src.cli validate-features --season 2025
```

## Model

- **MLPRegressor**: Predicts home spread distribution (mu, sigma) via Gaussian NLL loss
- **MLPClassifier**: Predicts home win probability via BCE loss
- Both use 37-feature input vectors combining adjusted efficiency ratings and rolling four-factor averages

## Testing

```bash
make test
```
