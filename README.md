# NBA Betting Classification Model

End-to-end NBA game prediction pipeline that ingests historical game data, engineers Elo-based features, trains a binary classifier, and generates matchup predictions from a schedule.

## Overview

This project is structured as a reproducible ML pipeline with four phases:

1. Data ingestion from Kaggle into SQLite.
2. Feature engineering with chronological Elo updates.
3. Model training using time-series splits.
4. Prediction from upcoming schedule or custom matchups.

The main orchestration entrypoint is main.py.

## Features

- Historical NBA data ingestion into a local SQLite database.
- Schedule ingestion and filtering for relevant NBA labels.
- Elo tracking with offseason regression.
- Chronological feature construction for supervised learning.
- Logistic Regression training pipeline with scaler.
- Prediction output for:
	- Each team's next game from schedule.
	- User-provided custom matchups.
- Optional notebook workflow for multi-model testing with MLflow.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- xgboost, lightgbm (notebook model comparison)
- mlflow (notebook experiment tracking)
- sqlite3 (local data store)
- kagglehub (dataset access)

## Project Structure

```text
betting_classification_model/
├─ data/
│  ├─ nba_pipeline.db
│  ├─ TeamStatistics.csv
│  ├─ final_dataset.csv
│  └─ ...
├─ models/
│  └─ logreg_model.joblib
├─ notebooks/
│  └─ model_testing.ipynb
├─ src/
│  ├─ data_ingestion.py
│  ├─ feature_engine.py
│  ├─ train.py
│  └─ predictor.py
├─ main.py
├─ requirements.txt
└─ README.md
```

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/NikhilBapat14/nba-game-predictor.git
cd betting_classification_model
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle credentials

This project downloads datasets via kagglehub. Make sure Kaggle authentication is available, for example using a kaggle.json file in your home Kaggle config directory.

## Usage

Run all phases end-to-end:

```bash
python main.py --full
```

Run each phase individually:

```bash
python main.py --ingest
python main.py --features
python main.py --train
python main.py --predict
```

Open interactive prediction menu:

```bash
python main.py --menu
```

Run a custom matchup prediction:

```bash
python main.py --custom --home Heat --away Sixers
```

## CLI Options

Available flags in main.py:

- --full: Run complete pipeline.
- --ingest: Run data ingestion only.
- --features: Run feature engineering only.
- --train: Run model training only.
- --predict: Run schedule-based predictions.
- --custom: Run custom matchup prediction.
- --home: Home team name (with --custom).
- --away: Away team name (with --custom).
- --menu: Open interactive prediction mode.

## Data Flow

1. src/data_ingestion.py
	 - Loads TeamStatistics and selected schedule data from Kaggle.
	 - Stores raw games in raw_games table.
	 - Stores filtered schedule in league_schedule table.

2. src/feature_engine.py
	 - Computes season and rest-based features.
	 - Updates Elo per game chronologically.
	 - Writes games_historical and team_states tables.

3. src/train.py
	 - Loads engineered data from games_historical.
	 - Trains StandardScaler + LogisticRegression pipeline.
	 - Saves model artifact to models/logreg_model.joblib.

4. src/predictor.py
	 - Loads current Elo state and trained model.
	 - Predicts upcoming schedule games.
	 - Supports custom matchup predictions.

## Outputs

- SQLite database: data/nba_pipeline.db
	- raw_games
	- league_schedule
	- games_historical
	- team_states
	- game_predictions

- Trained model artifact:
	- models/logreg_model.joblib

## Notebook Experiments

notebooks/model_testing.ipynb includes multi-model comparisons for LR, RF, XGB, and LGBM, with metric logging to MLflow (accuracy and log_loss).

If using MLflow UI:

```bash
mlflow ui --backend-store-uri "sqlite:////absolute/path/to/mlflow.db" --port 5001
```

## Troubleshooting

- Model loading error related to scikit-learn version mismatch:
	- Retrain model in the current environment with python src/train.py.

- Kaggle download issues:
	- Verify Kaggle credentials are configured and valid.

- Empty predictions:
	- Ensure ingestion and feature engineering completed successfully before prediction.

## Reproducibility Notes

- Use the same Python version across machines when possible.
- Install dependencies only from requirements.txt in a fresh virtual environment.
- Re-run pipeline in order after pulling fresh changes:
	1. python main.py --ingest
	2. python main.py --features
	3. python main.py --train
	4. python main.py --predict