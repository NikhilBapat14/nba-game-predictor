import os
import sqlite3
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def load_historical_data(db_path="./data/nba_pipeline.db"):
    """Ensures the local data directory exists and loads historical game features from SQLite."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df_historical_games = pd.read_sql("SELECT * FROM games_historical;", conn)
    conn.close()
    return df_historical_games


def prepare_features_and_target(df, target="win_home"):
    """Drops missing values and separates feature matrix X from target vector y,
    excluding non-numeric metadata columns."""
    df = df.dropna()
    
    cols_to_drop = [target, "gameId", "game_date", "teamName_home", "teamName_away"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df[target]
    return X, y


def get_train_test_split(X, y):
    """Uses TimeSeriesSplit to extract the final chronological train and test sets."""
    tss = TimeSeriesSplit(n_splits=5)
    
    X_train, X_test, y_train, y_test = None, None, None, None
    for train_index, test_index in tss.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model():
    """Orchestrates data loading, feature preprocessing, model training, evaluation, and saving."""
    print("Loading historical data from database...")
    df = load_historical_data()
    
    if df.empty:
        print("No data found in games_historical table. Run feature engineering first.")
        return

    print("Preparing feature matrix and target vector...")
    X, y = prepare_features_and_target(df)
    
    print("Splitting data chronologically...")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    print("Building Scikit-Learn Pipeline (StandardScaler + LogisticRegression)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=.01))
    ])

    print("Fitting model on training data...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model performance on test fold...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Log-Loss : {loss:.4f}\n")

    os.makedirs("models", exist_ok=True)
    model_path = "models/logreg_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Pipeline artifact successfully saved to {model_path}")


if __name__ == "__main__":
    train_and_evaluate_model()