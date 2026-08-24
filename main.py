import os
import sys
import argparse
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from data_ingestion import run_pipeline as run_data_ingestion
except ImportError:
    run_data_ingestion = None

try:
    from feature_engine import process_historical_features
except ImportError:
    process_historical_features = None

try:
    from train import train_and_evaluate_model
except ImportError:
    try:
        from src.train import train_and_evaluate_model
    except ImportError:
        train_and_evaluate_model = None

try:
    from predictor import (
        load_trained_model, 
        get_current_team_elos, 
        predict_from_schedule as predict_next_games_for_all_teams,
        predict_custom_matchup,
        main_menu as run_prediction_menu
    )
except ImportError:
    load_trained_model = None
    get_current_team_elos = None
    predict_next_games_for_all_teams = None
    predict_custom_matchup = None
    run_prediction_menu = None


def run_full_pipeline():
    """Executes the complete end-to-end pipeline sequentially."""

    print("\n[Phase 1/4] Running Data Ingestion...")
    try:
        if run_data_ingestion:
            run_data_ingestion()
            print("-> Data ingestion completed successfully.")
        else:
            print("Error: Data ingestion module could not be loaded.")
            return
    except Exception as e:
        print(f"Error during Data Ingestion: {e}")
        return

    print("\n[Phase 2/4] Running Feature Engineering & Elo Updates...")
    try:
        if process_historical_features:
            process_historical_features()
            print("-> Feature engineering and Elo states updated successfully.")
        else:
            print("Error: process_historical_features() not found.")
            return
    except Exception as e:
        print(f"Error during Feature Engineering: {e}")
        return

    print("\n[Phase 3/4] Running Model Training...")
    try:
        if train_and_evaluate_model:
            train_and_evaluate_model()
            print("-> Model training completed and artifact saved.")
        else:
            print("Warning: train_and_evaluate_model() not found. Skipping retraining.")
    except Exception as e:
        print(f"Error during Model Training: {e}")
        return

    print("\n[Phase 4/4] Running Inference & Predictions...")
    try:
        if all([load_trained_model, get_current_team_elos, predict_next_games_for_all_teams]):
            conn = sqlite3.connect("./data/nba_pipeline.db")
            model = load_trained_model()
            elo_container = get_current_team_elos(conn)
            
            predict_next_games_for_all_teams(conn, model, elo_container)
            conn.close()
            print("-> Inference and confidence-sorted predictions completed successfully.")
        else:
            print("Error: Predictor modules could not be loaded.")
            return
    except Exception as e:
        print(f"Error during Inference: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Machine Learning Pipeline Orchestrator")
    parser.add_argument("--full", action="store_true", help="Run the full end-to-end pipeline")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion only")
    parser.add_argument("--features", action="store_true", help="Run feature engineering & Elo updates only")
    parser.add_argument("--train", action="store_true", help="Run model training only")
    parser.add_argument("--predict", action="store_true", help="Run inference and confidence-sorted predictions from schedule")
    parser.add_argument("--custom", action="store_true", help="Predict a custom team matchup")
    parser.add_argument("--home", type=str, help="Home team name for custom prediction")
    parser.add_argument("--away", type=str, help="Away team name for custom prediction")
    parser.add_argument("--menu", action="store_true", help="Open interactive prediction selection menu")

    args = parser.parse_args()

    if args.ingest:
        print("Running Data Ingestion...")
        if run_data_ingestion: run_data_ingestion()
    elif args.features:
        print("Running Feature Engineering...")
        if process_historical_features: process_historical_features()
    elif args.train:
        print("Running Model Training...")
        if train_and_evaluate_model: train_and_evaluate_model()
    elif args.custom:
        print("Running Custom Matchup Prediction...")
        if all([load_trained_model, get_current_team_elos, predict_custom_matchup]):
            conn = sqlite3.connect("./data/nba_pipeline.db")
            model = load_trained_model()
            elos = get_current_team_elos(conn)
            predict_custom_matchup(conn, model, elos, home_team=args.home, away_team=args.away)
            conn.close()
    elif args.menu:
        if run_prediction_menu:
            run_prediction_menu()
    elif args.predict:
        print("Running Inference & Predictions...")
        if all([load_trained_model, get_current_team_elos, predict_next_games_for_all_teams]):
            conn = sqlite3.connect("./data/nba_pipeline.db")
            model = load_trained_model()
            elos = get_current_team_elos(conn)
            predict_next_games_for_all_teams(conn, model, elos)
            conn.close()
    else:
        run_full_pipeline()