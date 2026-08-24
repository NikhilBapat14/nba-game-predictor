import os
import sqlite3
import joblib
import pandas as pd


def load_trained_model(model_path="models/logreg_model.joblib"):
    """Loads the serialized scikit-learn pipeline artifact."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please run train.py first.")
    return joblib.load(model_path)


def get_current_team_elos(conn):
    """Loads current live team Elo ratings from the team_states table into a dictionary."""
    try:
        df_states = pd.read_sql("SELECT teamName, elo FROM team_states;", conn)
        return dict(zip(df_states['teamName'], df_states['elo']))
    except Exception:
        print("Warning: Could not read 'team_states' table. Defaulting team Elos to 1500.")
        return {}


def predict_matchup(model, home_team, away_team, home_elo, away_elo):
    """Core prediction helper for a single matchup given team names and Elos."""
    features = pd.DataFrame([[
        home_elo,
        0,  # is_B2B_home placeholder
        away_elo,
        0,  # is_B2B_away placeholder
        home_elo - away_elo
    ]], columns=["pre_game_elo_home", "is_B2B_home", "pre_game_elo_away", "is_B2B_away", "pre_game_elo_diff"])

    win_prob = model.predict_proba(features)[0][1]
    predicted_class = model.predict(features)[0]
    predicted_winner = home_team if predicted_class == 1 else away_team

    return {
        "teamName_home": home_team,
        "teamName_away": away_team,
        "pre_game_elo_home": round(home_elo, 1),
        "pre_game_elo_away": round(away_elo, 1),
        "home_win_probability": round(win_prob * 100, 2),
        "predicted_winner": predicted_winner
    }


def predict_from_schedule(conn, model, elo_container):
    """Option 1: Finds the next upcoming game for all 30 teams (15 games total),
    generates predictions, and sorts them by model confidence."""
    print("\n--- Generating Predictions for Each Team's Next Game ---")

    try:
        query = "SELECT * FROM league_schedule;"
        df_schedule = pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error querying 'league_schedule' table: {e}. Make sure ingestion has run.")
        return

    if df_schedule.empty:
        print("The 'league_schedule' table is empty.")
        return

    df_schedule["gameDateTimeEst"] = pd.to_datetime(df_schedule["gameDateTimeEst"])
    
    current_time = pd.Timestamp.now()
    df_future = df_schedule[df_schedule["gameDateTimeEst"] >= current_time].sort_values("gameDateTimeEst")

    if df_future.empty:
        df_future = df_schedule.sort_values("gameDateTimeEst")

    teams_processed = set()
    next_game_rows = []

    for _, row in df_future.iterrows():
        home = row["homeTeamName"]
        away = row["awayTeamName"]

        if home not in teams_processed or away not in teams_processed:
            if not any(g["gameId"] == row["gameId"] for g in next_game_rows):
                next_game_rows.append(row)
                teams_processed.add(home)
                teams_processed.add(away)

        if len(teams_processed) >= 30:
            break

    if not next_game_rows:
        print("Could not isolate upcoming games from the schedule.")
        return

    prediction_records = []

    for row in next_game_rows:
        game_id = row["gameId"]
        game_date = row["gameDateTimeEst"]
        home_name = row["homeTeamName"]
        away_name = row["awayTeamName"]

        home_elo = elo_container.get(home_name, 1500.0)
        away_elo = elo_container.get(away_name, 1500.0)

        result = predict_matchup(model, home_name, away_name, home_elo, away_elo)
        result["gameId"] = game_id
        result["game_date"] = str(game_date)

        win_prob = result["home_win_probability"]
        if result["predicted_winner"] == home_name:
            result["confidence"] = win_prob
        else:
            result["confidence"] = 100.0 - win_prob

        prediction_records.append(result)

    df_predictions = pd.DataFrame(prediction_records)
    df_predictions = df_predictions.sort_values(by="confidence", ascending=False).reset_index(drop=True)

    df_display = df_predictions.copy()
    df_display["Matchup"] = df_display["teamName_away"] + " @ " + df_display["teamName_home"]
    df_display["Home Elo"] = df_display["pre_game_elo_home"].map(lambda x: f"{x:.1f}")
    df_display["Away Elo"] = df_display["pre_game_elo_away"].map(lambda x: f"{x:.1f}")
    df_display["Home Win Prob"] = df_display["home_win_probability"].map(lambda x: f"{x:.2f}%")
    df_display["Predicted Winner"] = df_display["predicted_winner"]

    df_predictions.to_sql("game_predictions", conn, if_exists="replace", index=False)

    print(f"\nSuccessfully predicted each team's next game ({len(df_predictions)} total matchups), sorted by model confidence:\n")
    print(df_display[[
        "game_date", "Matchup", "Home Elo", "Away Elo", "Home Win Prob", "Predicted Winner"
    ]].to_string(index=False))

def predict_custom_matchup(conn, model, elo_container, home_team=None, away_team=None):
    """Option 2: Predicts a single custom matchup based on user-entered team names."""
    print("\n--- Custom Matchup Prediction ---")
    
    elo_lookup = {k.lower(): (k, v) for k, v in elo_container.items()}

    if not home_team:
        home_team = input("Enter Home Team Name (e.g. Lakers, Celtics, Bucks): ").strip()
    if not away_team:
        away_team = input("Enter Away Team Name (e.g. Warriors, Heat, Suns): ").strip()

    home_key = home_team.lower()
    away_key = away_team.lower()

    if home_key in elo_lookup:
        official_home_name, home_elo = elo_lookup[home_key]
    else:
        official_home_name, home_elo = home_team, 1500.0
        print(f"Note: '{home_team}' not found in team_states. Using default Elo of 1500.0.")

    if away_key in elo_lookup:
        official_away_name, away_elo = elo_lookup[away_key]
    else:
        official_away_name, away_elo = away_team, 1500.0
        print(f"Note: '{away_team}' not found in team_states. Using default Elo of 1500.0.")

    result = predict_matchup(model, official_home_name, official_away_name, home_elo, away_elo)

    print("\n================ PREDICTION RESULT ================")
    print(f" Matchup           : {official_away_name} @ {official_home_name}")
    print(f" Home Elo          : {result['pre_game_elo_home']}")
    print(f" Away Elo          : {result['pre_game_elo_away']}")
    print(f" Home Win Prob     : {result['home_win_probability']}%")
    print(f" Predicted Winner  : {result['predicted_winner']}")
    print("===================================================\n")


def main_menu(db_path="./data/nba_pipeline.db", model_path="models/logreg_model.joblib"):
    """CLI Menu to select prediction option."""
    model = load_trained_model(model_path)
    conn = sqlite3.connect(db_path)
    elo_container = get_current_team_elos(conn)

    print("\nSelect Prediction Mode:")
    print("1. Predict upcoming games from League Schedule")
    print("2. Enter custom team matchup manually")
    
    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        predict_from_schedule(conn, model, elo_container)
    elif choice == "2":
        predict_custom_matchup(conn, model, elo_container)
    else:
        print("Invalid option selected.")

    conn.close()


if __name__ == "__main__":
    main_menu()