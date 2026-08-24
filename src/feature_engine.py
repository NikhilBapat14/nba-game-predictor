import os
import sqlite3
import numpy as np
import pandas as pd


def current_team_state(db_path="./data/nba_pipeline.db"):
    """Ensures the local data directory and team_states table exist, then returns a connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    updated_elo_query = """
    CREATE TABLE IF NOT EXISTS team_states (
        teamName TEXT PRIMARY KEY,
        elo REAL,
        gameDateTimeEst TEXT
    );
    """
    cursor.execute(updated_elo_query)
    connection.commit()
    return connection


def historical_games(db_path="./data/nba_pipeline.db"):
    """Ensures the local data directory and games_historical table exist, then returns a connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    historical_games_query = """
    CREATE TABLE IF NOT EXISTS games_historical (
        gameId INTEGER,
        game_date TEXT,
        teamName_home TEXT,
        teamName_away TEXT,
        pre_game_elo_home REAL,
        is_B2B_home INTEGER,
        pre_game_elo_away REAL,
        is_B2B_away INTEGER,
        pre_game_elo_diff REAL,
        win_home INTEGER
    );
    """
    cursor.execute(historical_games_query)
    connection.commit()
    return connection


def load_default_elo(path="./data/starting_elo.csv"):
    """Loads starting Elo ratings into a dictionary, falling back to 1500 if file doesn't exist."""
    if os.path.exists(path):
        starting_elo_df = pd.read_csv(path)

        if {"teamName", "elo"}.issubset(starting_elo_df.columns):
            return dict(zip(starting_elo_df["teamName"], starting_elo_df["elo"]))

        if not starting_elo_df.empty:
            first_row = starting_elo_df.iloc[0]
            elo_map = pd.to_numeric(first_row, errors="coerce").dropna().to_dict()
            if elo_map:
                return elo_map

        print(f"Warning: Could not parse starting Elo file at {path}. Falling back to defaults.")
    return {}


def season_column(df):
    """Adds a season column based on game month/year."""
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    df["season"] = np.where(df["gameDateTimeEst"].dt.month >= 10, df["gameDateTimeEst"].dt.year + 1, df["gameDateTimeEst"].dt.year)
    return df


def calculate_new_elo(old_elo, m, s, e, k=20):
    """Calculates updated Elo rating."""
    new_elo = old_elo + k * m * (s - e)
    return new_elo


def margin_of_victory(mov, elo_diff):
    """Calculates margin of victory multiplier."""
    m = ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)
    return m


def off_season_regression(elo_container):
    """Regresses team Elo ratings toward the mean (1505) during the off-season."""
    for team_name, current_elo in elo_container.items():
        regressed_elo = (current_elo * 0.75) + (1505 * 0.25)
        elo_container[team_name] = regressed_elo
    return elo_container


def process_historical_features(db_path="./data/nba_pipeline.db"):
    """Path 1 Pipeline: Processes raw games chronologically, updates Elo states, and logs historical features."""
    conn = sqlite3.connect(db_path)
    
    df_raw = pd.read_sql("SELECT * FROM raw_games;", conn)
    if df_raw.empty:
        print("No raw games found in database.")
        conn.close()
        return

    df_raw = season_column(df_raw)
    df_raw = df_raw.sort_values(by=["gameDateTimeEst", "gameId", "home"])

    df_raw["game_date"] = pd.to_datetime(df_raw["gameDateTimeEst"]).dt.date
    df_raw["last_game_date"] = df_raw.groupby("teamName")["game_date"].shift(1)
    df_raw["days_rest"] = (pd.to_datetime(df_raw["game_date"]) - pd.to_datetime(df_raw["last_game_date"])).dt.days
    df_raw["is_B2B"] = np.where(df_raw["days_rest"] == 1, 1, 0)

    elo_container = load_default_elo()

    historical_records = []
    current_season = df_raw["season"].iloc[0]

    for game_id, team_df in df_raw.groupby("gameId", sort=False):
        game_season = team_df["season"].iloc[0]
        if current_season != game_season:
            elo_container = off_season_regression(elo_container)
            current_season = game_season

        if len(team_df) != 2:
            continue

        home_row = team_df[team_df["home"] == 1].iloc[0]
        away_row = team_df[team_df["home"] == 0].iloc[0]

        home_name = home_row["teamName"]
        away_name = away_row["teamName"]

        home_pre = elo_container.get(home_name, 1500.0)
        away_pre = elo_container.get(away_name, 1500.0)

        d = (home_pre + 100) - away_pre
        expected_home = 1 / (1 + 10 ** (-d / 400))
        expected_away = 1 - expected_home

        if home_row["win"] == 1:
            mov = home_row["teamScore"] - home_row["opponentScore"]
            elo_diff = home_pre - away_pre
            home_s, away_s = 1, 0
        else:
            mov = away_row["teamScore"] - away_row["opponentScore"]
            elo_diff = away_pre - home_pre
            home_s, away_s = 0, 1

        m = margin_of_victory(mov, elo_diff)

        home_post = calculate_new_elo(home_pre, m, home_s, expected_home)
        away_post = calculate_new_elo(away_pre, m, away_s, expected_away)

        elo_container[home_name] = home_post
        elo_container[away_name] = away_post

        historical_records.append({
            "gameId": game_id,
            "game_date": str(home_row["gameDateTimeEst"]),
            "teamName_home": home_name,
            "teamName_away": away_name,
            "pre_game_elo_home": home_pre,
            "is_B2B_home": int(home_row["is_B2B"]),
            "pre_game_elo_away": away_pre,
            "is_B2B_away": int(away_row["is_B2B"]),
            "pre_game_elo_diff": home_pre - away_pre,
            "win_home": int(home_row["win"])
        })

    df_historical = pd.DataFrame(historical_records)
    df_historical.to_sql("games_historical", conn, if_exists="replace", index=False)

    df_states_out = pd.DataFrame(list(elo_container.items()), columns=["teamName", "elo"])
    df_states_out["gameDateTimeEst"] = str(pd.Timestamp.now())
    df_states_out.to_sql("team_states", conn, if_exists="replace", index=False)

    conn.close()
    print(f"Successfully processed and saved {len(df_historical)} historical games and team states.")


if __name__ == "__main__":
    current_team_state()
    historical_games()
    process_historical_features()