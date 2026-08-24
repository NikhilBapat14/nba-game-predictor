import os
import sqlite3
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

def database_connect(db_path="./data/nba_pipeline.db"):
    """Ensures the local data directory and raw_games table exist, then returns a connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    raw_games_table_query = """
    CREATE TABLE IF NOT EXISTS raw_games (
        gameId INTEGER,
        gameDateTimeEst TEXT,
        teamCity TEXT,
        teamName TEXT,
        teamId INTEGER,
        opponentTeamName TEXT,
        opponentTeamId INTEGER,
        home INTEGER,
        win INTEGER,
        teamScore INTEGER,
        opponentScore INTEGER,
        fieldGoalsAttempted INTEGER,
        fieldGoalsMade INTEGER,
        threePointersAttempted INTEGER,
        threePointersMade INTEGER,
        freeThrowsAttempted INTEGER,
        reboundsDefensive INTEGER,
        reboundsOffensive INTEGER,
        turnovers INTEGER,
        gameType TEXT
    );
    """
    cursor.execute(raw_games_table_query)
    connection.commit()
    return connection

def get_existing_game_ids(connection):
    """Queries the SQLite database for existing gameIds and returns them as a set for fast lookup."""
    query = "SELECT DISTINCT gameId FROM raw_games;"
    cursor = connection.cursor()
    cursor.execute(query)
    existing_ids = {row[0] for row in cursor.fetchall()}
    return existing_ids

def fetch_kaggle_data():
    """Downloads and loads the target CSV file from Kaggle into a Pandas DataFrame."""
    dataset_handle = "eoinamoore/historical-nba-data-and-player-box-scores"
    file_path = "TeamStatistics.csv"

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_handle,
        file_path
    )

    selected_columns = [
        "gameId",
        "gameDateTimeEst",
        "teamCity",
        "teamName",
        "teamId",
        "opponentTeamName",
        "opponentTeamId",
        "home",
        "win",
        "teamScore",
        "opponentScore",
        "fieldGoalsAttempted",
        "fieldGoalsMade",
        "threePointersAttempted",
        "threePointersMade",
        "freeThrowsAttempted",
        "reboundsDefensive",
        "reboundsOffensive",
        "turnovers",
        "gameType",
    ]

    df = df[selected_columns]

    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    df = df.loc[df["gameDateTimeEst"] >= "2012-07-30 20:00:00"]
    df = df[(df.gameType == "Playoffs") | (df.gameType == "Regular Season")]
    
    conditions = [
        (df['teamCity'] == 'New Orleans') & (df['teamName'] == 'Hornets'),
        (df['teamCity'] == 'Charlotte') & (df['teamName'] == 'Bobcats'),
    ]

    new_names = ['Pelicans', 'Hornets']
    df['teamName'] = np.select(conditions, new_names, default=df['teamName'])

    return df

def fetch_and_save_schedule(db_path="./data/nba_pipeline.db"):
    """Downloads the upcoming league schedule from Kaggle and saves it to SQLite."""
    dataset_handle = "eoinamoore/historical-nba-data-and-player-box-scores"
    file_path = "LeagueSchedule25_26.csv"

    print("Downloading league schedule from Kaggle...")
    df_schedule = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_handle,
        file_path
    )

    selected_columns = [
        "gameId",
        "gameDateTimeEst",
        "homeTeamName",
        "homeTeamId",
        "awayTeamName",
        "awayTeamId",
        "gameLabel"
    ]

    df_schedule = df_schedule[selected_columns]

    important_labels = [
    "East First Round",
    "West First Round",
    "East Conf. Semifinals",
    "West Conf. Semifinals",
    "East Conf. Finals",
    "West Conf. Finals",
    "Finals",
    "SoFi Play-In Tournament"
    ]

    mask = df_schedule["gameLabel"].isna() | df_schedule["gameLabel"].isin(important_labels)
    df_schedule = df_schedule[mask].copy()
    conn = sqlite3.connect(db_path)
    df_schedule.to_sql("league_schedule", conn, if_exists="replace", index=False)
    conn.close()
    print("League schedule successfully ingested into 'league_schedule' table.")

def filter_new_games(df, existing_ids):
    """Filters the incoming DataFrame to exclude gameIds already present in the database."""
    if existing_ids:
        df_new = df[~df['gameId'].isin(existing_ids)]
    else:
        df_new = df.copy()
        
    return df_new

def save_new_games(df_filtered, connection):
    """Appends filtered new records directly to the SQLite table using pandas to_sql."""
    if df_filtered.empty:
        print("No new games to ingest.")
        return 0
    
    df_filtered.to_sql('raw_games', connection, if_exists='append', index=False)
    print(f"Successfully ingested {len(df_filtered)} new games into the database.")
    return len(df_filtered)

def run_pipeline():
    """Coordinates the full data ingestion workflow."""
    print("Connecting to database...")
    conn = database_connect()
    
    try:
        print("Fetching existing game IDs from storage...")
        existing_ids = get_existing_game_ids(conn)
        
        print("Downloading dataset from Kaggle...")
        df_raw = fetch_kaggle_data()

        print("Downloading schedule from Kaggle...")
        fetch_and_save_schedule()
        
        print("Filtering for new, unseen games...")
        df_new = filter_new_games(df_raw, existing_ids)
        
        print("Saving new records...")
        save_new_games(df_new, conn)
        
    finally:
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    run_pipeline()