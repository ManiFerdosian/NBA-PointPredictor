"""
Data pipeline to load NBA player game data from CSV into SQLite database.
"""
import os
import sqlite3
import pandas as pd
from pathlib import Path


def load_nba_data(csv_path: str, db_path: str):
    """
    Load NBA player game data from CSV and insert into SQLite database.
    
    Args:
        csv_path: Path to the CSV file
        db_path: Path to the SQLite database file
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Read CSV
    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean data: convert numeric columns and drop rows with missing values
    numeric_columns = [
        'minutes', 'points', 'rebounds', 'assists',
        'field_goals_attempted', 'field_goals_made',
        'three_pa', 'three_pm',
        'free_throws_attempted', 'free_throws_made'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing values.")
    
    # Create label column: over_20 (1 if points > 30, else 0)
    # Note: Column name kept as 'over_20' for compatibility, but threshold is now 30
    TARGET_LINE = 30
    df['over_20'] = (df['points'] > TARGET_LINE).astype(int)
    print(f"Labeling threshold: {TARGET_LINE} points")
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    print(f"Creating table player_games in {db_path}...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team TEXT NOT NULL,
            opponent TEXT NOT NULL,
            minutes REAL,
            points INTEGER,
            rebounds REAL,
            assists REAL,
            field_goals_attempted INTEGER,
            field_goals_made INTEGER,
            three_pa INTEGER,
            three_pm INTEGER,
            free_throws_attempted INTEGER,
            free_throws_made INTEGER,
            over_20 INTEGER
        )
    """)
    
    # Clear existing data (for idempotency)
    cursor.execute("DELETE FROM player_games")
    
    # Insert data
    print(f"Inserting {len(df)} rows into database...")
    df.to_sql('player_games', conn, if_exists='append', index=False)
    
    # Verify insertion
    cursor.execute("SELECT COUNT(*) FROM player_games")
    count = cursor.fetchone()[0]
    print(f"Successfully inserted {count} rows into player_games table.")
    
    conn.commit()
    conn.close()
    print("Data pipeline completed successfully.")


if __name__ == "__main__":
    # Get paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "assets" / "nbaplayers_500games.csv"
    db_path = project_root / "data" / "db.nba.sqlite"
    
    load_nba_data(str(csv_path), str(db_path))

