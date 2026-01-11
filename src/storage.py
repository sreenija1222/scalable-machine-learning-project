from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "user_data.sqlite"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS daily_entries (
  entry_date TEXT PRIMARY KEY,
  phase TEXT NOT NULL,
  is_weekend INTEGER NOT NULL,
  sleep_duration_minutes INTEGER NOT NULL,
  resting_heart_rate__value INTEGER NOT NULL,
  cramps_num INTEGER NOT NULL,
  headaches_num INTEGER NOT NULL,
  sleepissue_num INTEGER NOT NULL,
  stress_num INTEGER NOT NULL,
  lag1_mood INTEGER,
  lag1_energy INTEGER,
  gt_mood INTEGER,
  gt_energy INTEGER,
  mood_pred INTEGER,
  energy_pred INTEGER,
  route TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(CREATE_SQL)
    conn.commit()
    return conn

def upsert_entry(row: Dict[str, Any]) -> None:
    """
    insert or rpelace an entry by date, row keys match table col
    """
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO daily_entries ({','.join(cols)}) VALUES ({placeholders}) " \
          f"ON CONFLICT(entry_date) DO UPDATE SET " + ",".join([f"{c}=excluded.{c}" for c in cols if c != "entry_date"])

    with _connect() as conn:
        conn.execute(sql, [row[c] for c in cols])
        conn.commit()

def fetch_entries(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    return enteries bw start date and end date (inclusive), date must be YYYY-MM-DD format
    """
    where = []
    params: List[Any] = []
    if start_date:
        where.append("entry_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("entry_date <= ?")
        params.append(end_date)

    sql = "SELECT * FROM daily_entries"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY entry_date ASC"

    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if not df.empty:
        df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df

def fetch_entry_by_date(entry_date: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM daily_entries WHERE entry_date = ?", (entry_date,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

def delete_entry(entry_date: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM daily_entries WHERE entry_date = ?", (entry_date,))
        conn.commit()
