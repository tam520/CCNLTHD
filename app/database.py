import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = "lpr_history.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def save_detection(filename: str, plate_text: str, confidence: float):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO detections (filename, plate_text, confidence, created_at) VALUES (?, ?, ?, ?)",
            (filename, plate_text, confidence, datetime.now().isoformat())
        )
        conn.commit()

def get_history(limit: int = 50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]

def search_plate(plate_text: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM detections WHERE plate_text LIKE ? ORDER BY created_at DESC",
            (f"%{plate_text}%",)
        ).fetchall()
    return [dict(row) for row in rows]