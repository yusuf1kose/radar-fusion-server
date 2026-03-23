import sqlite3
import json
from pathlib import Path


def get_connection(db_path: str = "radar.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str = "radar.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    for node in [1, 2, 3]:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS raw_node{node} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                frame_number INTEGER NOT NULL,
                point_cloud TEXT NOT NULL
            )
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fused_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            fused_cloud TEXT NOT NULL,
            num_points INTEGER NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")


def insert_raw_frame(conn, node_id: int, timestamp: float, frame_number: int, point_cloud: list):
    conn.execute(
        f"INSERT INTO raw_node{node_id} (timestamp, frame_number, point_cloud) VALUES (?, ?, ?)",
        (timestamp, frame_number, json.dumps(point_cloud))
    )


def insert_fused_frame(conn, timestamp: float, fused_cloud: list):
    conn.execute(
        "INSERT INTO fused_data (timestamp, fused_cloud, num_points) VALUES (?, ?, ?)",
        (timestamp, json.dumps(fused_cloud), len(fused_cloud))
    )


if __name__ == "__main__":
    init_db()
    print("Testing inserts...")
    conn = get_connection()
    insert_raw_frame(conn, 1, 1234567890.0, 1, [[0.1, 0.2, 0.3]])
    insert_fused_frame(conn, 1234567890.0, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    conn.commit()
    conn.close()
    print("Done. Check radar.db exists.")