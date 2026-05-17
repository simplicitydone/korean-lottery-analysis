import sqlite3
import json

db_path = 'lottery.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def get_tables():
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [t[0] for t in cursor.fetchall()]

def get_count(table):
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    return cursor.fetchone()[0]

def get_latest(table):
    try:
        cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 1")
        cols = [d[0] for d in cursor.description]
        row = cursor.fetchone()
        if row:
            return dict(zip(cols, row))
        return None
    except Exception as e:
        return str(e)

tables = get_tables()
summary = {}
for t in tables:
    summary[t] = {
        "count": get_count(t),
        "latest": get_latest(t)
    }

print(json.dumps(summary, indent=2, ensure_ascii=False))
conn.close()
