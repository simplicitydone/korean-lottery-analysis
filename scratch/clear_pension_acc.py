import sqlite3
import os

db_path = "lottery.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM prediction_accuracy_pension_v3")
    conn.commit()
    print(f"Cleared {cur.rowcount} corrupt records from prediction_accuracy_pension_v3")
    conn.close()
else:
    print("Database not found")
