import sqlite3
import json
import os

DB_PATH = "lottery.db"

def load_accuracy_results(db_path, mode="LOTTO"):
    table_name = "prediction_accuracy_pension_v3" if mode.upper() == "PENSION" else "prediction_accuracy_v3"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT draw_no, draw_date, actual_nums, ensemble_best, ensemble_hits, method_results, training_size, computed_at FROM {table_name} ORDER BY draw_no ASC")
    rows = cur.fetchall()
    conn.close()
    out = []
    for row in rows:
        out.append({
            "draw_no":       row[0],
            "draw_date":     row[1],
            "actual_nums":   json.loads(row[2]),
            "ensemble_best": row[3],
            "ensemble_hits": json.loads(row[4]),
            "method_results":json.loads(row[5]),
            "training_size": row[6],
            "computed_at":   row[7]
        })
    return out

results = load_accuracy_results(DB_PATH, mode="LOTTO")
# Just check the latest one
if results:
    r = results[-1]
    # Apply sanitization like app.py
    r["best_1st"] = r.get("ensemble_best", 0)
    r["best_bonus"] = 0
    print(json.dumps(r, indent=2))
else:
    print("EMPTY")
