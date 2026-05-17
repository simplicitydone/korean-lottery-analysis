import sqlite3
import json

db_path = 'lottery.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

def check_table(table):
    print(f"--- Checking {table} ---")
    cur.execute(f"SELECT draw_no, actual_nums FROM {table} ORDER BY draw_no DESC LIMIT 5")
    rows = cur.fetchall()
    for row in rows:
        d_no, a_nums = row
        try:
            parsed = json.loads(a_nums)
            print(f"Draw {d_no}: {parsed} (Type: {type(parsed)})")
        except Exception as e:
            print(f"Draw {d_no}: ERROR parsing {a_nums} - {e}")

check_table("prediction_accuracy_v3")
check_table("prediction_accuracy_pension_v3")
conn.close()
