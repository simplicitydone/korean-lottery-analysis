import sqlite3
import json

DB_PATH = 'lottery.db'

def verify():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print("Checking pension_results for Draw 310:")
    cur.execute("SELECT draw_no, draw_date, matches, is_bonus FROM (SELECT draw_no, draw_date, (n1||n2||n3||n4||n5||n6) as matches, is_bonus FROM pension_results) WHERE draw_no=310")
    rows = cur.fetchall()
    for r in rows:
        print(f"  {r}")
        
    print("\nChecking prediction_accuracy_pension_v3 for Draw 310:")
    cur.execute("SELECT draw_no, draw_date, ensemble_best, method_results FROM prediction_accuracy_pension_v3 WHERE draw_no=310")
    row = cur.fetchone()
    if row:
        print(f"  Draw: {row[0]}, Date: {row[1]}, Best: {row[2]}")
        # Check if method_results contains the new hits_bonus field (it might not yet because I haven't re-run the compute)
        methods = json.loads(row[3])
        first_method = list(methods.values())[0] if methods else {}
        print(f"  Sample Method Keys: {list(first_method.keys())}")
    else:
        print("  Accuracy not found for 310.")
        
    conn.close()

if __name__ == "__main__":
    verify()
