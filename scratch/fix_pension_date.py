import sqlite3
conn = sqlite3.connect('lottery.db')
cur = conn.cursor()
cur.execute("UPDATE pension_results SET draw_date='2026-04-09' WHERE draw_no=310 AND draw_date='2026-04-16'")
rowcount = cur.rowcount

# Also check prediction_accuracy_pension_v3
cur.execute("UPDATE prediction_accuracy_pension_v3 SET draw_date='2026-04-09' WHERE draw_no=310 AND draw_date='2026-04-16'")
rowcount_acc = cur.rowcount

conn.commit()
print(f"pension_results updated: {rowcount}")
print(f"prediction_accuracy_pension_v3 updated: {rowcount_acc}")
conn.close()
