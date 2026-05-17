import sqlite3
import pandas as pd
import json

db_path = 'lottery.db'
conn = sqlite3.connect(db_path)

# Mocking the engine logic
query = "SELECT draw_no, draw_date, win1, win2, win3, win4, win5, win6, bonus FROM draw_results ORDER BY draw_no DESC LIMIT 8"
df = pd.read_sql(query, conn)
conn.close()

print(df.to_json(orient="records"))
