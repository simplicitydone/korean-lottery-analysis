import sqlite3
import datetime

DB_PATH = 'lottery.db'

# Data from https://signalfire85.tistory.com/277
# Format: (draw_no, draw_date, group_no, n1..n6, is_bonus)
# Note: n1..n6 are the 6 digits.
PENSION_DATA = [
    # 310회
    (310, '2026-04-09', 3, 0, 5, 6, 2, 0, 4, 0),
    (310, '2026-04-09', 0, 3, 3, 1, 1, 1, 9, 1), # Bonus
    # 309회
    (309, '2026-04-02', 4, 8, 6, 7, 5, 4, 9, 0),
    (309, '2026-04-02', 0, 6, 5, 3, 8, 4, 3, 1), # Bonus
    # 308회
    (308, '2026-03-26', 5, 9, 2, 0, 3, 8, 8, 0),
    (308, '2026-03-26', 0, 7, 5, 2, 7, 4, 4, 1), # Bonus
    # 307회
    (307, '2026-03-19', 1, 9, 8, 7, 6, 4, 5, 0),
    (307, '2026-03-19', 0, 8, 5, 9, 1, 2, 7, 1), # Bonus
    # 306회
    (306, '2026-03-12', 4, 5, 4, 5, 3, 8, 0, 0),
    (306, '2026-03-12', 0, 3, 9, 4, 4, 0, 2, 1), # Bonus
    # 305회
    (305, '2026-03-05', 2, 3, 3, 1, 3, 1, 6, 0),
    (305, '2026-03-05', 0, 5, 0, 9, 9, 8, 8, 1), # Bonus
    # 304회
    (304, '2026-02-26', 5, 2, 3, 4, 9, 3, 1, 0),
    (304, '2026-02-26', 0, 4, 3, 8, 9, 8, 5, 1), # Bonus
    # 303회
    (303, '2026-02-19', 4, 6, 3, 9, 5, 6, 6, 0),
    (303, '2026-02-19', 0, 6, 1, 9, 1, 3, 6, 1), # Bonus
    # 302회
    (302, '2026-02-12', 4, 6, 3, 9, 9, 2, 7, 0),
    (302, '2026-02-12', 0, 0, 5, 5, 4, 9, 5, 1), # Bonus
    # 301회
    (301, '2026-02-05', 4, 7, 4, 8, 5, 5, 9, 0),
    (301, '2026-02-05', 0, 3, 2, 3, 3, 8, 4, 1), # Bonus
]

def backfill():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # 1. Clear existing bad data for these draws to ensure clean insert
    draw_nos = list(set([d[0] for d in PENSION_DATA]))
    cur.execute(f"DELETE FROM pension_results WHERE draw_no IN ({','.join(map(str, draw_nos))})")
    print(f"Cleared existing data for draws: {draw_nos}")

    # 2. Insert corrected data
    cur.executemany("""
        INSERT INTO pension_results (draw_no, draw_date, group_no, n1, n2, n3, n4, n5, n6, is_bonus)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, PENSION_DATA)
    
    # 3. Correct prediction_accuracy_pension_v3 dates as well if they exist
    cur.execute(f"UPDATE prediction_accuracy_pension_v3 SET draw_date = '2026-04-09' WHERE draw_no = 310")
    
    conn.commit()
    print(f"Successfully backfilled {len(PENSION_DATA)} records.")
    conn.close()

if __name__ == "__main__":
    backfill()
