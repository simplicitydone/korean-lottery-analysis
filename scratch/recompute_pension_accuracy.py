import logging
from auto_updater import RetrospectiveAccuracyEngine, init_accuracy_table, sqlite3

logging.basicConfig(level=logging.INFO)

DB_PATH = 'lottery.db'

def recompute_pension_accuracy():
    # 1. Clear existing accuracy for recent draws to force re-calculation
    # Draws 301-310
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM prediction_accuracy_pension_v3 WHERE draw_no >= 301")
    conn.commit()
    conn.close()
    print("Cleared accuracy results for draws >= 301")

    # 2. Re-compute
    engine = RetrospectiveAccuracyEngine(DB_PATH, lottery_type="PENSION")
    # Setting start_draw_no will trigger get_uncomputed_draw_nos
    engine.compute(draw_nos_to_compute=list(range(301, 311)))
    print("Re-computation finished.")

if __name__ == "__main__":
    recompute_pension_accuracy()
