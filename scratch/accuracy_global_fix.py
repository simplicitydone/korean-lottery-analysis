import sqlite3
import json
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AccuracyMigration")

def migrate_global_best(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    tables = ["prediction_accuracy_v3", "prediction_accuracy_pension_v3"]
    
    for table in tables:
        log.info(f"Processing table: {table}")
        cursor.execute(f"SELECT draw_no, method_results FROM {table}")
        rows = cursor.fetchall()
        
        for draw_no, method_results_json in rows:
            try:
                method_results = json.loads(method_results_json)
                global_best = 0
                for m_name, m_data in method_results.items():
                    best_hit = m_data.get("best_hit", 0)
                    global_best = max(global_best, best_hit)
                
                # Update the row
                cursor.execute(f"UPDATE {table} SET ensemble_best = ? WHERE draw_no = ?", (global_best, draw_no))
            except Exception as e:
                log.error(f"Error processing draw {draw_no} in {table}: {e}")
    
    conn.commit()
    conn.close()
    log.info("Migration complete.")

if __name__ == "__main__":
    migrate_global_best("lottery.db")
