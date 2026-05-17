import logging
from auto_updater import RetrospectiveAccuracyEngine

logging.basicConfig(level=logging.INFO)

def run_backfill():
    print("Starting Pension Accuracy Backfill (Corrected Logic)...")
    # Compute for the last 15 draws
    engine = RetrospectiveAccuracyEngine("lottery.db", mode="PENSION")
    results = engine.compute(start_draw_no=295)
    print(f"Backfill complete. {len(results)} records generated.")

if __name__ == "__main__":
    run_backfill()
