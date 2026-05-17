import argparse
import json
import time

from auto_updater import RetrospectiveAccuracyEngine, get_uncomputed_draw_nos


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill lottery prediction accuracy records.")
    parser.add_argument("--db", default="lottery.db", help="SQLite database path")
    parser.add_argument("--mode", choices=["LOTTO", "PENSION"], default="LOTTO")
    parser.add_argument("--start", type=int, default=1, help="Minimum draw number to consider")
    parser.add_argument("--count", type=int, default=5, help="Maximum draws to compute")
    parser.add_argument("--dry-run", action="store_true", help="Print target draw numbers without computing")
    return parser.parse_args()


def main():
    args = parse_args()
    draw_nos = get_uncomputed_draw_nos(args.db, args.start, mode=args.mode)[:max(1, args.count)]
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "draw_nos": draw_nos}, ensure_ascii=False))
        return
    if not draw_nos:
        print(json.dumps({"mode": args.mode, "saved": 0, "message": "미계산 회차가 없습니다."}, ensure_ascii=False))
        return

    started_at = time.time()

    def progress(current, total, message):
        print(json.dumps({
            "mode": args.mode,
            "current": current,
            "total": total,
            "message": message,
        }, ensure_ascii=False), flush=True)

    engine = RetrospectiveAccuracyEngine(
        db_path=args.db,
        mode="accurate",
        lottery_type=args.mode,
        progress_callback=progress,
    )
    results = engine.compute(draw_nos_to_compute=draw_nos)
    print(json.dumps({
        "mode": args.mode,
        "requested": draw_nos,
        "saved": len(results),
        "elapsed_sec": round(time.time() - started_at, 1),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
