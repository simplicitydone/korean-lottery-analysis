"""
auto_updater.py — Lottery Auto-Update & Retrospective Accuracy Module (v2.0)

Features:
  1. Weekly-schedule-based fetch: lotto Sunday 10:00, pension Friday 10:00
  2. Fallback: retry once/day for up to 3 extra days (handles delay/reschedule)
  3. Startup catch-up: fetches any missed draws since last DB entry
  4. Persistent accuracy archive: prediction_accuracy table in lottery.db
     - Each draw's prediction computed BEFORE that draw using prior data only
     - Saved once, never changed (tamper-proof credibility record)
  5. Auto-compute accuracy for every newly fetched lotto draw
"""

import sqlite3
import threading
import time
import re
import json
import datetime
import logging
import requests
import numpy as np

log = logging.getLogger("AutoUpdater")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# dhlottery.co.kr redesigned in 2026-05: the old gameResult.do pages redirect to
# /errorPage. The new site exposes JSON endpoints with draw-number range params.
LOTTO_URL   = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"
PENSION_URL = "https://www.dhlottery.co.kr/pt720/selectPstPt720Info.do"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://www.dhlottery.co.kr/",
}


# ═══════════════════════════ Scraping Functions ════════════════════════════

def _ymd_to_date(ymd: str) -> str:
    """'20260627' → '2026-06-27'."""
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def fetch_lotto(draw_no: int) -> dict | None:
    """Fetch one lotto draw from the dhlottery JSON API.
    Returns {draw_no, draw_date, win1..6, bonus} or None.
    """
    try:
        r = requests.get(LOTTO_URL, headers=HEADERS, timeout=15,
                         params={"srchStrLtEpsd": draw_no, "srchEndLtEpsd": draw_no})
        rows = (r.json().get("data") or {}).get("list") or []
        # The API returns the latest draw when the requested one doesn't exist yet.
        row = next((x for x in rows if x.get("ltEpsd") == draw_no), None)
        if not row:
            log.warning(f"lotto draw {draw_no} not in API response "
                        f"(latest available: {rows[0]['ltEpsd'] if rows else 'none'})")
            return None

        nums = [row[f"tm{i}WnNo"] for i in range(1, 7)]
        bonus = row["bnsWnNo"]
        if any(n is None for n in nums) or bonus is None:
            log.warning(f"lotto draw {draw_no}: incomplete numbers in API response")
            return None

        draw_date = (_ymd_to_date(str(row["ltRflYmd"])) if row.get("ltRflYmd")
                     else str(datetime.date(2002, 12, 7) + datetime.timedelta(weeks=draw_no - 1)))

        return {"draw_no": draw_no, "draw_date": draw_date,
                "win1": int(nums[0]), "win2": int(nums[1]), "win3": int(nums[2]),
                "win4": int(nums[3]), "win5": int(nums[4]), "win6": int(nums[5]),
                "bonus": int(bonus)}
    except Exception as e:
        log.error(f"fetch_lotto({draw_no}): {e}")
        return None


def fetch_pension(draw_no: int) -> list[dict] | None:
    """Fetch one pension (연금복권720+) draw from the dhlottery JSON API.
    Returns a list of records (1st prize + Bonus) or None.

    API rows: wnSqNo 1 = 1st prize (wnBndNo = 조, wnRnkVl = 6 digits, may have
    leading zeros), wnSqNo 21 = bonus (wnRnkVl = 6 digits).
    """
    try:
        r = requests.get(PENSION_URL, headers=HEADERS, timeout=15,
                         params={"srchStrPsltEpsd": draw_no, "srchEndPsltEpsd": draw_no})
        rows = (r.json().get("data") or {}).get("result") or []
        # The API returns the latest draw when the requested one doesn't exist yet.
        rows = [x for x in rows if x.get("psltEpsd") == draw_no]
        if not rows:
            log.warning(f"pension draw {draw_no} not in API response")
            return None

        results = []
        for row in rows:
            sq = row.get("wnSqNo")
            if sq not in (1, 21):
                continue
            digits = str(row.get("wnRnkVl") or "")
            if len(digits) != 6 or not digits.isdigit():
                log.warning(f"pension draw {draw_no} wnSqNo={sq}: bad digits {digits!r}")
                continue
            draw_date = (_ymd_to_date(str(row["psltRflYmd"])) if row.get("psltRflYmd")
                         else str(datetime.date(2020, 5, 7) + datetime.timedelta(weeks=draw_no - 1)))
            is_bonus = 1 if sq == 21 else 0
            group_no = 0 if is_bonus else int(row.get("wnBndNo") or 0)
            results.append({
                "draw_no": draw_no, "draw_date": draw_date, "group_no": group_no,
                "n1": int(digits[0]), "n2": int(digits[1]), "n3": int(digits[2]),
                "n4": int(digits[3]), "n5": int(digits[4]), "n6": int(digits[5]),
                "is_bonus": is_bonus
            })

        return results if results else None
    except Exception as e:
        log.error(f"fetch_pension({draw_no}): {e}")
        return None


# ═══════════════════════════ Draw DB Operations ════════════════════════════

def get_latest_draw_no(db_path: str, mode: str) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if mode == "lotto":
        cur.execute("SELECT MAX(draw_no) FROM draw_results")
    else:
        cur.execute("SELECT MAX(draw_no) FROM pension_results WHERE is_bonus=0")
    result = cur.fetchone()[0] or 0
    conn.close()
    return result


def insert_lotto(db_path: str, data: dict) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM draw_results WHERE draw_no=?", (data["draw_no"],))
        if cur.fetchone()[0] > 0:
            conn.close()
            return False
        cur.execute("""
            INSERT INTO draw_results (draw_no,draw_date,win1,win2,win3,win4,win5,win6,bonus)
            VALUES (:draw_no,:draw_date,:win1,:win2,:win3,:win4,:win5,:win6,:bonus)
        """, data)
        conn.commit()
        conn.close()
        log.info(f"Inserted lotto {data['draw_no']} ({data['draw_date']}): "
                 f"{data['win1']}-{data['win2']}-{data['win3']}-"
                 f"{data['win4']}-{data['win5']}-{data['win6']} +{data['bonus']}")
        return True
    except Exception as e:
        log.error(f"insert_lotto: {e}")
        return False


def insert_pension(db_path: str, data: dict) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM pension_results WHERE draw_no=? AND is_bonus=?",
                    (data["draw_no"], data["is_bonus"]))
        if cur.fetchone()[0] > 0:
            conn.close()
            return False
        cur.execute("""
            INSERT INTO pension_results (draw_no,draw_date,group_no,n1,n2,n3,n4,n5,n6,is_bonus)
            VALUES (:draw_no,:draw_date,:group_no,:n1,:n2,:n3,:n4,:n5,:n6,:is_bonus)
        """, data)
        conn.commit()
        conn.close()
        type_str = "Bonus" if data["is_bonus"] else f"{data['group_no']}조"
        log.info(f"Inserted pension {data['draw_no']} ({data['draw_date']}) - {type_str}: "
                 f"{data['n1']}{data['n2']}{data['n3']}{data['n4']}{data['n5']}{data['n6']}")
        return True
    except Exception as e:
        log.error(f"insert_pension: {e}")
        return False


# ═══════════════════════════ Accuracy DB Layer ═════════════════════════════

_ACCURACY_DDL = """
CREATE TABLE IF NOT EXISTS prediction_accuracy_v3 (
    draw_no         INTEGER PRIMARY KEY,
    draw_date       TEXT,
    actual_nums     TEXT,
    ensemble_best   INTEGER,
    ensemble_hits   TEXT,
    method_results  TEXT,
    training_size   INTEGER,
    computed_at     TEXT
)
"""
_ACCURACY_PENSION_DDL = """
CREATE TABLE IF NOT EXISTS prediction_accuracy_pension_v3 (
    draw_no         INTEGER PRIMARY KEY,
    draw_date       TEXT,
    actual_nums     TEXT,
    ensemble_best   INTEGER,
    ensemble_hits   TEXT,
    method_results  TEXT,
    training_size   INTEGER,
    computed_at     TEXT
)
"""

def init_accuracy_table(db_path: str):
    """Create prediction_accuracy tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute(_ACCURACY_DDL)
    conn.execute(_ACCURACY_PENSION_DDL)
    conn.commit()
    conn.close()


def save_accuracy_result(db_path: str, result: dict, mode: str = "LOTTO"):
    table_name = "prediction_accuracy_pension_v3" if mode.upper() == "PENSION" else "prediction_accuracy_v3"
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(f"""
            INSERT OR IGNORE INTO {table_name}
            (draw_no, draw_date, actual_nums, ensemble_best, 
             ensemble_hits, method_results, training_size, computed_at)
            VALUES (?,?,?,?, ?,?,?,?)
        """, (
            result["draw_no"], result["draw_date"],
            json.dumps(result["actual_nums"]),
            result["ensemble_best"],
            json.dumps(result["ensemble_hits"]),
            json.dumps(result.get("method_results", {})),
            result.get("training_size", 0),
            datetime.datetime.now().isoformat(timespec="seconds")
        ))
        conn.commit()
        conn.close()
        log.info(f"Saved accuracy draw {result['draw_no']}: best_hit={result['ensemble_best']}/6")
    except Exception as e:
        log.error(f"save_accuracy_result: {e}")


def load_accuracy_results(db_path: str, mode: str = "LOTTO") -> list:
    """Load all stored results, sorted draw_no ascending."""
    table_name = "prediction_accuracy_pension_v3" if mode.upper() == "PENSION" else "prediction_accuracy_v3"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"""
            SELECT draw_no, draw_date, actual_nums, ensemble_best,
                   ensemble_hits, method_results, training_size, computed_at
            FROM {table_name} ORDER BY draw_no ASC
        """)
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
    except Exception as e:
        log.error(f"load_accuracy_results: {e}")
        return []


def get_uncomputed_draw_nos(db_path: str, start_draw_no: int, mode: str = "LOTTO") -> list:
    """Return draw_nos >= start_draw_no that are NOT yet in prediction_accuracy."""
    src_table = "pension_results" if mode.upper() == "PENSION" else "draw_results"
    dst_table = "prediction_accuracy_pension_v3" if mode.upper() == "PENSION" else "prediction_accuracy_v3"
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT draw_no FROM {src_table} WHERE draw_no >= ?", (start_draw_no,))
        all_draws = {row[0] for row in cur.fetchall()}
        
        cur.execute(f"SELECT draw_no FROM {dst_table} WHERE draw_no >= ?", (start_draw_no,))
        computed = {row[0] for row in cur.fetchall()}
        conn.close()
        return sorted(list(all_draws - computed))
    except Exception as e:
        log.error(f"get_uncomputed_draw_nos: {e}")
        return []


# ═══════════════════════════ Auto-Update Scheduler ═════════════════════════

class LotteryAutoUpdater:
    """Weekly-schedule-based lottery draw fetcher.

    Schedule:
      Lotto   draws every Saturday  → check Sunday  10:00 KST
      Pension draws every Thursday  → check Friday  10:00 KST

    Fallback (draw delayed or rescheduled):
      If draw not available on scheduled day, retry once/day for up to
      MAX_RETRIES additional days, then wait until next week.
    """

    LOTTO_CHECK_DAY   = 6   # Sunday   (Mon=0…Sun=6)
    PENSION_CHECK_DAY = 4   # Friday
    CHECK_HOUR        = 10  # 10:00
    MAX_RETRIES       = 3
    MAX_CATCHUP       = 100  # startup catch-up: max consecutive draws to backfill

    def __init__(self, db_path: str, on_lotto_update=None, on_pension_update=None):
        self.db_path = db_path
        self.on_lotto_update = on_lotto_update
        self.on_pension_update = on_pension_update
        self._stop = threading.Event()
        self._timer: threading.Timer | None = None

    def start(self):
        log.info("AutoUpdater started")
        threading.Thread(target=self._startup_catchup, daemon=True).start()

    def stop(self):
        self._stop.set()
        if self._timer:
            self._timer.cancel()

    # ── Startup ───────────────────────────────────────────────────────────

    def _startup_catchup(self):
        """Catch up on missed draws, then schedule next weekly check."""
        init_accuracy_table(self.db_path)
        try:
            # Lotto catch-up (until no newer draw is available)
            latest = get_latest_draw_no(self.db_path, "lotto")
            for delta in range(1, self.MAX_CATCHUP + 1):
                data = fetch_lotto(latest + delta)
                if data:
                    inserted = insert_lotto(self.db_path, data)
                    if inserted:
                        self._compute_accuracy_async(latest + delta, lottery_type="LOTTO")
                        if self.on_lotto_update:
                            self.on_lotto_update(data)
                    time.sleep(1)
                else:
                    break

            # Pension catch-up
            latest = get_latest_draw_no(self.db_path, "pension")
            for delta in range(1, self.MAX_CATCHUP + 1):
                results = fetch_pension(latest + delta)
                if results:
                    inserted_any = False
                    for data in results:
                        inserted = insert_pension(self.db_path, data)
                        inserted_any = inserted_any or inserted
                        if inserted and self.on_pension_update:
                            self.on_pension_update(data)
                    if inserted_any:
                        self._compute_accuracy_async(latest + delta, lottery_type="PENSION")
                    time.sleep(1)
                else:
                    break
        except Exception as e:
            log.error(f"Startup catch-up: {e}")
        finally:
            self._schedule_next()

    # ── Weekly scheduler ──────────────────────────────────────────────────

    def _seconds_until(self, weekday: int, hour: int) -> float:
        """Seconds until next occurrence of (weekday, hour:00)."""
        now = datetime.datetime.now()
        days = (weekday - now.weekday()) % 7
        if days == 0 and now.hour >= hour:
            days = 7
        target = (now + datetime.timedelta(days=days)).replace(
            hour=hour, minute=0, second=0, microsecond=0)
        return max(0.0, (target - now).total_seconds())

    def _schedule_next(self):
        if self._stop.is_set():
            return
        sl = self._seconds_until(self.LOTTO_CHECK_DAY,   self.CHECK_HOUR)
        sp = self._seconds_until(self.PENSION_CHECK_DAY, self.CHECK_HOUR)
        delay = min(sl, sp)
        at = datetime.datetime.now() + datetime.timedelta(seconds=delay)
        label = "lotto(Sun)" if sl < sp else "pension(Fri)"
        log.info(f"Next auto-update: {at.strftime('%Y-%m-%d %H:%M')} [{label}]")
        self._timer = threading.Timer(delay, self._weekly_check)
        self._timer.daemon = True
        self._timer.start()

    def _weekly_check(self):
        if self._stop.is_set():
            return
        today = datetime.datetime.now().weekday()
        if today == self.LOTTO_CHECK_DAY:
            self._try_fetch_new("lotto")
        if today == self.PENSION_CHECK_DAY:
            self._try_fetch_new("pension")
        self._schedule_next()

    def _try_fetch_new(self, mode: str):
        latest = get_latest_draw_no(self.db_path, mode)
        next_no = latest + 1
        if mode == "lotto":
            data = fetch_lotto(next_no)
            if data:
                if insert_lotto(self.db_path, data):
                    self._compute_accuracy_async(next_no, lottery_type="LOTTO")
                    if self.on_lotto_update:
                        self.on_lotto_update(data)
                return
        else:
            p_results = fetch_pension(next_no)
            if p_results:
                inserted_any = False
                for data in p_results:
                    if insert_pension(self.db_path, data):
                        inserted_any = True
                        if self.on_pension_update: self.on_pension_update(data)
                if inserted_any:
                    self._compute_accuracy_async(next_no, lottery_type="PENSION")
                return
        log.warning(f"{mode} draw {next_no} not yet available — fallback retry in 24h")
        self._schedule_fallback(mode, next_no, retry=1)

    def _schedule_fallback(self, mode: str, draw_no: int, retry: int):
        if self._stop.is_set() or retry > self.MAX_RETRIES:
            log.warning(f"{mode} draw {draw_no}: gave up after {self.MAX_RETRIES} retries")
            return

        def _retry():
            if self._stop.is_set():
                return
            if mode == "lotto":
                data = fetch_lotto(draw_no)
                if data and insert_lotto(self.db_path, data):
                    self._compute_accuracy_async(draw_no, lottery_type="LOTTO")
                    if self.on_lotto_update: self.on_lotto_update(data)
                elif data:
                    log.info(f"{mode} draw {draw_no} already in DB")
                else:
                    log.warning(f"{mode} draw {draw_no} still unavailable (retry {retry}/{self.MAX_RETRIES})")
                    self._schedule_fallback(mode, draw_no, retry + 1)
            else:
                p_results = fetch_pension(draw_no)
                if p_results:
                    inserted_any = False
                    for data in p_results:
                        inserted_any = insert_pension(self.db_path, data) or inserted_any
                    if inserted_any:
                        self._compute_accuracy_async(draw_no, lottery_type="PENSION")
                    if self.on_pension_update: self.on_pension_update(p_results[0])
                else:
                    log.warning(f"{mode} draw {draw_no} still unavailable (retry {retry}/{self.MAX_RETRIES})")
                    self._schedule_fallback(mode, draw_no, retry + 1)

        t = threading.Timer(24 * 3600, _retry)
        t.daemon = True
        t.start()
        log.info(f"Fallback retry #{retry} for {mode} draw {draw_no} in 24h")

    def _compute_accuracy_async(self, draw_no: int, lottery_type: str = "LOTTO"):
        """Compute and save accuracy for draw_no in a background thread."""
        def _run():
            try:
                engine = RetrospectiveAccuracyEngine(
                    db_path=self.db_path,
                    mode="accurate",
                    lottery_type=lottery_type,
                )
                engine.compute_single(draw_no)
            except Exception as e:
                log.error(f"_compute_accuracy_async({lottery_type}, {draw_no}): {e}")
        threading.Thread(target=_run, daemon=True).start()

    def manual_update(self, mode: str, data: dict) -> bool:
        """Manually insert a draw result (fallback when scraping fails)."""
        return (insert_lotto if mode == "lotto" else insert_pension)(self.db_path, data)


# ═══════════════════════════ Retrospective Accuracy ════════════════════════

class RetrospectiveAccuracyEngine:
    """Computes and persists prediction accuracy for lotto draws.

    For each draw N, trains MLP on draws 1..N-1 ONLY (no data leakage),
    generates probability ranking, records hit counts, saves to DB.

    Once saved to DB, results are immutable — they serve as a tamper-proof
    prediction record for credibility evidence.
    """

    def __init__(self, db_path: str, mode: str = "accurate",
                 progress_callback=None, lottery_type: str = "LOTTO"):
        self.db_path = db_path
        self.mode = mode
        self.lottery_type = lottery_type
        self.progress_callback = progress_callback  # fn(current, total, msg)

    def _load_all_draws(self):
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        table = "pension_results" if self.lottery_type.upper() == "PENSION" else "draw_results"
        query = (
            f"SELECT draw_no, draw_date, group_no, n1 as win1, n2 as win2, n3 as win3, n4 as win4, n5 as win5, n6 as win6 FROM {table} WHERE is_bonus=0 ORDER BY draw_no ASC" 
            if self.lottery_type.upper() == "PENSION" else 
            f"SELECT draw_no, draw_date, win1, win2, win3, win4, win5, win6 FROM {table} ORDER BY draw_no ASC"
        )
        df = pd.read_sql(query, conn)
        conn.close()
        for col in ["win1","win2","win3","win4","win5","win6"]:
            df[col] = df[col].astype(int)
        return df

    def _build_result(self, draw_no: int, draw_date: str, actual_data: list,
                      methods_outputs: dict, training_size: int) -> dict:
        
        method_results = {}
        is_pension = (self.lottery_type.upper() == "PENSION")
        
        # for pension, actual_data is a list of [ (group, n1..n6, is_bonus), ... ]
        # or simplified [group, n1..n6] for 1st and [0, b1..b6] for Bonus
        a_1st = actual_data[0] if is_pension else actual_data
        a_bonus = actual_data[1] if is_pension and len(actual_data) > 1 else None

        global_best_1st = 0
        global_best_bonus = 0
        
        ensemble_hits = []
        for m_name, m_data in methods_outputs.items():
            sets = m_data.get("sets", [])
            hits_1st = []
            hits_bonus = []
            
            for s in sets:
                if is_pension:
                    p_group = s.get("group")
                    p_digits = s.get("digits", [])
                    
                    # 1. Check against 1st Prize Draw
                    count_1st = 0
                    if p_digits and a_1st:
                        a_digits_1st = a_1st[1:]
                        for i in range(5, -1, -1):
                            if p_digits[i] == a_digits_1st[i]: count_1st += 1
                            else: break
                    if count_1st == 6 and p_group == a_1st[0]: count_1st = 7
                    hits_1st.append(count_1st)
                    
                    # 2. Check against Bonus Prize Draw (2nd Prize)
                    count_bonus = 0
                    if p_digits and a_bonus:
                        a_digits_bonus = a_bonus[1:]
                        for i in range(5, -1, -1):
                            if p_digits[i] == a_digits_bonus[i]: count_bonus += 1
                            else: break
                    hits_bonus.append(count_bonus)
                else:
                    nums = set(s.get("numbers", []))
                    hits_1st.append(len(nums & set(actual_data)))
            
            best_1st = max(hits_1st) if hits_1st else 0
            best_bonus = max(hits_bonus) if hits_bonus else 0
            global_best_1st = max(global_best_1st, best_1st)
            global_best_bonus = max(global_best_bonus, best_bonus)
            
            if "Ensemble" in m_name:
                ensemble_hits = hits_1st

            generated = []
            for s in sets:
                if is_pension:
                    generated.append({"group": s.get("group"), "digits": s.get("digits")})
                else:
                    generated.append(s.get("numbers"))

            method_results[m_name] = {
                "hits_1st": hits_1st,
                "hits_bonus": hits_bonus,
                "best_1st": best_1st,
                "best_bonus": best_bonus,
                "sets_generated": generated
            }
            
        return {
            "draw_no":       draw_no,
            "draw_date":     draw_date,
            "actual_nums":   actual_data,
            "ensemble_best": global_best_1st, # Compatible with legacy UI
            "ensemble_hits": ensemble_hits,  # Restore for DB persistence
            "pension_best":  {"1st": global_best_1st, "bonus": global_best_bonus},
            "training_size": training_size,
            "method_results": method_results,
            "mode":          self.mode,
        }

    def compute(self, draw_nos_to_compute: list | None = None,
                start_draw_no: int = 1) -> list:
        """Compute accuracy for specified draws."""
        from predictor_engine import LottoPredictor
        from pension_engine import PensionPredictor
        
        df_all = self._load_all_draws()
        if draw_nos_to_compute is None:
            draw_nos_to_compute = get_uncomputed_draw_nos(self.db_path, start_draw_no, mode=self.lottery_type)

        total = len(draw_nos_to_compute)
        if total == 0:
            if self.progress_callback:
                self.progress_callback(0, 0, "모든 회차 이미 계산 완료.")
            return []

        new_results = []
        for idx, draw_no in enumerate(draw_nos_to_compute):
            rows = df_all[df_all["draw_no"] == draw_no]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if self.lottery_type.upper() == "PENSION":
                # Fetch both 1st and Bonus
                conn = sqlite3.connect(self.db_path)
                curr = conn.cursor()
                curr.execute("SELECT group_no, n1, n2, n3, n4, n5, n6, is_bonus FROM pension_results WHERE draw_no=?", (draw_no,))
                p_rows = curr.fetchall()
                conn.close()
                actual_data = []
                # Sort so results[0] is 1st prize (is_bonus=0)
                p_rows = sorted(p_rows, key=lambda x: x[7])
                for r in p_rows:
                    actual_data.append([int(r[i]) for i in range(7)])
            else:
                actual_data = [int(row[f"win{i}"]) for i in range(1, 7)]
            
            draw_date = str(row["draw_date"])[:10]
            
            # Using max_draw_no guarantees no data leakage
            df_train = df_all[df_all["draw_no"] < draw_no]
            if len(df_train) < 20:
                log.warning(f"Skipping draw {draw_no}: only {len(df_train)} training draws")
                continue

            if self.progress_callback:
                self.progress_callback(
                    idx + 1, total,
                    f"DL 앙상블 백테스트 Draw {draw_no} ({idx+1}/{total}) | 학습: {len(df_train)}회차"
                )

            # Delegate ML to predictor_engine logic to keep them in sync
            if self.lottery_type.upper() == "PENSION":
                predictor = PensionPredictor(self.db_path, max_draw_no=draw_no - 1)
            else:
                predictor = LottoPredictor(self.db_path, max_draw_no=draw_no - 1)
            
            # Fetch the actual 5-ticket generation representing realistic user purchases
            methods_outputs = predictor.predict_all_methods()

            result = self._build_result(draw_no, draw_date, actual_data, methods_outputs, len(df_train))
            save_accuracy_result(self.db_path, result, mode=self.lottery_type)   # persist immediately
            new_results.append(result)

        if self.progress_callback:
            self.progress_callback(total, total, f"소급 분석 완료! {len(new_results)}회 저장됨")
        return new_results

    def compute_single(self, draw_no: int) -> dict | None:
        """Compute accuracy for one specific draw (e.g. after auto-fetch).
        Returns the result if newly computed, None if already in DB.
        """
        existing = get_uncomputed_draw_nos(self.db_path, draw_no, mode=self.lottery_type)
        if draw_no not in existing:
            log.info(f"Accuracy for draw {draw_no} already in DB — skipped.")
            return None
        results = self.compute(draw_nos_to_compute=[draw_no])
        return results[0] if results else None
