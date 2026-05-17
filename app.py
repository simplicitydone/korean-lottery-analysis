from functools import wraps
import threading
import sqlite3
from flask import Flask, jsonify, request, send_from_directory, session
import os
import logging
from predictor_engine import LottoPredictor
from pension_engine import PensionPredictor
from auto_updater import (
    LotteryAutoUpdater,
    RetrospectiveAccuracyEngine,
    get_uncomputed_draw_nos,
    load_accuracy_results,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("LottoWebApp")


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.environ.get("LOTTERY_SECRET_KEY", "change-this-secret-key")

# Shared Database Path
DB_PATH = "lottery.db"
ACCESS_PASSWORD = os.environ.get("LOTTERY_ACCESS_PASSWORD", "1emdrkwmdk!")
if app.secret_key == "change-this-secret-key" or ACCESS_PASSWORD == "1emdrkwmdk!":
    log.warning("Using default LOTTERY_SECRET_KEY or LOTTERY_ACCESS_PASSWORD. Set .env for private use.")

# Initialize Engines
log.info("Initializing engines...")
lotto_engine = LottoPredictor(DB_PATH, train_on_init=False)
pension_engine = PensionPredictor(DB_PATH, train_on_init=False)

# Setup Auto-Updater
updater = None
backfill_lock = threading.Lock()
backfill_jobs = {
    "LOTTO": {"running": False, "current": 0, "total": 0, "message": "대기 중", "last_result": None},
    "PENSION": {"running": False, "current": 0, "total": 0, "message": "대기 중", "last_result": None},
}

def on_data_update(data):
    log.info(f"Auto-update detected new data: {data}")
    # Refresh engines if needed
    lotto_engine.load_history()
    lotto_engine._trained = False
    pension_engine.load_history()
    pension_engine._trained = False

def start_auto_updater():
    global updater
    if os.environ.get("LOTTERY_AUTO_UPDATE", "1") == "0":
        log.info("AutoUpdater disabled by LOTTERY_AUTO_UPDATE=0")
        return
    if updater is not None:
        return
    log.info("Starting AutoUpdater...")
    updater = LotteryAutoUpdater(
        DB_PATH, 
        on_lotto_update=on_data_update, 
        on_pension_update=on_data_update
    )
    updater.start()


def warm_models_async():
    if os.environ.get("LOTTERY_WARM_MODELS", "1") == "0":
        return

    def _run():
        try:
            log.info("Warming prediction models...")
            lotto_engine.ensure_trained()
            pension_engine.ensure_trained()
            log.info("Prediction models ready.")
        except Exception as e:
            log.error(f"Model warmup failed: {e}")

    threading.Thread(target=_run, daemon=True).start()

# ─────────────────────────── API Endpoints ─────────────────────────────────

def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authorized"):
            return jsonify({"error": "인증이 필요합니다."}), 401
        return view(*args, **kwargs)
    return wrapped


def _pension_best_from_methods(method_results):
    best_1st = 0
    best_bonus = 0
    for result in (method_results or {}).values():
        best_1st = max(best_1st, int(result.get("best_1st", 0) or 0))
        best_bonus = max(best_bonus, int(result.get("best_bonus", 0) or 0))
    return best_1st, best_bonus


def _accuracy_coverage(mode):
    is_pension = mode == "PENSION"
    src_table = "pension_results" if is_pension else "draw_results"
    dst_table = "prediction_accuracy_pension_v3" if is_pension else "prediction_accuracy_v3"
    src_where = "WHERE is_bonus=0" if is_pension else ""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*), MIN(draw_no), MAX(draw_no) FROM {src_table} {src_where}")
        total_count, source_min, source_max = cur.fetchone()
        cur.execute(f"SELECT COUNT(*), MIN(draw_no), MAX(draw_no) FROM {dst_table}")
        computed_count, computed_min, computed_max = cur.fetchone()
    coverage = (computed_count / total_count * 100) if total_count else 0
    return {
        "mode": mode,
        "total_draws": int(total_count or 0),
        "computed_draws": int(computed_count or 0),
        "missing_draws": int((total_count or 0) - (computed_count or 0)),
        "coverage_pct": round(coverage, 1),
        "source_min": int(source_min or 0),
        "source_max": int(source_max or 0),
        "computed_min": int(computed_min or 0),
        "computed_max": int(computed_max or 0),
    }


def _backfill_status(mode):
    with backfill_lock:
        status = dict(backfill_jobs[mode])
    status["coverage"] = _accuracy_coverage(mode)
    return status


def _run_backfill(mode, draw_nos):
    def progress(current, total, message):
        with backfill_lock:
            backfill_jobs[mode].update({"current": current, "total": total, "message": message})

    try:
        engine = RetrospectiveAccuracyEngine(
            db_path=DB_PATH,
            mode="accurate",
            lottery_type=mode,
            progress_callback=progress,
        )
        results = engine.compute(draw_nos_to_compute=draw_nos)
        with backfill_lock:
            backfill_jobs[mode].update({
                "running": False,
                "current": len(draw_nos),
                "total": len(draw_nos),
                "message": f"완료: {len(results)}회 저장",
                "last_result": {"requested": draw_nos, "saved": len(results)},
            })
    except Exception as e:
        log.error(f"Backfill failed ({mode}): {e}")
        with backfill_lock:
            backfill_jobs[mode].update({
                "running": False,
                "message": f"오류: {e}",
                "last_result": {"requested": draw_nos, "error": str(e)},
            })

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.errorhandler(Exception)
def handle_exception(e):
    log.error(f"Unhandled Exception: {e}")
    return jsonify({"error": "데이터 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.", "details": str(e)}), 500

@app.route("/api/status")
def get_status():
    lotto_latest = lotto_engine.df['draw_no'].max() if not lotto_engine.df.empty else 0
    pension_latest = pension_engine.df['draw_no'].max() if not pension_engine.df.empty else 0
    
    # Use fallback 0 for missing data to prevent NoneType errors
    return jsonify({
        "lotto_latest": int(lotto_latest or 0),
        "pension_latest": int(pension_latest or 0),
        "status": "online",
        "authenticated": bool(session.get("authorized"))
    })

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    if data.get("password") == ACCESS_PASSWORD:
        session["authorized"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "잘못된 비밀번호입니다."}), 401

@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("authorized", None)
    return jsonify({"ok": True})

@app.route("/api/predict/lotto")
@require_auth
def predict_lotto():
    res = lotto_engine.predict_all_methods()
    # Ensemble Best 5
    all_sets = []
    for m_data in res.values():
        all_sets.extend(m_data["sets"])
    best5 = sorted(all_sets, key=lambda x: x["score"], reverse=True)[:5]
    
    return jsonify({
        "methods": res,
        "best5": best5
    })

@app.route("/api/predict/pension")
@require_auth
def predict_pension():
    res = pension_engine.predict_all_methods()
    all_sets = []
    for m_data in res.values():
        all_sets.extend(m_data["sets"])
    best5 = sorted(all_sets, key=lambda x: x["score"], reverse=True)[:5]
    
    return jsonify({
        "methods": res,
        "best5": best5
    })

@app.route("/api/history/lotto")
def history_lotto():
    count = request.args.get("count", default=10, type=int)
    history = lotto_engine.get_latest_history(count)
    return jsonify(history.to_dict(orient="records"))

@app.route("/api/history/pension")
def history_pension():
    count = request.args.get("count", default=10, type=int)
    history = pension_engine.get_latest_history(count)
    return jsonify(history.to_dict(orient="records"))

@app.route("/api/accuracy")
def get_accuracy():
    mode = request.args.get('mode', 'LOTTO').upper()
    try:
        results = load_accuracy_results(DB_PATH, mode=mode)
        sanitized = []
        for r in results:
            # Map mode-specific fields to standardized frontend names
            if mode == "LOTTO":
                r["best_1st"] = r.get("ensemble_best", 0)
                r["best_bonus"] = 0
            else:
                best_1st, best_bonus = _pension_best_from_methods(r.get("method_results"))
                r["best_1st"] = best_1st or r.get("ensemble_best", 0)
                r["best_bonus"] = best_bonus
            sanitized.append(r)
        return jsonify(sanitized)
    except Exception as e:
        log.error(f"API Accuracy Error: {e}")
        return jsonify([])

@app.route("/api/accuracy/coverage")
def get_accuracy_coverage():
    mode = request.args.get('mode', 'LOTTO').upper()
    if mode not in {"LOTTO", "PENSION"}:
        return jsonify({"error": "mode는 LOTTO 또는 PENSION이어야 합니다."}), 400
    try:
        return jsonify(_accuracy_coverage(mode))
    except Exception as e:
        log.error(f"API Accuracy Coverage Error: {e}")
        return jsonify({"error": "정확도 커버리지 계산 중 오류가 발생했습니다."}), 500

@app.route("/api/accuracy/backfill/status")
@require_auth
def get_backfill_status():
    mode = request.args.get('mode', 'LOTTO').upper()
    if mode not in {"LOTTO", "PENSION"}:
        return jsonify({"error": "mode는 LOTTO 또는 PENSION이어야 합니다."}), 400
    return jsonify(_backfill_status(mode))

@app.route("/api/accuracy/backfill/start", methods=["POST"])
@require_auth
def start_backfill():
    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "LOTTO")).upper()
    if mode not in {"LOTTO", "PENSION"}:
        return jsonify({"error": "mode는 LOTTO 또는 PENSION이어야 합니다."}), 400
    count = max(1, min(int(data.get("count", 5) or 5), 25))
    start_draw_no = max(1, int(data.get("start_draw_no", 1) or 1))

    with backfill_lock:
        if backfill_jobs[mode]["running"]:
            return jsonify({"error": "이미 백필이 실행 중입니다.", "status": dict(backfill_jobs[mode])}), 409

    draw_nos = get_uncomputed_draw_nos(DB_PATH, start_draw_no, mode=mode)[:count]
    if not draw_nos:
        return jsonify({"ok": True, "message": "미계산 회차가 없습니다.", "status": _backfill_status(mode)})

    with backfill_lock:
        backfill_jobs[mode].update({
            "running": True,
            "current": 0,
            "total": len(draw_nos),
            "message": f"시작 대기: {draw_nos[0]}~{draw_nos[-1]}회",
            "last_result": None,
        })

    threading.Thread(target=_run_backfill, args=(mode, draw_nos), daemon=True).start()
    return jsonify({"ok": True, "draw_nos": draw_nos, "status": _backfill_status(mode)})

@app.route("/api/evaluate/lotto", methods=["POST"])
@require_auth
def evaluate_lotto():
    data = request.get_json(silent=True) or {}
    nums = data.get("numbers", [])
    try:
        res = lotto_engine.evaluate_custom_set(nums)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/evaluate/pension", methods=["POST"])
@require_auth
def evaluate_pension():
    data = request.get_json(silent=True) or {}
    group = data.get("group")
    digits = data.get("digits", [])
    try:
        res = pension_engine.evaluate_custom_set(group, digits)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/analyze/lotto")
@require_auth
def analyze_lotto():
    df = lotto_engine.df
    num_cols = ["win1", "win2", "win3", "win4", "win5", "win6"]
    
    sums = df[num_cols].sum(axis=1).tolist()
    
    all_nums = df[num_cols].values.flatten()
    freqs = [int((all_nums == n).sum()) for n in range(1, 46)]
    
    odds = df[num_cols].apply(lambda x: int((x % 2 != 0).sum()), axis=1).tolist()
    odd_counts = [odds.count(i) for i in range(7)]
    
    rolling_sums = df[num_cols].sum(axis=1).rolling(50).mean().tail(100).fillna(0).tolist()
    
    # 5. Consecutive Pairs Distribution
    consecs = []
    for _, row in df[num_cols].iterrows():
        nums = sorted(row.values)
        c = sum(1 for i in range(5) if nums[i+1] - nums[i] == 1)
        consecs.append(c)
    consec_counts = [consecs.count(i) for i in range(6)]
    
    # 6. Colors Distribution (1-10 Yellow, 11-20 Blue, etc.)
    colors = [0, 0, 0, 0, 0]
    for n in all_nums:
        idx = (n - 1) // 10
        if idx > 4: idx = 4
        colors[idx] += 1
        
    # 7. High/Low (High >= 23)
    highs = df[num_cols].apply(lambda x: int((x >= 23).sum()), axis=1).tolist()
    high_counts = [highs.count(i) for i in range(7)]
    
    # 8. AC Value (Approx complexity: num unique diffs - 5)
    def calc_ac(nums):
        diffs = set()
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                diffs.add(abs(nums[i] - nums[j]))
        return max(0, len(diffs) - 5)
        
    ac_values = df[num_cols].apply(lambda x: calc_ac(sorted(x.values)), axis=1).tolist()
    ac_counts = [ac_values.count(i) for i in range(11)]

    return jsonify({
        "sums": sums,
        "freqs": freqs,
        "odd_counts": odd_counts,
        "rolling_sums": rolling_sums,
        "draw_nos": df["draw_no"].tolist(),
        "consec_counts": consec_counts,
        "colors": colors,
        "high_counts": high_counts,
        "ac_counts": ac_counts
    })

@app.route("/api/analyze/pension")
@require_auth
def analyze_pension():
    df = pension_engine.df
    digit_cols = ["n1", "n2", "n3", "n4", "n5", "n6"]
    
    # 1. High/Low distribution in Digits (High = 5-9, Low = 0-4)
    highs = df[digit_cols].apply(lambda x: int((x >= 5).sum()), axis=1).tolist()
    high_counts = [highs.count(i) for i in range(7)]
    
    # 2. Overall Digit Frequencies (0-9)
    all_digits = df[digit_cols].values.flatten()
    digit_freqs = [int((all_digits == d).sum()) for d in range(10)]
    
    # 3. Sum of Details (Like Lotto sums, but 0-54 range)
    sums = df[digit_cols].sum(axis=1).tolist()
    
    # 4. Odd/Even Balance in Digits
    odds = df[digit_cols].apply(lambda x: int((x % 2 != 0).sum()), axis=1).tolist()
    odd_counts = [odds.count(i) for i in range(7)]
    
    # 5. Position 1 (Tens of Thousands) specific freq
    pos1_freqs = [int((df["n1"] == d).sum()) for d in range(10)]
    
    # 6. Position 6 (Units) specific freq
    pos6_freqs = [int((df["n6"] == d).sum()) for d in range(10)]
    
    # 7. Consecutive Identical Digits (e.g. 112344 -> 2 pairs)
    identicals = []
    for _, row in df[digit_cols].iterrows():
        nums = row.values
        c = sum(1 for i in range(5) if nums[i+1] == nums[i])
        identicals.append(c)
    identical_counts = [identicals.count(i) for i in range(6)]
    
    # 8. Rolling Sum of D1-D6
    rolling_sums = df[digit_cols].sum(axis=1).rolling(50).mean().tail(100).fillna(0).tolist()
    
    return jsonify({
        "high_counts": high_counts,
        "digit_freqs": digit_freqs,
        "sums": sums,
        "odd_counts": odd_counts,
        "pos1_freqs": pos1_freqs,
        "pos6_freqs": pos6_freqs,
        "identical_counts": identical_counts,
        "rolling_sums": rolling_sums,
        "draw_nos": df["draw_no"].tolist()
    })

if __name__ == "__main__":
    # In production use a real server like waitress or gunicorn
    # For user's laptop server use: host="0.0.0.0"
    debug = os.environ.get("FLASK_DEBUG", "1") != "0"
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_auto_updater()
        warm_models_async()
    else:
        log.info("Skipping AutoUpdater in reloader parent process.")
    app.run(debug=debug, host="0.0.0.0", port=5000)
