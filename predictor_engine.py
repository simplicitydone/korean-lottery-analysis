import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


class LottoPredictor:
    def __init__(self, db_path='lottery.db', max_draw_no=None, train_on_init=True):
        self.db_path = db_path
        self.max_draw_no = max_draw_no
        self.dl_model = None
        self.ml_model = None
        self._trained = False
        self.load_history()
        if train_on_init:
            self.train_models()

    def _latest_draw_no(self):
        return int(self.df['draw_no'].max()) if not self.df.empty else 0

    def _cache_path(self):
        return Path(".model_cache") / f"lotto_current_{self._latest_draw_no()}.joblib"

    def _cache_enabled(self):
        return self.max_draw_no is None and self._latest_draw_no() > 0

    def _load_model_cache(self):
        if not self._cache_enabled():
            return False
        cache_path = self._cache_path()
        if not cache_path.exists():
            return False
        try:
            payload = joblib.load(cache_path)
            if payload.get("latest_draw_no") != self._latest_draw_no():
                return False
            self.dl_model = payload.get("dl_model")
            self.ml_model = payload.get("ml_model")
            self._trained = True
            return True
        except Exception:
            return False

    def _save_model_cache(self):
        if not self._cache_enabled():
            return
        cache_path = self._cache_path()
        cache_path.parent.mkdir(exist_ok=True)
        payload = {
            "latest_draw_no": self._latest_draw_no(),
            "dl_model": self.dl_model,
            "ml_model": self.ml_model,
        }
        joblib.dump(payload, cache_path, compress=3)

    def load_history(self):
        conn = sqlite3.connect(self.db_path)
        for table in ['lotto_results', 'draw_results']:
            try:
                query = f'SELECT draw_no, draw_date, win1, win2, win3, win4, win5, win6, bonus FROM {table}'
                if self.max_draw_no is not None:
                    query += f' WHERE draw_no <= {self.max_draw_no}'
                query += ' ORDER BY draw_no ASC'
                self.df = pd.read_sql(query, conn)
                if len(self.df) > 0:
                    break
            except Exception:
                pass
        conn.close()

        for col in ['win1', 'win2', 'win3', 'win4', 'win5', 'win6']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)

        all_nums = self.df[['win1', 'win2', 'win3', 'win4', 'win5', 'win6']].values.flatten()
        self.freq = pd.Series(np.arange(1, 46)).map(
            pd.Series(all_nums).value_counts()).fillna(0).astype(float)
        self.freq.index = range(1, 46)

        # Cold cycle: draws since last appearance
        max_draw = self.df['draw_no'].max()
        cold = {}
        for i in range(1, 46):
            mask = (self.df[['win1', 'win2', 'win3', 'win4', 'win5', 'win6']] == i).any(axis=1)
            last = self.df[mask]['draw_no'].max() if mask.any() else 0
            cold[i] = max_draw - last
        self.cold_cycle = pd.Series(cold)

        # Uniform baseline for score normalization (1/45)
        self._uniform = 1.0 / 45.0

    def _make_binary_vectors(self):
        draws_bin = []
        for _, row in self.df.iterrows():
            vec = np.zeros(45)
            for i in range(1, 7):
                idx = int(row[f'win{i}']) - 1
                if 0 <= idx < 45:
                    vec[idx] = 1
            draws_bin.append(vec)
        return draws_bin

    def train_models(self):
        window = 12
        draws_bin = self._make_binary_vectors()

        # [FIX #3/#5] Monte Carlo: Laplace-smoothed all-time frequency
        total = self.freq.sum()
        self.p_freq = (self.freq + 1) / (total + 45)

        # [FIX #5] Hot Streak: last 50 draws only (differentiates from MC)
        recent50 = self.df.tail(50)
        r50_nums = recent50[['win1', 'win2', 'win3', 'win4', 'win5', 'win6']].values.flatten()
        r50_freq = pd.Series(r50_nums).value_counts()
        hot_counts = np.zeros(45)
        for num, cnt in r50_freq.items():
            idx = int(num) - 1
            if 0 <= idx < 45:
                hot_counts[idx] = cnt
        self.p_hot = (hot_counts + 1) / (hot_counts.sum() + 45)

        if self._load_model_cache():
            return

        X, y = [], []
        for i in range(len(draws_bin) - window):
            X.append(np.array(draws_bin[i:i+window]).flatten())
            y.append(draws_bin[i+window])

        if len(X) < 10:
            self.dl_model = None
            self.ml_model = None
        else:
            X = np.array(X)
            y = np.array(y).astype(int)

            base_mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            self.dl_model = MultiOutputClassifier(base_mlp, n_jobs=-1)
            self.dl_model.fit(X, y)

            base_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            self.ml_model = MultiOutputClassifier(base_rf, n_jobs=-1)
            self.ml_model.fit(X, y)
        self._trained = True
        self._save_model_cache()

    def ensure_trained(self):
        if not self._trained:
            self.train_models()

    def _extract_proba(self, model, X_input):
        """Safely extract P(=1) per number from MultiOutputClassifier.
        Applies minimum floor to prevent degenerate distributions.
        """
        # [FIX #2] Apply minimum floor of 1/90 to avoid near-zero probabilities
        MIN_P = 1.0 / 90.0
        try:
            proba_list = model.predict_proba(X_input)
            result = []
            for p_arr in proba_list:
                if p_arr.shape[1] == 2:
                    result.append(float(p_arr[0, 1]))
                else:
                    result.append(float(p_arr[0, 0]))
            arr = np.array(result)
            arr = np.maximum(arr, MIN_P)
            return arr / arr.sum()
        except Exception:
            return self.p_freq.values.copy()

    def _get_X_input(self):
        window = 12
        draws_bin = self._make_binary_vectors()
        if len(draws_bin) < window:
            return None
        segment = draws_bin[-window:]
        return np.array(segment).flatten().reshape(1, -1)

    def get_ac(self, nums):
        diffs = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                diffs.add(abs(nums[i] - nums[j]))
        return len(diffs) - 5

    def get_latest_history(self, count=10):
        return self.df.tail(count).iloc[::-1]

    def _calc_score(self, p, nums):
        """[FIX #1] Normalize score relative to uniform baseline (1/45).
        Score > 100 means above-average probability selection.
        Scale: 0 (very cold) → 100 (average) → 200 (very hot).
        We map to 0-100 display range using tanh-like normalization.
        """
        avg_lift = np.mean(p[np.array(nums) - 1]) / self._uniform
        # avg_lift=1.0 → score≈50, avg_lift=2.0 → score≈75, avg_lift=0.5 → score≈25
        score = min(99.9, max(0.1, round((1 - 1 / (1 + avg_lift)) * 100, 1)))
        return score

    def generate_jackpot_set(self, p, count=5):
        """Generate sets matching 1st-prize statistical profile.
        Progressive 3-tier relaxation guarantees 'count' results.
        """
        if p is None or p.sum() == 0:
            p = self.p_freq.values.copy()
        p = np.array(p, dtype=float)
        p = np.clip(p, 1e-8, None)
        p = p / p.sum()

        results = []
        filter_tiers = [
            dict(sum_min=100, sum_max=180, ac_min=7, odds_ok={2, 3, 4}),
            dict(sum_min=90,  sum_max=195, ac_min=6, odds_ok={2, 3, 4}),
            dict(sum_min=80,  sum_max=210, ac_min=5, odds_ok={1, 2, 3, 4, 5}),
        ]
        seen = set()
        for flt in filter_tiers:
            for _ in range(15000):
                if len(results) >= count:
                    break
                nums = sorted(list(map(int,
                    np.random.choice(range(1, 46), size=6, replace=False, p=p))))
                key = tuple(nums)
                if key in seen:
                    continue
                s = sum(nums)
                ac = self.get_ac(nums)
                odds = sum(1 for n in nums if n % 2 != 0)
                if not (flt['sum_min'] <= s <= flt['sum_max']):
                    continue
                if ac < flt['ac_min']:
                    continue
                if odds not in flt['odds_ok']:
                    continue
                # [FIX #1] Real normalized score
                score = self._calc_score(p, nums)
                seen.add(key)
                results.append({
                    "numbers": nums,
                    "score": score,
                    "logic": f"총합:{s} | 복잡도(AC):{ac} | 홀수:{odds}개"
                })
            if len(results) >= count:
                break
        return results[:count]

    def get_all_method_probabilities(self):
        """Return 45-length probability arrays for all 8 methods."""
        self.ensure_trained()
        X_input = self._get_X_input()

        p_dl = self._extract_proba(self.dl_model, X_input) if self.dl_model and X_input is not None else self.p_freq.values.copy()
        p_ml = self._extract_proba(self.ml_model, X_input) if self.ml_model and X_input is not None else self.p_freq.values.copy()
        p_mc = self.p_freq.values.copy()
        
        p_cold = (self.cold_cycle.values.astype(float) + 1)
        p_cold = p_cold / p_cold.sum()

        p_hot = self.p_hot.copy()

        p_inv = 1.0 / (self.freq.values.astype(float) + 1)
        p_inv = p_inv / p_inv.sum()

        recent_freq = pd.Series(self.df.tail(100)[['win1','win2','win3','win4','win5','win6']].values.flatten()).value_counts()
        p_recent = np.zeros(45)
        for num, cnt in recent_freq.items():
            if 0 <= int(num) - 1 < 45: p_recent[int(num)-1] = cnt
        p_recent = (p_recent + 1) / (p_recent.sum() + 45)

        # Dynamic Ensemble Weighting based on last 5 draws accuracy
        weights = {"Deep Learning": 0.5, "Random Forest": 0.3, "Monte Carlo": 0.2}
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT method_results FROM prediction_accuracy_v3 ORDER BY draw_no DESC LIMIT 5")
            rows = cur.fetchall()
            conn.close()
            import json
            hits = {"Deep Learning": 0, "Random Forest": 0, "Monte Carlo": 0}
            if rows:
                for row in rows:
                    if row[0]:
                        m_res = json.loads(row[0])
                        for k, v in hits.items():
                            # Map method names in DB to simplified weight keys
                            found_key = None
                            for db_k in m_res.keys():
                                if k in db_k: found_key = db_k
                            if found_key:
                                # Use hits_1st for lotto, sum of counts
                                hits[k] += sum(m_res[found_key].get("hits_1st", []))
                total_hits = sum(hits.values())
                if total_hits > 0:
                    weights = {k: v / total_hits for k, v in hits.items()}
        except Exception:
            pass # fallback to static weights

        p_ensemble = weights["Deep Learning"] * p_dl + weights["Random Forest"] * p_ml + weights["Monte Carlo"] * p_mc
        p_ensemble = p_ensemble / p_ensemble.sum()

        return {
            "Deep Learning": p_dl,
            "Random Forest": p_ml,
            "Monte Carlo": p_mc,
            "Bayesian Cold": p_cold,
            "DL+ML Ensemble": p_ensemble,
            "Hot Streak": p_hot,
            "Contrarian": p_inv,
            "Recent Trend": p_recent
        }

    def predict_all_methods(self):
        """Return 8 statistically distinct prediction methods, each with 5 sets."""
        probs = self.get_all_method_probabilities()

        return {
            "1. Deep Learning":  {"desc": "MLP 256-128-64, 12-draw 시퀀스 학습 (딥러닝 패턴 인식)",     "sets": self.generate_jackpot_set(probs["Deep Learning"])},
            "2. Random Forest":  {"desc": "RF 200-tree 앙상블, 다회차 상관관계 추적",                    "sets": self.generate_jackpot_set(probs["Random Forest"])},
            "3. Monte Carlo":    {"desc": "전체 이력 Laplace 빈도 기반 MC 샘플링",                      "sets": self.generate_jackpot_set(probs["Monte Carlo"])},
            "4. Bayesian Cold":  {"desc": "미출현 장기 번호 역주기 전략 (통계 보장 없음, 역발상)",        "sets": self.generate_jackpot_set(probs["Bayesian Cold"])},
            "5. DL+ML Ensemble": {"desc": "성능 기반 동적 가중치 융합 앙상블 (최근 5회차 성능 반영)",   "sets": self.generate_jackpot_set(probs["DL+ML Ensemble"])},
            "6. Hot Streak":     {"desc": "최근 50회 핫 번호 선택 (단기 트렌드 추종)",                   "sets": self.generate_jackpot_set(probs["Hot Streak"])},
            "7. Contrarian":     {"desc": "전체 이력 최저 빈도 번호 역발상 선택",                        "sets": self.generate_jackpot_set(probs["Contrarian"])},
            "8. Recent Trend":   {"desc": "최근 100회 빈도 기반 중기 트렌드 분석",                       "sets": self.generate_jackpot_set(probs["Recent Trend"])},
        }

    def evaluate_custom_set(self, nums):
        nums_int = sorted(list(map(int, nums)))
        # [FIX #6] Validate: count, range, AND duplicates
        if len(nums_int) != 6:
            raise ValueError("숫자 6개를 입력하세요")
        if any(n < 1 or n > 45 for n in nums_int):
            raise ValueError("각 번호는 1~45 사이여야 합니다")
        if len(set(nums_int)) != 6:
            raise ValueError("중복 번호가 있습니다. 서로 다른 6개 번호를 입력하세요")

        s = sum(nums_int)
        ac = self.get_ac(nums_int)
        odds = sum(1 for n in nums_int if n % 2 != 0)

        score = 0
        reasons = []
        if 100 <= s <= 180:
            score += 40
            reasons.append(f"합계:{s} [OK](100~180)")
        else:
            reasons.append(f"합계:{s} [X](목표:100~180, 차이:{min(abs(s-100), abs(s-180))})")

        if ac >= 7:
            score += 30
            reasons.append(f"AC:{ac} [OK](>=7)")
        else:
            reasons.append(f"AC:{ac} [X](목표:>=7, 부족:{7-ac})")

        if odds in [2, 3, 4]:
            score += 30
            reasons.append(f"홀짝:{odds}:{6-odds} [OK]")
        else:
            reasons.append(f"홀짝:{odds}:{6-odds} [X](목표:2~4개 홀수)")

        grade = "F"
        if score >= 90:   grade = "S"
        elif score >= 70: grade = "A"
        elif score >= 50: grade = "B"
        elif score >= 30: grade = "C"

        details = " | ".join(reasons)
        return {"grade": grade, "score": score, "details": details}

    def get_suggested_set(self, base_nums):
        self.ensure_trained()
        X_input = self._get_X_input()
        if self.dl_model is not None and X_input is not None:
            p = self._extract_proba(self.dl_model, X_input)
        else:
            p = self.p_freq.values.copy()
        suggestions = self.generate_jackpot_set(p, count=1)
        return suggestions[0] if suggestions else None
