import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


class PensionPredictor:
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
        return Path(".model_cache") / f"pension_current_{self._latest_draw_no()}.joblib"

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
        query = 'SELECT draw_no, draw_date, group_no, n1, n2, n3, n4, n5, n6 FROM pension_results WHERE is_bonus=0'
        if self.max_draw_no is not None:
            query += f' AND draw_no <= {self.max_draw_no}'
        query += ' ORDER BY draw_no ASC'
        self.df = pd.read_sql(query, conn)
        conn.close()

        for col in ['group_no', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)

        # Positional Probabilities: pos_freq[i][d] = P(digit d at position i)
        self.pos_freq = []
        for i in range(1, 7):
            counts = np.zeros(10)
            for d in range(10):
                counts[d] = (self.df[f'n{i}'] == d).sum()
            counts += 1  # Laplace smoothing
            self.pos_freq.append(counts / counts.sum())

        # Group frequency (1~5), Laplace-smoothed
        group_counts = np.zeros(5)
        for g in range(1, 6):
            group_counts[g - 1] = (self.df['group_no'] == g).sum()
        group_counts += 1
        self.group_p = group_counts / group_counts.sum()
        self.group_freq = self.df['group_no'].value_counts().sort_index()

        # [FIX #9] Contrarian group: inverse frequency
        inv_group = 1.0 / (group_counts + 1e-6)
        self.group_p_inv = inv_group / inv_group.sum()

    def get_latest_history(self, count=10):
        return self.df.tail(count).iloc[::-1]

    def train_models(self):
        window = 10
        history = self.df[['group_no', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values

        if len(history) <= window + 5:
            self.dl_model = None
            self.ml_model = None
            self._trained = True
            return

        if self._load_model_cache():
            return

        X, y = [], []
        for i in range(len(history) - window):
            X.append(history[i:i+window].flatten().astype(float))
            y.append(history[i + window].astype(int))

        X = np.array(X)
        y = np.array(y)  # shape (N, 7): [group_no, n1..n6]

        base_mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        self.dl_model = MultiOutputClassifier(base_mlp, n_jobs=-1)
        self.dl_model.fit(X, y)

        base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.ml_model = MultiOutputClassifier(base_rf, n_jobs=-1)
        self.ml_model.fit(X, y)
        self._trained = True
        self._save_model_cache()

    def ensure_trained(self):
        if not self._trained:
            self.train_models()

    def _get_X_last(self):
        window = 10
        return self.df[['group_no', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']]\
            .tail(window).values.flatten().reshape(1, -1).astype(float)

    def _calc_score(self, group, digits):
        """[FIX #7] Real score from actual positional probability.
        Average log odds ratio relative to uniform baseline.
        Uniform group P = 0.2, uniform digit P = 0.1.
        """
        g_p = self.group_p[group - 1]
        g_lift = g_p / 0.2  # relative to uniform 1/5
        d_lifts = []
        for i, d in enumerate(digits):
            d_p = self.pos_freq[i][d]
            d_lifts.append(d_p / 0.1)  # relative to uniform 1/10
        avg_lift = np.mean([g_lift] + d_lifts)
        # Map: lift=1.0 → 50, lift=2.0 → 75, lift=0.5 → 25
        score = min(99.9, max(0.1, round((1 - 1 / (1 + avg_lift)) * 100, 1)))
        return score

    def _extract_proba_for_position(self, model, pos_idx):
        """[FIX #8] Extract class probability distribution for one output position.
        Returns a probability array of the possible digit classes.
        pos_idx: 0=group_no, 1..6=n1..n6
        """
        try:
            X_last = self._get_X_last()
            proba_list = model.predict_proba(X_last)
            if pos_idx < len(proba_list):
                p_arr = proba_list[pos_idx][0]  # shape: (n_classes,)
                return p_arr
        except Exception:
            pass
        return None

    def _sample_from_model(self, model, pos_idx, possible_values, fallback_p):
        """[FIX #8] Sample from model's predicted probability distribution
        instead of always taking argmax (which gives identical results every call).
        """
        p_arr = self._extract_proba_for_position(model, pos_idx)
        if p_arr is not None and len(p_arr) == len(possible_values):
            p_arr = np.clip(p_arr, 1e-8, None)
            p_arr = p_arr / p_arr.sum()
            return int(np.random.choice(possible_values, p=p_arr))
        return int(np.random.choice(possible_values, p=fallback_p))

    def _sample_stat(self):
        """Positional probability sampling from all-time stats."""
        g = int(np.random.choice(range(1, 6), p=self.group_p))
        digits = [int(np.random.choice(range(10), p=self.pos_freq[i])) for i in range(6)]
        return g, digits

    def _make_dl_sample(self):
        """[FIX #8] Generate one DL-based set using predict_proba sampling."""
        if self.dl_model is None:
            return self._sample_stat()
        # Position 0: group_no (classes 1..5)
        g = self._sample_from_model(self.dl_model, 0, range(1, 6), self.group_p)
        g = max(1, min(5, g))
        # Positions 1..6: n1..n6 (classes 0..9)
        digits = []
        for pos in range(1, 7):
            d = self._sample_from_model(self.dl_model, pos, range(10), self.pos_freq[pos - 1])
            digits.append(max(0, min(9, d)))
        return g, digits

    def _make_rf_sample(self):
        """[FIX #8] Generate one RF-based set using predict_proba sampling."""
        if self.ml_model is None:
            return self._sample_stat()
        g = self._sample_from_model(self.ml_model, 0, range(1, 6), self.group_p)
        g = max(1, min(5, g))
        digits = []
        for pos in range(1, 7):
            d = self._sample_from_model(self.ml_model, pos, range(10), self.pos_freq[pos - 1])
            digits.append(max(0, min(9, d)))
        return g, digits

    def _make_set(self, group, digits, logic):
        """[FIX #7] Score is now computed from real positional probabilities."""
        score = self._calc_score(group, digits)
        return {"group": group, "digits": digits, "score": score, "logic": logic}

    def predict_all_methods(self):
        """Return 8 statistically distinct prediction methods, each with 5 sets."""
        self.ensure_trained()
        sets = {}

        # Method 1: Deep Learning MLP — predict_proba sampling [FIX #8]
        dl_sets = []
        for _ in range(5):
            g, d = self._make_dl_sample()
            dl_sets.append(self._make_set(g, d, "MLP 확률 분포 샘플링"))
        sets["1. Deep Learning"] = {"desc": "MLP 256-128 신경망 확률 분포 샘플링 (다양성 보장)", "sets": dl_sets}

        # Method 2: Random Forest — predict_proba sampling [FIX #8]
        rf_sets = []
        for _ in range(5):
            g, d = self._make_rf_sample()
            rf_sets.append(self._make_set(g, d, "RF 확률 분포 샘플링"))
        sets["2. Random Forest"] = {"desc": "RF 100-tree 앙상블 확률 분포 샘플링", "sets": rf_sets}

        # Method 3: Monte Carlo — all-time positional frequency
        stat_sets = []
        for _ in range(5):
            g, d = self._sample_stat()
            stat_sets.append(self._make_set(g, d, "전체 이력 위치별 빈도 MC"))
        sets["3. Monte Carlo"] = {"desc": "전체 이력 위치별 빈도 기반 MC 샘플링", "sets": stat_sets}

        # Method 4: Hot Group — fix group to most-frequent, sample digits statistically
        hot_group = int(self.group_freq.idxmax()) if len(self.group_freq) > 0 else 1
        hot_sets = []
        for _ in range(5):
            _, d = self._sample_stat()
            hot_sets.append(self._make_set(hot_group, d, f"핫그룹={hot_group}조 고정"))
        sets["4. Hot Group"] = {"desc": f"최빈 조({hot_group}조) 고정 + 위치 확률 샘플링", "sets": hot_sets}

        # Method 5: Cold Group — fix group to least-frequent, sample digits statistically
        cold_group = int(self.group_freq.idxmin()) if len(self.group_freq) > 0 else 5
        cold_sets = []
        for _ in range(5):
            _, d = self._sample_stat()
            cold_sets.append(self._make_set(cold_group, d, f"콜드그룹={cold_group}조 고정"))
        sets["5. Cold Cycle"] = {"desc": f"미출현 조({cold_group}조) 역주기 전략 (통계 보장없음, 역발상)", "sets": cold_sets}

        # Method 6: DL + Stat Ensemble — blend DL proba with stat sampling
        ens_sets = []
        for _ in range(5):
            g_dl, d_dl = self._make_dl_sample()
            g_st, d_st = self._sample_stat()
            # 60% DL, 40% Stat for each digit
            d_blend = [d_dl[i] if np.random.random() < 0.6 else d_st[i] for i in range(6)]
            g_blend = g_dl if np.random.random() < 0.6 else g_st
            ens_sets.append(self._make_set(g_blend, d_blend, "DL 60% + MC 40% 혼합"))
        sets["6. Ensemble"] = {"desc": "딥러닝 60% + 통계 MC 40% 가중 혼합 전략", "sets": ens_sets}

        # Method 7: Recent Trend — last 50 draws positional frequency
        recent = self.df.tail(50)
        recent_pos = []
        for i in range(1, 7):
            cnt = np.zeros(10)
            for d in range(10):
                cnt[d] = (recent[f'n{i}'] == d).sum()
            cnt += 1
            recent_pos.append(cnt / cnt.sum())
        recent_group_cnt = np.zeros(5)
        for g in range(1, 6):
            recent_group_cnt[g - 1] = (recent['group_no'] == g).sum()
        recent_group_cnt += 1
        recent_group_p = recent_group_cnt / recent_group_cnt.sum()
        trend_sets = []
        for _ in range(5):
            g = int(np.random.choice(range(1, 6), p=recent_group_p))
            d = [int(np.random.choice(range(10), p=recent_pos[i])) for i in range(6)]
            trend_sets.append(self._make_set(g, d, "최근 50회 트렌드"))
        sets["7. Recent Trend"] = {"desc": "최근 50회차 위치별 편향 분석", "sets": trend_sets}

        # Method 8: Contrarian — inverse frequency for both group and digits [FIX #9]
        contra_pos = []
        for i in range(6):
            inv = 1.0 / (self.pos_freq[i] + 1e-6)
            contra_pos.append(inv / inv.sum())
        contra_sets = []
        for _ in range(5):
            # [FIX #9] Use inverse group probability too
            g = int(np.random.choice(range(1, 6), p=self.group_p_inv))
            d = [int(np.random.choice(range(10), p=contra_pos[i])) for i in range(6)]
            contra_sets.append(self._make_set(g, d, "역발상 저빈도 샘플링"))
        sets["8. Contrarian"] = {"desc": "전 위치 저빈도 역발상 전략 (조+자릿수 모두 반전)", "sets": contra_sets}

        return sets

    def evaluate_custom_set(self, group, digits):
        # Prevent NoneType errors with fallbacks
        if digits is None: digits = []
        if group is None: group = 1

        try:
            digits = list(map(int, digits))
            group = int(group)
        except (TypeError, ValueError):
            raise ValueError("입력 데이터의 형식이 올바르지 않습니다.")

        if len(digits) != 6 or any(d < 0 or d > 9 for d in digits):
            raise ValueError("각 자리는 0~9 사이의 숫자여야 합니다")
        if not 1 <= group <= 5:
            raise ValueError("조는 1~5 사이여야 합니다")

        # [FIX #10] Proper score: compare each position's digit probability
        # against uniform baseline (1/10 per digit, 1/5 per group)
        g_p = self.group_p[group - 1]
        g_lift = g_p / 0.2
        d_lifts = []
        detail_parts = [f"{group}조(p={g_p:.3f})"]
        for i, d in enumerate(digits):
            p = self.pos_freq[i][d]
            d_lifts.append(p / 0.1)
            detail_parts.append(f"자리{i+1}={d}(p={p:.3f})")

        avg_lift = np.mean([g_lift] + d_lifts)
        # [FIX #10] Consistent scoring with _calc_score
        score = int(min(100, max(0, (1 - 1 / (1 + avg_lift)) * 100)))

        # Grade thresholds calibrated to uniform=50
        grade = "F"
        if score >= 75:   grade = "S"
        elif score >= 60: grade = "A"
        elif score >= 50: grade = "B"
        elif score >= 40: grade = "C"

        details = " | ".join(detail_parts)
        return {"grade": grade, "score": score, "details": details}

    def get_suggested_set(self, group, digits):
        self.ensure_trained()
        g, d = self._make_dl_sample() if self.dl_model is not None else self._sample_stat()
        return {"group": g, "digits": d}
