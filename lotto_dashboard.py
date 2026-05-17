import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import threading
from predictor_engine import LottoPredictor
from pension_engine import PensionPredictor
from auto_updater import (LotteryAutoUpdater, RetrospectiveAccuracyEngine,
                          load_accuracy_results, get_uncomputed_draw_nos)


class LotteryStableHub:
    def __init__(self, root):
        self.root = root
        self.root.title("Antigravity Intelligence Hub v11.0")

        # Centralized Font System (compact, readable)
        self.fonts = {
            "header":  ("Segoe UI", 18, "bold"),
            "sub":     ("Segoe UI", 14, "bold"),
            "base":    ("Segoe UI", 10),
            "base_b":  ("Segoe UI", 10, "bold"),
            "small":   ("Segoe UI", 9),
            "mono":    ("Consolas", 10, "bold"),
            "ball":    ("Segoe UI", 15, "bold"),
            "verdict": ("Segoe UI", 42, "bold"),
        }

        self.root.geometry("1400x850")
        self.root.configure(bg="#0f111a")
        self.root.minsize(900, 600)

        self.colors = {
            "bg":      "#0f111a",
            "sidebar": "#14162d",
            "accent":  "#00f2ff",
            "pension": "#00ff88",
            "text":    "#e8eaf6",
            "card":    "#1e212b",
            "gold":    "#ffd700",
            "score":   "#ff4f7b",
            "hot":     "#ffb347",
        }

        self.current_mode = "LOTTO"
        self.engine = None
        self.engines_ready = False
        self.analyze_canvas = None
        self.predict_canvas = None
        self.analyze_rendered_mode = None  # track which mode rendered
        self.accuracy_results = []          # retrospective results (lotto only)
        self.accuracy_computing = False     # guard to avoid double-start
        self.auto_updater = None

        self.setup_styles()
        self.create_initial_layout()

        threading.Thread(target=self.boot_engines, daemon=True).start()

    # ─────────────────────────────── Setup ───────────────────────────────

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",    background=self.colors["bg"])
        style.configure("Sidebar.TFrame", background=self.colors["sidebar"])
        style.configure("Card.TFrame",    background=self.colors["card"], relief="flat")
        style.configure("TLabel",  background=self.colors["bg"], foreground=self.colors["text"],
                        font=self.fonts["base"])
        style.configure("Header.TLabel",  font=self.fonts["header"], foreground=self.colors["accent"])
        style.configure("TNotebook",      background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",  background=self.colors["sidebar"], foreground="white",
                        font=self.fonts["small"], padding=[6, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", self.colors["accent"])],
                  foreground=[("selected", "black")])
        style.configure("TSeparator", background="#3a3d4f")

    def create_initial_layout(self):
        # Header
        self.header_frame = ttk.Frame(self.root, height=120)
        self.header_frame.pack(side="top", fill="x", padx=20, pady=10)
        self.header_frame.pack_propagate(False)

        # Sidebar
        self.sidebar = ttk.Frame(self.root, width=300, style="Sidebar.TFrame")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        ttk.Label(self.sidebar, text="STRATEGIC HUB", font=self.fonts["header"],
                  background=self.colors["sidebar"], foreground=self.colors["accent"]).pack(pady=20)

        # Main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        self.loading_label = tk.Label(
            self.main_container,
            text="⏳  BIG DATA ENGINES INITIALIZING...\nMLP 256-128-64 training on 1,200+ draws\n(30-60 seconds)",
            font=self.fonts["sub"], bg=self.colors["bg"], fg=self.colors["accent"],
            justify="center")
        self.loading_label.place(relx=0.5, rely=0.45, anchor="center")

        self.pages = {}
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

    # ─────────────────────────── Engine Boot ──────────────────────────────

    def boot_engines(self):
        try:
            self.lotto_engine = LottoPredictor(db_path="lottery.db")
            self.pension_engine = PensionPredictor(db_path="lottery.db")
            self.engine = self.lotto_engine
            self.engines_ready = True
            self.root.after(0, self.on_engines_ready)
        except Exception as e:
            self.root.after(0, lambda err=e: messagebox.showerror("Engine Error", str(err)))

    def on_engines_ready(self):
        self.loading_label.destroy()
        self.create_ready_widgets()
        # Start Auto-Updater in background
        self.auto_updater = LotteryAutoUpdater(
            db_path="lottery.db",
            on_lotto_update=self._on_new_lotto_draw,
            on_pension_update=self._on_new_pension_draw
        )
        self.auto_updater.start()
        self.refresh_mode()

    def _on_new_lotto_draw(self, data):
        """Called by AutoUpdater when a new lotto draw is fetched."""
        self.root.after(0, lambda d=data: self._handle_new_lotto(d))

    def _handle_new_lotto(self, data):
        # Reload lotto engine data and refresh header
        try:
            self.lotto_engine.load_history()
            self.update_header()
            # Invalidate accuracy cache
            self.accuracy_results = []
            messagebox.showinfo(
                "신규 당첨 업데이트",
                f"로또 {data['draw_no']}회 ({data['draw_date']}) 자동 업데이트 완료!\n"
                f"{data['win1']}-{data['win2']}-{data['win3']}-"
                f"{data['win4']}-{data['win5']}-{data['win6']} (보너스:{data['bonus']})"
            )
        except Exception as e:
            pass

    def _on_new_pension_draw(self, data):
        self.root.after(0, lambda d=data: self._handle_new_pension(d))

    def _handle_new_pension(self, data):
        try:
            self.pension_engine.load_history()
            self.update_header()
            messagebox.showinfo(
                "신규 당첨 업데이트",
                f"연금복권 {data['draw_no']}회 ({data['draw_date']}) 자동 업데이트 완료!\n"
                f"{data['group_no']}조 "
                f"{data['n1']}{data['n2']}{data['n3']}{data['n4']}{data['n5']}{data['n6']}"
            )
        except Exception as e:
            pass

    def create_ready_widgets(self):
        self.mode_btn = tk.Button(
            self.sidebar, text="MODE: LOTTO 6/45", command=self.toggle_mode,
            font=self.fonts["sub"], bg=self.colors["accent"], fg="black", bd=0, pady=12)
        self.mode_btn.pack(fill="x", padx=20, pady=12)
        ttk.Separator(self.sidebar).pack(fill="x", padx=10)

        menu_items = [
            ("[AI]  AI PREDICTOR",    "predict"),
            ("[Chart]  BIG DATA TRENDS", "analyze"),
            ("[Lab]  CUSTOM EVALUATOR","lab"),
            ("[Accuracy]  ACCURACY LAB",  "accuracy"),
        ]
        for label, page_name in menu_items:
            btn = tk.Button(
                self.sidebar, text=label,
                command=lambda p=page_name: self.show_page(p),
                font=self.fonts["base_b"], bg=self.colors["sidebar"], fg=self.colors["text"],
                bd=0, pady=10, activebackground=self.colors["card"],
                activeforeground=self.colors["accent"])
            btn.pack(fill="x", padx=20, pady=2)

        self.pages["predict"]  = ttk.Frame(self.main_container)
        self.pages["analyze"]  = ttk.Frame(self.main_container)
        self.pages["lab"]      = ttk.Frame(self.main_container)
        self.pages["accuracy"] = ttk.Frame(self.main_container)

    # ─────────────────────────── Mode Control ─────────────────────────────

    def toggle_mode(self):
        self.current_mode = "PENSION" if self.current_mode == "LOTTO" else "LOTTO"
        self.engine = self.lotto_engine if self.current_mode == "LOTTO" else self.pension_engine
        self.analyze_rendered_mode = None  # force re-render on mode switch
        self.refresh_mode()

    def refresh_mode(self):
        color = self.colors["accent"] if self.current_mode == "LOTTO" else self.colors["pension"]
        self.mode_btn.config(text=f"MODE: {self.current_mode}", bg=color)
        self.update_header()
        self.refresh_lab_page()
        self.show_page("predict")
        self.refresh_predict_page()

    def show_page(self, name):
        if not self.engines_ready:
            return
        for p in self.pages.values():
            p.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        # Render analyze on first visit or when mode changed
        if name == "analyze" and self.analyze_rendered_mode != self.current_mode:
            self.refresh_analyze_page()
        # Render accuracy lab (lotto only)
        if name == "accuracy" and self.current_mode == "LOTTO":
            self.show_accuracy_page()

    # ─────────────────────────── Header ───────────────────────────────────

    def update_header(self):
        for w in self.header_frame.winfo_children():
            w.destroy()
        if not self.engines_ready:
            return
        history = self.engine.get_latest_history(4)
        color = self.colors["accent"] if self.current_mode == "LOTTO" else self.colors["pension"]
        ttk.Label(self.header_frame, text=f"{self.current_mode} FEED",
                  font=self.fonts["header"], foreground=color).pack(side="left", padx=15)
        for _, row in history.iterrows():
            f = ttk.Frame(self.header_frame, style="Card.TFrame", padding=8)
            f.pack(side="left", padx=8)
            dt = str(row["draw_date"])[:10]
            if self.current_mode == "LOTTO":
                val = "-".join(str(int(row[f"win{i}"])) for i in range(1, 7))
            else:
                val = f"{int(row['group_no'])}조 " + "".join(str(int(row[f"n{i}"])) for i in range(1, 7))
            ttk.Label(f, text=f"Dr.{int(row['draw_no'])} ({dt})",
                      font=self.fonts["small"], background=self.colors["card"],
                      foreground="#aaaaaa").pack()
            ttk.Label(f, text=val, font=self.fonts["sub"],
                      background=self.colors["card"], foreground=color).pack()

    # ─────────────────────── AI Predictor Page ────────────────────────────

    def refresh_predict_page(self):
        page = self.pages["predict"]
        for w in page.winfo_children():
            w.destroy()

        res = self.engine.predict_all_methods()

        # ── Ensemble Best 5 ──
        all_sets = []
        for m_data in res.values():
            all_sets.extend(m_data["sets"])
        best5 = sorted(all_sets, key=lambda x: x["score"], reverse=True)[:5]

        best_f = ttk.Frame(page, style="Card.TFrame", padding=12)
        best_f.pack(fill="x", padx=15, pady=8)
        ttk.Label(best_f, text="🏆  ENSEMBLE BEST 5  —  STATISTICAL CONSENSUS",
                  font=self.fonts["sub"], foreground=self.colors["gold"],
                  background=self.colors["card"]).pack(anchor="w", pady=4)

        for s_data in best5:
            row_f = ttk.Frame(best_f, style="Card.TFrame")
            row_f.pack(anchor="w", pady=3)
            self._draw_balls(row_f, s_data, accent=self.colors["gold"])
            ttk.Label(row_f, text=f"  [{s_data['score']:.1f}%] {s_data.get('logic','')}",
                      font=self.fonts["small"], foreground="#aaaaaa",
                      background=self.colors["card"]).pack(side="left", padx=5)

        ttk.Separator(page).pack(fill="x", padx=15, pady=6)

        # ── Expert Tabs (8 methods) ──
        nb = ttk.Notebook(page)
        nb.pack(fill="both", expand=True, padx=15, pady=5)
        self.notebook = nb

        for method_name, data in res.items():
            tab_frame = ttk.Frame(nb)
            nb.add(tab_frame, text=method_name)

            c = tk.Canvas(tab_frame, bg=self.colors["bg"], highlightthickness=0)
            sc = ttk.Scrollbar(tab_frame, orient="vertical", command=c.yview)
            inner = ttk.Frame(c)
            inner.bind("<Configure>",
                       lambda e, canvas=c: canvas.configure(scrollregion=canvas.bbox("all")))
            c.create_window((0, 0), window=inner, anchor="nw")
            c.configure(yscrollcommand=sc.set)
            c.pack(side="left", fill="both", expand=True)
            sc.pack(side="right", fill="y")
            self.predict_canvas = c

            ttk.Label(inner, text=f"전략: {method_name}",
                      font=self.fonts["sub"], foreground=self.colors["accent"]).pack(
                pady=8, padx=15, anchor="w")
            ttk.Label(inner, text=data["desc"],
                      font=self.fonts["small"], foreground="#8888aa").pack(
                padx=20, anchor="w")

            for idx, s_data in enumerate(data["sets"], 1):
                card = ttk.Frame(inner, style="Card.TFrame", padding=10)
                card.pack(fill="x", pady=6, padx=15)
                ttk.Label(card, text=f"#{idx}", font=self.fonts["mono"],
                          foreground=self.colors["accent"],
                          background=self.colors["card"]).pack(side="left", padx=5)
                self._draw_balls(card, s_data)
                ttk.Label(card, text=f"Score {s_data['score']:.1f}%",
                          font=self.fonts["base_b"], foreground=self.colors["hot"],
                          background=self.colors["card"]).pack(side="right", padx=10)

    def _draw_balls(self, parent, s_data, accent=None):
        if accent is None:
            accent = self.colors["accent"] if self.current_mode == "LOTTO" else self.colors["pension"]
        if self.current_mode == "LOTTO":
            for n in s_data.get("numbers", []):
                tk.Label(parent, text=str(n), width=3, bg=accent, fg="#0f111a",
                         font=self.fonts["ball"], relief="raised", bd=2).pack(side="left", padx=3)
        else:
            tk.Label(parent, text=f"{s_data.get('group','-')}조",
                     width=4, bg="#ff5555", fg="white",
                     font=self.fonts["ball"], relief="raised", bd=2).pack(side="left", padx=3)
            for n in s_data.get("digits", []):
                tk.Label(parent, text=str(n), width=2, bg=accent, fg="#0f111a",
                         font=self.fonts["ball"], relief="raised", bd=2).pack(side="left", padx=2)

    # ──────────────────── Big Data Analysis Page ──────────────────────────

    def refresh_analyze_page(self):
        page = self.pages["analyze"]
        for w in page.winfo_children():
            w.destroy()
        self.analyze_rendered_mode = self.current_mode

        c = tk.Canvas(page, bg=self.colors["bg"], highlightthickness=0)
        sc = ttk.Scrollbar(page, orient="vertical", command=c.yview)
        grid_frame = ttk.Frame(c)
        grid_frame.bind("<Configure>",
                        lambda e, canvas=c: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = c.create_window((0, 0), window=grid_frame, anchor="nw")
        page.bind("<Configure>",
                  lambda e, canvas=c, wid=win_id: canvas.itemconfig(wid, width=e.width))
        c.configure(yscrollcommand=sc.set)
        c.pack(side="left", fill="both", expand=True)
        sc.pack(side="right", fill="y")
        self.analyze_canvas = c

        self.draw_big_data_charts(grid_frame)

    def draw_big_data_charts(self, parent):
        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.dpi":       100,
            "axes.facecolor":   "#12141d",
            "figure.facecolor": self.colors["bg"],
            "text.color":       "white",
            "axes.labelcolor":  "white",
            "xtick.color":      "#aaaaaa",
            "ytick.color":      "#aaaaaa",
            "axes.edgecolor":   "#3a3d4f",
            "grid.color":       "#2a2d3a",
            "grid.alpha":       0.4,
            # Korean font fix (Malgun Gothic is always present on Windows)
            "font.family":      "Malgun Gothic",
            "axes.unicode_minus": False,
        })

        data = self.engine.df
        accent = self.colors["accent"] if self.current_mode == "LOTTO" else self.colors["pension"]
        title_color = accent
        fig, axes = plt.subplots(8, 1, figsize=(13, 64), facecolor=self.colors["bg"])
        fig.tight_layout(pad=10.0)

        if self.current_mode == "LOTTO":
            num_cols = ["win1", "win2", "win3", "win4", "win5", "win6"]
            sums = data[num_cols].sum(axis=1)

            # Panel 1: Sum Distribution (Histogram)
            ax = axes[0]
            ax.hist(sums, bins=40, color=accent, alpha=0.85, edgecolor="#0f111a")
            ax.axvspan(100, 180, alpha=0.12, color="yellow", label="1등 황금 구간(100~180)")
            ax.axvline(sums.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"평균={sums.mean():.1f}")
            ax.set_title("① 전체 합계(Sum) 분포 — Big Data Sum Density", color=title_color, fontsize=15)
            ax.set_xlabel("합계"); ax.set_ylabel("빈도수")
            ax.legend(fontsize=9); ax.grid(True)

            # Panel 2: 50-draw Rolling Average Trend
            ax = axes[1]
            roll_mean = sums.rolling(50).mean()
            ax.plot(data["draw_no"], sums, "o", markersize=1.5, alpha=0.2, color=accent)
            ax.plot(data["draw_no"], roll_mean, color="#ff79c6", linewidth=2, label="50회 이동평균")
            ax.set_title("② 합계 트렌드 변동성 — 50-Draw Rolling Mean", color=title_color, fontsize=15)
            ax.set_xlabel("회차"); ax.set_ylabel("합계")
            ax.legend(fontsize=9); ax.grid(True)

            # Panel 3: Number Frequency Heatmap (bar chart)
            ax = axes[2]
            freq_series = pd.Series(data[num_cols].values.flatten()).value_counts().sort_index()
            colors_bar = [self.colors["score"] if freq_series.get(n, 0) >= freq_series.quantile(0.75)
                          else self.colors["hot"] if freq_series.get(n, 0) >= freq_series.median()
                          else "#555577" for n in range(1, 46)]
            ax.bar(range(1, 46), [freq_series.get(n, 0) for n in range(1, 46)],
                   color=colors_bar, edgecolor="#0f111a")
            ax.set_title("③ 번호별 출현 빈도 히트맵 — Number Frequency", color=title_color, fontsize=15)
            ax.set_xlabel("번호 (1~45)"); ax.set_ylabel("출현 횟수")
            ax.set_xticks(range(1, 46))
            ax.grid(True, axis="y")

            # Panel 4: Odd/Even Distribution
            ax = axes[3]
            odds = data[num_cols].apply(lambda x: int((x % 2 != 0).sum()), axis=1)
            counts = [int((odds == i).sum()) for i in range(7)]
            bars = ax.bar(range(7), counts,
                          color=[accent if i in [2, 3, 4] else "#555577" for i in range(7)],
                          edgecolor="#0f111a")
            ax.set_title("④ 홀수 개수 분포 (1등 최적: 2~4개) — Odd Count Distribution",
                         color=title_color, fontsize=15)
            ax.set_xlabel("홀수 개수"); ax.set_ylabel("해당 회차 수")
            ax.set_xticks(range(7))
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                            str(cnt), ha="center", va="bottom", fontsize=8, color="white")
            ax.grid(True, axis="y")

            # Panel 5: AC (Arithmetic Complexity) Distribution
            ax = axes[4]
            ac_vals = []
            for _, row in data.iterrows():
                nums = sorted([int(row[c]) for c in num_cols])
                diffs = set()
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        diffs.add(abs(nums[i] - nums[j]))
                ac_vals.append(len(diffs) - 5)
            ac_series = pd.Series(ac_vals)
            ax.hist(ac_series, bins=range(0, 16), color=accent, alpha=0.85, edgecolor="#0f111a")
            ax.axvline(7, color="red", linestyle="--", linewidth=2, label="1등 기준선 AC=7")
            ax.set_title("⑤ AC값 분포 (1등 목표: AC≥7) — Arithmetic Complexity",
                         color=title_color, fontsize=15)
            ax.set_xlabel("AC 값"); ax.set_ylabel("빈도")
            ax.legend(fontsize=9); ax.grid(True)

            # Panel 6: Consecutive Number Analysis
            ax = axes[5]
            consec_counts = []
            for _, row in data.iterrows():
                nums = sorted([int(row[c]) for c in num_cols])
                consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
                consec_counts.append(consec)
            consec_s = pd.Series(consec_counts)
            cnt_vals = [int((consec_s == i).sum()) for i in range(6)]
            ax.bar(range(6), cnt_vals, color=accent, edgecolor="#0f111a", alpha=0.85)
            ax.set_title("⑥ 연속 번호 쌍 개수 분포 — Consecutive Number Analysis",
                         color=title_color, fontsize=15)
            ax.set_xlabel("연속 번호 쌍 수"); ax.set_ylabel("해당 회차 수")
            ax.grid(True, axis="y")

            # Panel 7: Last-digit (끝자리) Distribution
            ax = axes[6]
            last_digits = data[num_cols].apply(lambda x: x % 10, axis=1)
            all_ld = last_digits.values.flatten()
            ld_counts = pd.Series(all_ld).value_counts().sort_index()
            ax.bar(ld_counts.index, ld_counts.values, color=accent, edgecolor="#0f111a", alpha=0.85)
            ax.set_title("⑦ 끝자리 분포 (편향 분석) — Last Digit Distribution",
                         color=title_color, fontsize=15)
            ax.set_xlabel("끝자리 (0~9)"); ax.set_ylabel("출현 횟수")
            ax.set_xticks(range(10)); ax.grid(True, axis="y")

            # Panel 8: Sum × AC Scatter (Jackpot Zone Visualization)
            ax = axes[7]
            ax.scatter(ac_series, sums, alpha=0.15, s=8, color=accent)
            ax.axvspan(7, 15, alpha=0.15, color="yellow")
            ax.axhspan(100, 180, alpha=0.12, color="lime")
            ax.set_title("⑧ 1등 잭팟 존 산점도 (AC≥7 & Sum 100~180) — Jackpot Zone",
                         color=title_color, fontsize=15)
            ax.set_xlabel("AC 값"); ax.set_ylabel("합계(Sum)")
            ax.grid(True)

        else:
            # PENSION MODE — 8 panels
            grps = [int(g) for g in data["group_no"]]

            # Panel 1: Group Distribution
            ax = axes[0]
            ax.hist(grps, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color=accent,
                    alpha=0.85, edgecolor="#0f111a", rwidth=0.8)
            ax.set_title("① 조(Group) 출현 빈도 분포", color=title_color, fontsize=15)
            ax.set_xlabel("조"); ax.set_ylabel("당첨 횟수")
            ax.set_xticks([1, 2, 3, 4, 5]); ax.grid(True, axis="y")

            # Panels 2~7: Per-position digit distribution
            for pos_i in range(6):
                ax = axes[pos_i + 1]
                col = f"n{pos_i+1}"
                digit_cnt = [int((data[col] == d).sum()) for d in range(10)]
                ax.bar(range(10), digit_cnt, color=accent, edgecolor="#0f111a", alpha=0.85)
                ax.set_title(f"② 자리 {pos_i+1} — 숫자별 출현 빈도 (Position {pos_i+1} Frequency)",
                             color=title_color, fontsize=14)
                ax.set_xlabel("숫자 (0~9)"); ax.set_ylabel("출현 횟수")
                ax.set_xticks(range(10)); ax.grid(True, axis="y")

            # Panel 8: Rolling group trend
            ax = axes[7]
            ax.plot(data["draw_no"], grps, "o", markersize=2, alpha=0.3, color=accent)
            roll = pd.Series(grps).rolling(20).mean()
            ax.plot(data["draw_no"], roll.values, color="#ff79c6", linewidth=2,
                    label="20회 이동평균")
            ax.set_title("⑧ 조 번호 추이 트렌드 — Group Trend", color=title_color, fontsize=15)
            ax.set_xlabel("회차"); ax.set_ylabel("조")
            ax.legend(); ax.grid(True)

        fig.subplots_adjust(hspace=0.5)

        canvas_widget = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ─────────────────────── Custom Evaluator ─────────────────────────────

    def refresh_lab_page(self):
        page = self.pages.get("lab")
        if page is None:
            return
        for w in page.winfo_children():
            w.destroy()

        color = self.colors["accent"] if self.current_mode == "LOTTO" else self.colors["pension"]
        ttk.Label(page, text="🔬  JACKPOT CUSTOM EVALUATOR",
                  font=self.fonts["header"], foreground=color).pack(pady=15)

        # Input area
        input_f = ttk.Frame(page, style="Card.TFrame", padding=20)
        input_f.pack(fill="x", padx=30, pady=5)

        if self.current_mode == "LOTTO":
            ttk.Label(input_f, text="번호 6개 입력 (1~45):",
                      font=self.fonts["base_b"], background=self.colors["card"]).pack(anchor="w")
            entry_f = ttk.Frame(input_f, style="Card.TFrame")
            entry_f.pack(pady=8)
            self.custom_entries = []
            for _ in range(6):
                e = tk.Entry(entry_f, width=4, font=self.fonts["ball"],
                             justify="center", bg="#2a2d3e", fg="white",
                             insertbackground="white")
                e.pack(side="left", padx=4)
                self.custom_entries.append(e)
        else:
            ttk.Label(input_f, text="조(1~5) + 번호 6자리 (0~9) 입력:",
                      font=self.fonts["base_b"], background=self.colors["card"]).pack(anchor="w")
            entry_f = ttk.Frame(input_f, style="Card.TFrame")
            entry_f.pack(pady=8)
            self.custom_entries = []
            ge = tk.Entry(entry_f, width=3, font=self.fonts["ball"],
                          justify="center", bg="#ff5555", fg="white")
            ge.pack(side="left", padx=6)
            tk.Label(entry_f, text="조", font=self.fonts["base"],
                     bg=self.colors["card"], fg="white").pack(side="left")
            self.custom_entries.append(ge)
            for _ in range(6):
                e = tk.Entry(entry_f, width=3, font=self.fonts["ball"],
                             justify="center", bg="#2a2d3e", fg="white")
                e.pack(side="left", padx=3)
                self.custom_entries.append(e)

        tk.Button(input_f, text="🎯  AUDIT JACKPOT PROFILE",
                  command=self.perform_audit,
                  font=self.fonts["base_b"], bg=color, fg="black", bd=0, pady=10,
                  activebackground=self.colors["gold"]).pack(pady=12)

        self.verdict_frame = ttk.Frame(page)
        self.verdict_frame.pack(fill="both", expand=True, padx=30)

    def perform_audit(self):
        try:
            vals = [e.get().strip() for e in self.custom_entries]
            if any(v == "" for v in vals):
                raise ValueError("모든 칸을 채워 주세요.")

            if self.current_mode == "LOTTO":
                res = self.engine.evaluate_custom_set(vals)
                suggested = self.engine.get_suggested_set(vals)
            else:
                res = self.engine.evaluate_custom_set(vals[0], vals[1:])
                suggested = self.engine.get_suggested_set(vals[0], vals[1:])

            for w in self.verdict_frame.winfo_children():
                w.destroy()

            color = self.colors["accent"] if self.current_mode == "LOTTO" else self.colors["pension"]
            grade_color = {"S": "#00ff88", "A": "#00f2ff", "B": "#ffd700",
                           "C": "#ffb347", "F": "#ff4f7b"}.get(res["grade"], "white")

            verdict_card = ttk.Frame(self.verdict_frame, style="Card.TFrame", padding=20)
            verdict_card.pack(fill="x", pady=10)

            tk.Label(verdict_card, text=res["grade"], font=self.fonts["verdict"],
                     fg=grade_color, bg=self.colors["card"]).pack(side="left", padx=20)

            right_f = ttk.Frame(verdict_card, style="Card.TFrame")
            right_f.pack(side="left", fill="both", expand=True)
            ttk.Label(right_f, text=f"Jackpot Score: {res['score']}",
                      font=self.fonts["sub"], foreground=grade_color,
                      background=self.colors["card"]).pack(anchor="w")
            ttk.Label(right_f, text=res["details"], font=self.fonts["base"],
                      foreground="#aaaaaa", background=self.colors["card"],
                      wraplength=700).pack(anchor="w", pady=5)

            if suggested:
                sugg_f = ttk.Frame(self.verdict_frame, style="Card.TFrame", padding=15)
                sugg_f.pack(fill="x", pady=5)
                ttk.Label(sugg_f, text="🚀  AI 최적화 제안 (Auto-Correction)",
                          font=self.fonts["sub"], foreground=self.colors["pension"],
                          background=self.colors["card"]).pack(anchor="w", pady=5)
                ball_row = ttk.Frame(sugg_f, style="Card.TFrame")
                ball_row.pack(anchor="w", pady=5)
                self._draw_balls(ball_row, suggested, accent=self.colors["pension"])

        except Exception as err:
            messagebox.showerror("입력 오류", str(err))

    # ─────────────────────── Mouse Wheel Scroll ───────────────────────────

    def _on_mousewheel(self, event):
        widget = event.widget
        # Walk up widget hierarchy to find a canvas
        for canvas in [self.analyze_canvas, self.predict_canvas]:
            if canvas and self._widget_in_canvas(widget, canvas):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return
        # Fallback: scroll whichever is visible
        for canvas in [self.analyze_canvas, self.predict_canvas]:
            if canvas:
                try:
                    if canvas.winfo_viewable():
                        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                        return
                except Exception:
                    pass

    def _widget_in_canvas(self, widget, canvas):
        try:
            w = widget
            while w:
                if w == canvas:
                    return True
                w = w.master
        except Exception:
            pass
        return False

    # ──────────────────── Accuracy Lab Page (DB-backed) ───────────────────

    # START_DRAW: 3월부터 누적 시작 (2026-03-07 = 1214회)
    ACCURACY_START_DRAW = 1214

    def show_accuracy_page(self):
        """Load persisted accuracy results from DB.
        If there are uncomputed draws, trigger background computation.
        """
        page = self.pages["accuracy"]
        for w in page.winfo_children():
            w.destroy()

        # Fast path: load from DB
        stored = load_accuracy_results("lottery.db")
        uncomputed = get_uncomputed_draw_nos("lottery.db", self.ACCURACY_START_DRAW)

        # Render whatever we have immediately
        self._build_accuracy_header(page, len(stored), len(uncomputed))
        if stored:
            self._render_accuracy_results(page, stored)

        # If there are pending draws, show progress bar and start background compute
        if uncomputed and not self.accuracy_computing:
            self._start_retro_compute(page, uncomputed, stored)
        elif uncomputed and self.accuracy_computing:
            tk.Label(page,
                     text=f"백그라운드 DL 계산 중... ({len(uncomputed)}회 잔여)",
                     font=self.fonts["small"], bg=self.colors["bg"],
                     fg=self.colors["accent"]).pack(pady=4)
        elif not stored and not uncomputed:
            tk.Label(page, text="3월 이후 회차 데이터가 없습니다.",
                     font=self.fonts["base"], bg=self.colors["bg"],
                     fg="white").pack(pady=20)

    def _build_accuracy_header(self, page, computed: int, pending: int):
        hf = ttk.Frame(page, style="Card.TFrame", padding=15)
        hf.pack(fill="x", padx=20, pady=8)
        ttk.Label(hf,
                  text="DL Retrospective Accuracy Archive — 예측 신뢰성 누적 기록",
                  font=self.fonts["sub"], foreground=self.colors["gold"],
                  background=self.colors["card"]).pack(anchor="w")
        ttk.Label(hf,
                  text=(f"MLP 256-128-64 | 해당 회차 이전 데이터로만 학습 | "
                        f"영구 저장 (조작 불가) | "
                        f"계산 완료: {computed}회 | 대기: {pending}회"),
                  font=self.fonts["small"], foreground="#aaaaaa",
                  background=self.colors["card"]).pack(anchor="w")

    def _start_retro_compute(self, page, uncomputed: list, already_stored: list):
        self.accuracy_computing = True
        self.acc_progress_var = tk.StringVar(value=f"DL 재학습 준비 중... ({len(uncomputed)}회 대기)")
        self.acc_progress_label = tk.Label(
            page, textvariable=self.acc_progress_var,
            font=self.fonts["small"], bg=self.colors["bg"], fg=self.colors["accent"])
        self.acc_progress_label.pack(pady=4)
        self.acc_result_frame = page

        engine = RetrospectiveAccuracyEngine(
            db_path="lottery.db", mode="accurate",
            progress_callback=self._on_retro_progress
        )
        t = threading.Thread(
            target=self._retro_thread,
            args=(engine, uncomputed),
            daemon=True
        )
        t.start()

    def _retro_thread(self, engine, draw_nos):
        new_results = engine.compute(draw_nos_to_compute=draw_nos)
        self.root.after(0, lambda r=new_results: self._on_retro_complete(r))

    def _on_retro_progress(self, current, total, msg):
        def _upd():
            try:
                self.acc_progress_var.set(f"{msg}  ({current}/{total})")
            except Exception:
                pass
        self.root.after(0, _upd)

    def _on_retro_complete(self, new_results):
        self.accuracy_computing = False
        try:
            self.acc_progress_label.destroy()
        except Exception:
            pass
        if new_results:
            # Reload all from DB and re-render the full page
            self.show_accuracy_page()

    def _render_accuracy_results(self, page, results):
        if not results:
            return

        avg_hit10 = np.mean([r["hit_top10"] for r in results])
        avg_hit20 = np.mean([r["hit_top20"] for r in results])
        avg_rank  = np.mean([r["avg_rank"]  for r in results])

        stat_f = ttk.Frame(page, style="Card.TFrame", padding=10)
        stat_f.pack(fill="x", padx=20, pady=4)
        ttk.Label(stat_f,
                  text=(f"[누적 통계]  "
                        f"평균 Hit@Top-10: {avg_hit10:.2f}/6  |  "
                        f"평균 Hit@Top-20: {avg_hit20:.2f}/6  |  "
                        f"평균 볼 순위: {avg_rank:.1f}/45  |  "
                        f"총 {len(results)}회 기록"),
                  font=self.fonts["mono"], foreground=self.colors["gold"],
                  background=self.colors["card"]).pack(anchor="w")

        # Scrollable table
        c = tk.Canvas(page, bg=self.colors["bg"], highlightthickness=0)
        sc = ttk.Scrollbar(page, orient="vertical", command=c.yview)
        inner = ttk.Frame(c)
        inner.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        c.create_window((0, 0), window=inner, anchor="nw")
        c.configure(yscrollcommand=sc.set)
        c.pack(side="left", fill="both", expand=True)
        sc.pack(side="right", fill="y")
        self.accuracy_canvas = c

        HIT_COLORS = {0: "#555577", 1: "#994400", 2: "#cc6600",
                      3: "#ffaa00", 4: "#ccff00", 5: "#00ff88", 6: "#00f2ff"}

        # Header
        hdr = ttk.Frame(inner, style="Card.TFrame", padding=5)
        hdr.pack(fill="x", pady=2, padx=8)
        for col_text, col_w in [
            ("Draw", 7), ("Date", 10), ("Actual", 40), ("Top-6 Pred", 40),
            ("@6", 5), ("@10", 5), ("@20", 5), ("AvgRk", 6), ("Train", 6), ("Computed", 18)
        ]:
            tk.Label(hdr, text=col_text, width=col_w,
                     bg=self.colors["card"], fg=self.colors["accent"],
                     font=self.fonts["base_b"]).pack(side="left", padx=2)

        for r in results:
            row_f = ttk.Frame(inner, style="Card.TFrame", padding=5)
            row_f.pack(fill="x", pady=2, padx=8)

            # Draw + date
            info_f = ttk.Frame(row_f, style="Card.TFrame")
            info_f.pack(side="left")
            tk.Label(info_f, text=f"#{r['draw_no']}", width=7,
                     bg=self.colors["card"], fg=self.colors["gold"],
                     font=self.fonts["base_b"]).pack()
            tk.Label(info_f, text=r["draw_date"][:10], width=10,
                     bg=self.colors["card"], fg="#777788",
                     font=self.fonts["small"]).pack()

            # Actual numbers
            actual_set = set(r["actual"])
            pred6_set  = set(r["predicted6"])
            act_f = ttk.Frame(row_f, style="Card.TFrame")
            act_f.pack(side="left", padx=6)
            for n in r["actual"]:
                hit = n in pred6_set
                tk.Label(act_f, text=str(n), width=3,
                         bg=self.colors["pension"] if hit else "#2a2d40",
                         fg="black" if hit else "white",
                         font=self.fonts["base_b"],
                         relief="raised", bd=1).pack(side="left", padx=1)

            # Top-6 predicted
            pred_f = ttk.Frame(row_f, style="Card.TFrame")
            pred_f.pack(side="left", padx=6)
            for n in r["predicted6"]:
                hit = n in actual_set
                tk.Label(pred_f, text=str(n), width=3,
                         bg=self.colors["accent"] if hit else "#1a2240",
                         fg="black" if hit else "#556688",
                         font=self.fonts["base_b"],
                         relief="raised", bd=1).pack(side="left", padx=1)

            # Hit counts
            for hit_val in [r["hit_top6"], r["hit_top10"], r["hit_top20"]]:
                tk.Label(row_f, text=f"{hit_val}/6", width=5,
                         bg=self.colors["card"],
                         fg=HIT_COLORS.get(hit_val, "white"),
                         font=self.fonts["mono"]).pack(side="left", padx=3)

            # Avg rank
            rank_col = (self.colors["pension"] if r["avg_rank"] <= 15 else
                        self.colors["gold"]    if r["avg_rank"] <= 25 else "#aaaaaa")
            tk.Label(row_f, text=f"{r['avg_rank']:.1f}", width=6,
                     bg=self.colors["card"], fg=rank_col,
                     font=self.fonts["mono"]).pack(side="left", padx=3)

            # Training size (credibility: shows data leakage proof)
            tk.Label(row_f, text=f"{r.get('training_size', '?')}회", width=6,
                     bg=self.colors["card"], fg="#666688",
                     font=self.fonts["small"]).pack(side="left", padx=3)

            # Computed at timestamp
            comp_at = (r.get("computed_at", "")[:16] if r.get("computed_at") else "")
            tk.Label(row_f, text=comp_at, width=18,
                     bg=self.colors["card"], fg="#445566",
                     font=self.fonts["small"]).pack(side="left", padx=3)

        # Bar chart
        if len(results) >= 2:
            fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.2), 4),
                                   facecolor=self.colors["bg"])
            draws = [f"#{r['draw_no']}" for r in results]
            x = np.arange(len(draws))
            w = 0.35
            ax.bar(x - w/2, [r["hit_top10"] for r in results], w,
                   label="Hit @ Top-10", color=self.colors["accent"], alpha=0.85)
            ax.bar(x + w/2, [r["hit_top20"] for r in results], w,
                   label="Hit @ Top-20", color=self.colors["pension"], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(draws, rotation=30, fontsize=9, color=self.colors["text"])
            ax.set_ylabel("Hit (out of 6)", color=self.colors["text"])
            ax.set_ylim(0, 7)
            ax.set_title("[Accuracy Archive] DL 소급 예측 적중 현황 (Data Leakage Free)",
                         fontsize=12, color=self.colors["gold"])
            ax.set_facecolor("#0f111a")
            ax.tick_params(colors=self.colors["text"])
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.25, color="#334")
            chart_canvas = FigureCanvasTkAgg(fig, master=inner)
            chart_canvas.draw()
            chart_canvas.get_tk_widget().pack(fill="x", padx=10, pady=12)
            plt.close(fig)


if __name__ == "__main__":
    root = tk.Tk()
    app = LotteryStableHub(root)
    root.mainloop()
