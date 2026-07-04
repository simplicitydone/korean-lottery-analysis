# 로또는 예측할 수 있을까? · Can the Lottery Be Predicted?

> **정직한 데이터 사이언스 스터디 · An Honest Data-Science Study**
> 로또 6/45 + 연금복권 데이터로 배우는 통계·머신러닝 방법론 — *정제 → 확률 → EDA → 검정 →
> 무작위성 → 베이지안 → ML → 비지도 → 백테스트*의 9단계 커리큘럼.

**결론(spoiler):** 두 복권은 공정한 균등 난수 생성기와 **통계적으로 구별되지 않습니다.** 빈도·핫/콜드·
딥러닝·그래디언트 부스팅 — 어떤 전략도 무작위 찍기를 이기지 못합니다. 이 저장소의 결과물은 *예측*이
아니라, 그것을 **다양한 SOTA 도구로 엄밀히 증명한 방법론**입니다.

> 원래 "AI 로또 예측 엔진"이었던 프로젝트를 정직하게 뒤집어, *예측이 왜 불가능한지*를 데이터로 보이는
> 교육용 스터디로 재구성했습니다. 학생이 따라 하며 배울 수 있도록 각 단계에서 *무엇을·왜·어떻게*를 자세히 설명합니다.

---

## 커리큘럼 (Curriculum) — `notebooks/`

| # | 노트북 | 배우는 데이터 사이언스 기법 | 핵심 결과 |
|---|--------|----------------------------|-----------|
| 00 | `00_overview` | 프로젝트 개요·방법론 지도·데이터 출처 | — |
| 01 | `01_data_cleaning` | tidy data, 인코딩 복구, 결측 분류 | 419열 → 9열, 유령/고아 제거 |
| 02 | `02_probability_foundations` | 조합론, **초기하분포**, CLT, 독립 vs 종속 | 기대 적중 0.8, 합계 μ=138 |
| 03 | `03_eda` | 분포·시각화·이론 대조 | 합계 평균 138.29 ≈ 138 |
| 04 | `04_hypothesis_testing` | χ²·런·자기상관, **효과크기·검정력·다중검정** | χ² p=0.97, w≈0.06 |
| 05 | `05_randomness_battery` | 엔트로피·KS·AD·ADF·**순열검정** | 6개 중 5개 통과 |
| 06 | `06_bayesian_estimation` | **Beta-Binomial 켤레사전분포**, 신용구간, 축소 | 44/45 구간이 6/45 포함 |
| 07 | `07_ml_done_right` | **walk-forward CV**, 보정, ROC/AUC, 순열 중요도 | AUC ≈ 0.50 |
| 08 | `08_unsupervised_structure` | **PCA·k-means·DBSCAN·t-SNE**, 실루엣 | 실루엣 ≈ 0.19 (군집 없음) |
| 09 | `09_backtest_and_verdict` | 누수 없는 백테스트, ROI, 종합 결론 | 모든 전략 ≈ 0.80 |

각 노트북은 **독립 실행**되며 (`nbconvert --execute` 무오류), 한국어 서술 + 영어 코드/용어(bilingual)로
작성되었습니다. 무거운 계산은 재사용 패키지 `src/lotto_ds/`가 담당하고, 노트북은 해석에 집중합니다.

## 왜 이 데이터인가 — 실제로 지저분한 원본
원본 `lottery.db`는 스프레드시트를 SQLite로 덤프한 파일이라 세 가지 오염이 있었습니다.
- **유령 열 399개** (`extra_*`, 전체 열의 95%) — 스프레드시트 잔여 셀.
- **인코딩 깨진 당첨금** — `prize_1st_total`이 `"2002006800��"`처럼 통화 접미사가 mojibake로 손상된 TEXT.
- **고아 시트 테이블 8개** — 의미 없는 `col_N` 헤더 덤프.

정제 파이프라인은 원본을 **수정하지 않고** 정규화된 `data/clean/lotto_clean.db`를 새로 만듭니다.

## SOTA 기법 하이라이트
- **초기하분포**로 등수별 확률과 기대값 0.8을 닫힌 형태로 유도 (백테스트 기준선).
- **베이지안 Beta-Binomial**: 각 번호 확률의 95% 신용구간 — 44/45가 공정값 6/45 포함.
- **올바른 ML**: 누수 없는 walk-forward로 그래디언트 부스팅 학습 → **AUC 0.49**, 특성 중요도 ≈ 0.
- **비지도 학습**: PCA·군집·t-SNE 모두 "숨은 구조 없음"으로 수렴 (실루엣 0.19).
- **정직한 통계**: 효과크기(Cohen's w)·검정력·다중검정 보정으로 p값의 함정을 함께 설명.
- 경계적 결과(Ljung–Box p=0.03, 스펙트럼 봉우리)를 숨기지 않고 **왜 신기루인지** 해부.

---

## 구조 · Layout

```
src/lotto_ds/          재사용 분석 패키지 (노트북 + 웹앱 공유)
  cleaning.py            raw dump → tidy, validated clean DB
  probability.py         조합론·초기하·CLT·독립성 (이론)
  features.py            draw-level 특성 + ML 특성 패널
  stats_tests.py         χ²·런·Ljung–Box·핫넘버 + 효과크기·검정력
  randomness.py          엔트로피·KS·AD·ADF·순열 스펙트럼 배터리
  bayesian.py            Beta-Binomial 후분포 + 신용구간
  ml_models.py           walk-forward 지도학습 (로지스틱·부스팅)·ROC·보정·중요도
  unsupervised.py        PCA·k-means·DBSCAN·t-SNE
  backtest.py            누수 없는 전략 백테스트 + 레거시 아카이브 대조
  viz.py                 하나의 검증된 시각 시스템 (palette·marks)
notebooks/00–09        재현 가능한 bilingual 커리큘럼
webapp.py + serve.py   재구성된 웹앱 — DS 리포트 API + 프로덕션 진입점
static_report/         self-contained 교육용 리포트 (Chart.js + KaTeX, 테마 지원)
tests/                 회귀 테스트 (cleaning·probability·bayesian·randomness, 26 cases)
lottery.db             원본 (수정 안 함) · data/clean/  생성물
```

## 실행 · Run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # Windows (notebooks+tests 포함)
# 배포용 런타임만: pip install -r requirements.txt

# 1) clean DB 생성
.venv/Scripts/python -m src.lotto_ds.cleaning

# 2) 테스트
.venv/Scripts/python -m pytest tests/ -q

# 3) 노트북 재현 (headless)
.venv/Scripts/python -m jupytext --to notebook notebooks/*.py
.venv/Scripts/python -m nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# 4) 웹 리포트
.venv/Scripts/python webapp.py            # http://127.0.0.1:5001
#   또는 프로덕션:  .venv/Scripts/python serve.py   (waitress, PORT=5000)
```

`webapp.py`의 모든 수치는 `/api/report`가 clean DB에서 **실시간 계산**합니다 (하드코딩 없음). 인증/비밀번호
없이 누구나 열람할 수 있는 공개 교육 자료입니다.

## 재현성 · Reproducibility
- 모든 노트북은 `nbconvert --execute`로 무오류 실행됩니다.
- 백테스트·ML은 시드 고정(`seed=42`), 각 회차는 오직 그 이전 데이터만 사용(누수 방지).
- 정제·확률·베이지안·무작위성 검증은 `tests/`에 회귀 테스트로 고정.

## 배포 · Deploy
- `Dockerfile` / `docker-compose.yml`: `serve.py`(waitress)로 포트 5000 서빙, `.env` 불필요.
- `lottery-hub.service.example`: systemd 등록 예시.

## 참고 · Notes
- 언어 정책: 서술은 한국어, 코드·기술 용어·차트 라벨은 영어(bilingual).
- 웹 수식은 KaTeX(CDN), 차트는 Chart.js(CDN) — 서빙되는 앱이므로 CDN 사용.
