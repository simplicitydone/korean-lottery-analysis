# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 · 개요 — 로또는 예측할 수 있을까?
#
# > **정직한 데이터 사이언스 스터디 · An Honest Data-Science Study**
# > 로또 6/45 + 연금복권 데이터로 배우는 통계·머신러닝 방법론
#
# 이 저장소는 "AI 로또 예측기"가 아닙니다. 오히려 그 반대입니다 — **예측이 왜 불가능한지**를
# 올바른 데이터 사이언스 도구로 증명하는 교육용 프로젝트입니다. 목표는 두 가지입니다.
#
# 1. 실제로 지저분한 데이터를 정제하고, 엄밀하게 분석하는 **전 과정**을 보여준다.
# 2. 학생이 따라 하며 배울 수 있도록, 각 단계에서 *무엇을·왜·어떻게* 했는지 자세히 설명한다.
#
# ## 핵심 질문과 결론
#
# > **Q. 과거 당첨 데이터로 다음 회차를 유리하게 예측할 수 있는가?**
# > **A. 없다.** 로또는 공정한 균등 난수 생성기와 통계적으로 구별되지 않으며, 어떤 전략(빈도·핫/콜드·
# > 딥러닝·그래디언트 부스팅)도 무작위 기대치를 넘지 못한다. **이 "귀무결과"를 엄밀히 증명하는 것**이
# > 이 프로젝트의 결과물이다.

# %% [markdown]
# ## 방법론 지도 (Methodology map)
#
# 이 커리큘럼은 데이터 사이언스의 표준 워크플로를 그대로 따릅니다.
#
# | # | 노트북 | 배우는 것 (technique) | 핵심 질문 |
# |---|--------|----------------------|-----------|
# | 01 | 데이터 정제 | tidy data, 인코딩 복구, 결측 분류 | 원본은 믿을 만한가? |
# | 02 | 확률 기초 | 조합론, 초기하분포, CLT, 독립 vs 종속 | 이론이 말하는 정답은? |
# | 03 | 탐색적 분석 (EDA) | 분포·시각화·이론 대조 | 데이터가 이론과 맞는가? |
# | 04 | 가설 검정 | χ²·런·자기상관, 효과크기·검정력·다중검정 | 편차는 우연인가 구조인가? |
# | 05 | 무작위성 배터리 | 엔트로피·KS·ADF·순열검정 | 이 수열은 정말 무작위인가? |
# | 06 | 베이지안 추정 | 켤레사전분포, 신용구간, 축소 | 각 번호의 진짜 확률과 불확실성은? |
# | 07 | 올바른 머신러닝 | walk-forward CV, 보정, ROC/AUC, 중요도 | ML은 신호를 찾는가? |
# | 08 | 비지도 구조 탐색 | PCA, 군집, t-SNE, 실루엣 | 숨은 패턴/군집이 있는가? |
# | 09 | 백테스트 & 최종 결론 | 누수 없는 평가, ROI, 앙상블 종합 | 어떤 전략이 무작위를 이기는가? |
#
# 모든 분석 코드는 재사용 패키지 `src/lotto_ds/`에 있고, 노트북은 그것을 호출해 *이야기*를 만듭니다.
# 언어 정책: 서술은 한국어, 코드·용어·라벨은 영어(bilingual).

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.lotto_ds import RAW_DB, CLEAN_DB, cleaning

# %% [markdown]
# ## 데이터 출처 (Provenance)
#
# - **로또 6/45**: 동행복권 공식 당첨 이력. 원본은 스프레드시트를 SQLite로 덤프한 `lottery.db`.
# - **연금복권 720+**: 회차별 당첨 조/번호.
# - 원본 `lottery.db`는 **절대 수정하지 않습니다.** 정제 파이프라인이 정규화된
#   `data/clean/lotto_clean.db`를 새로 생성하고, 모든 분석은 이 clean DB만 사용합니다.

# %%
# clean DB가 없으면 생성 (재현성 보장)
if not CLEAN_DB.exists():
    cleaning.build_clean_db()

draws = cleaning.load_clean("draws")
pension = cleaning.load_clean("pension_draws")

print(f"로또 6/45   : {len(draws):,}회  (draw {draws.draw_no.min()}–{draws.draw_no.max()}, "
      f"{draws.draw_date.min()} ~ {draws.draw_date.max()})")
print(f"연금복권    : {pension.draw_no.nunique():,}회")
print(f"\n최근 3회 로또:")
print(draws.tail(3).to_string(index=False))

# %% [markdown]
# ## 이 노트북을 읽는 법
#
# - 각 노트북은 **독립 실행**됩니다 (`nbconvert --execute`로 무오류 검증).
# - 코드 셀은 짧습니다 — 무거운 계산은 `src/lotto_ds`가 담당하고, 노트북은 해석에 집중합니다.
# - 각 절 끝의 **⚠️ 흔한 함정** 박스는 학생이 저지르기 쉬운 실수를 짚어줍니다.
# - 웹 리포트(`webapp.py`)는 같은 계산을 실시간으로 보여주는 인터랙티브 요약본입니다.
#
# > 다음: **01 · 데이터 정제** — 분석의 90%는 데이터를 믿을 수 있게 만드는 일입니다.
