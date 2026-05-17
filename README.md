# Lottery Pattern Hub

로또 6/45와 연금 720+ 데이터를 기반으로 통계 추천, 패턴 분석, 정확도 아카이브를 제공하는 Flask/SQLite 웹앱입니다.

## 실행

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

`.env`에서 `LOTTERY_SECRET_KEY`와 `LOTTERY_ACCESS_PASSWORD`를 바꾼 뒤 실행합니다.

```bash
LOTTERY_AUTO_UPDATE=0 .venv/bin/python serve.py
```

접속 주소는 기본값 기준 `http://127.0.0.1:5000`입니다.

## 주요 기능

- 최신 당첨 이력 조회
- 로그인 기반 통계 추천
- 로또/연금 패턴 분석 차트
- 사용자 입력 번호의 패턴 적합도 평가
- 정확도 아카이브 및 커버리지 표시
- 소량 단위 정확도 백필

## 백필 CLI

웹 서버와 별도로 정확도 아카이브를 채울 수 있습니다.

```bash
.venv/bin/python backfill_accuracy.py --mode LOTTO --count 5 --dry-run
.venv/bin/python backfill_accuracy.py --mode LOTTO --count 5
.venv/bin/python backfill_accuracy.py --mode PENSION --count 5
```

대량 백필은 오래 걸리므로 작은 단위로 나누어 실행하는 것을 권장합니다.

## 운영 참고

- `serve.py`는 `waitress`를 사용합니다.
- `LOTTERY_AUTO_UPDATE=1`이면 시작 시 누락 회차 자동 업데이트를 시도합니다.
- `LOTTERY_WARM_MODELS=1`이면 서버 시작 후 모델을 백그라운드에서 미리 로드합니다.
- `.model_cache/`는 재생성 가능한 모델 캐시입니다.
- `lottery-hub.service.example`은 systemd 등록 예시입니다.
