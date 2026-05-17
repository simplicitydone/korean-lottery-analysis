# Lottery Hub v13.0 Knowledge Update
**Last Updated: 2026-04-16**

## System Architecture Changes
- **Database Centralization**: All engines and UI modules now prioritize `lottery.db`. Prediction accuracy results are no longer temporary; they are saved to the `prediction_accuracy` table immediately after computation.
- **Data Integrity (Data Leakage Protection)**: The accuracy engine enforces a strict "prior-data-only" training policy. Each record in the DB includes the `training_size` to prove it was trained only on data available before the draw.
- **Web Transition**: The project is shifting towards a headless model where `app.py` serves as the primary intelligence provider via REST API, with a modern SPA (Single Page Application) frontend.

## Deployment Information
- **Server Configuration**: The web app is configured to bind to `0.0.0.0:5000` to allow access from other devices on the same network (e.g., using the laptop as a server).
- **Dependencies**: `Flask` is now a core dependency. Use `pip install -r requirements.txt`.

## Data Verification
- **Draw 1219 (Lotto)**: Latest as of 2026-04-11.
- **Accuracy Baseline**: 2026-03-07 (#1214) for Lotto.
- **Pension Errors Detected (2026-04-16)**:
    - Draw #310 date mismatch (recorded as 04-16, should be 04-09).
    - Scraper failing to parse dynamic winning numbers and dates reliably.
    - Bonus numbers (for 2nd prize) are currently omitted from tracking.
    - UI alignment: Training draw number labels are confusingly placed.
