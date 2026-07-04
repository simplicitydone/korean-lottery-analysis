"""serve.py — production entrypoint (waitress) for the data-science report web app.

Serves `webapp:app`. Builds the clean DB on startup if it is missing. No auth, no auto-updater.
"""

import os

from waitress import serve

from webapp import app, _ensure_clean_db

if __name__ == "__main__":
    _ensure_clean_db()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    threads = int(os.environ.get("THREADS", "8"))
    serve(app, host=host, port=port, threads=threads)
