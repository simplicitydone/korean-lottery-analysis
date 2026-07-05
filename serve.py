"""serve.py — production entrypoint (waitress) for the data-science report web app.

Serves `webapp:app`. Builds the clean DB on startup, then warms the report cache in a background
thread so the first visitor gets an instant response instead of waiting ~30s for the initial build.
No auth, no auto-updater.
"""

import os
import threading

from waitress import serve

from webapp import app, _ensure_clean_db, warm_cache

if __name__ == "__main__":
    _ensure_clean_db()
    # Serve immediately; build the (expensive) report cache in the background.
    threading.Thread(target=warm_cache, daemon=True).start()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    threads = int(os.environ.get("THREADS", "8"))
    serve(app, host=host, port=port, threads=threads)
