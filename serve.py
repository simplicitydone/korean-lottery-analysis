import os

from waitress import serve

from app import app, start_auto_updater, warm_models_async


if __name__ == "__main__":
    host = os.environ.get("LOTTERY_HOST", "0.0.0.0")
    port = int(os.environ.get("LOTTERY_PORT", "5000"))
    threads = int(os.environ.get("LOTTERY_THREADS", "8"))
    start_auto_updater()
    warm_models_async()
    serve(app, host=host, port=port, threads=threads)
