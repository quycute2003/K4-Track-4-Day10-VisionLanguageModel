"""Stage messages and periodic status while a blocking model operation runs."""
import threading
import time
from contextlib import contextmanager


@contextmanager
def progress_stage(label, interval=30):
    started = time.monotonic()
    stop = threading.Event()
    print(label, flush=True)

    def heartbeat():
        while not stop.wait(interval):
            print(f"{label} — đang chờ hoàn thành ({time.monotonic()-started:.0f}s)", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
