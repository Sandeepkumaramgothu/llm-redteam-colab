# src/app/serve_process.py
import threading, time
from typing import Optional
from .server import run_app

PORT: Optional[int] = None
THREAD: Optional[threading.Thread] = None

def start_server(port: int = 8200):
    global PORT, THREAD
    PORT = port
    if THREAD and THREAD.is_alive():
        return {"ok": False, "msg": "already running", "port": PORT}
    THREAD = threading.Thread(target=lambda: run_app(PORT), daemon=True)
    THREAD.start()
    time.sleep(0.6)  # give the server a moment to start
    return {"ok": True, "port": PORT}

def stop_server():
    import requests
    if PORT is None:
        return {"ok": False, "msg": "not running"}
    try:
        requests.post(f"http://127.0.0.1:{PORT}/shutdown", timeout=1.0)
        time.sleep(0.5)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def server_port():
    return PORT or 8200
