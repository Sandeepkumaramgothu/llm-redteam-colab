from __future__ import annotations
from typing import Optional
import multiprocessing as mp
from src.app.server import run_app

# Use a SPAWN context so CUDA initializes cleanly inside the child process
_ctx = mp.get_context("spawn")

_server_proc: Optional[mp.Process] = None
_server_port: int = 8000

def start_server(port: int = 8000) -> bool:
    """
    Start Flask in a separate PROCESS (spawned) so CUDA works.
    Returns True if started, False if already running.
    """
    global _server_proc, _server_port
    if _server_proc is not None and _server_proc.is_alive():
        return False
    _server_port = port
    p = _ctx.Process(target=run_app, kwargs={"port": port}, daemon=True)
    p.start()
    _server_proc = p
    return True

def stop_server() -> bool:
    """
    Stop the Flask process if it is running.
    """
    global _server_proc
    if _server_proc is not None and _server_proc.is_alive():
        _server_proc.terminate()
        _server_proc.join(timeout=2)
        _server_proc = None
        return True
    return False

def server_port() -> int:
    return _server_port
