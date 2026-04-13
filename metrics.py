"""
System and application metrics logging for RAG Kanoon.

Captures CPU, memory, disk, and per-operation timing, and writes
structured log lines to both stdout and logs/app.log.
"""

import logging
import os
import time
from contextlib import contextmanager
from functools import wraps

import psutil

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8")
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logger = logging.getLogger("rag_kanoon")
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# System snapshot
# ---------------------------------------------------------------------------

def system_snapshot() -> dict:
    """Return a dict of current system resource usage."""
    proc = psutil.Process()
    mem = psutil.virtual_memory()
    cpu_total = psutil.cpu_percent(interval=None)   # non-blocking; uses last interval
    proc_mem = proc.memory_info()

    return {
        "cpu_total_pct": cpu_total,
        "cpu_proc_pct": proc.cpu_percent(interval=None),
        "mem_total_gb": round(mem.total / 1e9, 2),
        "mem_used_gb": round(mem.used / 1e9, 2),
        "mem_available_gb": round(mem.available / 1e9, 2),
        "mem_pct": mem.percent,
        "proc_rss_mb": round(proc_mem.rss / 1e6, 1),
        "proc_vms_mb": round(proc_mem.vms / 1e6, 1),
    }


def log_system_metrics(label: str = "system") -> None:
    """Log a one-line snapshot of current resource usage."""
    s = system_snapshot()
    logger.info(
        "[%s] cpu=%.1f%% (proc=%.1f%%)  mem=%.1f%%  "
        "mem_used=%.2fGB/%.2fGB  proc_rss=%.1fMB",
        label,
        s["cpu_total_pct"],
        s["cpu_proc_pct"],
        s["mem_pct"],
        s["mem_used_gb"],
        s["mem_total_gb"],
        s["proc_rss_mb"],
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@contextmanager
def timed(label: str, level: int = logging.INFO, log_metrics: bool = False):
    """
    Context manager that logs how long a block takes.

    Usage::
        with timed("chromadb_query"):
            results = collection.query(...)
    """
    start = time.perf_counter()
    logger.log(level, "[%s] start", label)
    if log_metrics:
        log_system_metrics(f"{label}/before")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "[%s] done  elapsed=%.3fs", label, elapsed)
        if log_metrics:
            log_system_metrics(f"{label}/after")


def timed_fn(label: str | None = None, log_metrics: bool = False):
    """
    Decorator that logs the duration of the wrapped function.

    Usage::
        @timed_fn("load_pdfs")
        def load_pdfs(folder): ...
    """
    def decorator(fn):
        op_label = label or fn.__qualname__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info("[%s] start", op_label)
            if log_metrics:
                log_system_metrics(f"{op_label}/before")
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[%s] done  elapsed=%.3fs", op_label, elapsed)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error("[%s] error after %.3fs: %s", op_label, elapsed, exc)
                raise
            finally:
                if log_metrics:
                    log_system_metrics(f"{op_label}/after")

        return wrapper
    return decorator
