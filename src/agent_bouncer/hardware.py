"""Capture the hardware + runtime environment for a run, so model performance can be
compared **fairly** across machines (a 500 ms decoder on a laptop CPU is not the same
number on an A100). Pure stdlib with best-effort probes; never raises.
"""

from __future__ import annotations

import os
import platform
import subprocess


def _mem_gb() -> float | None:
    try:  # cross-platform if psutil is present
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    try:  # Linux
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
        return round(int(out.stdout.strip()) / 1e9, 1)
    except Exception:  # noqa: BLE001
        return None


def _cpu_name() -> str:
    name = platform.processor() or ""
    if not name and platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=2)
            name = out.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return name or platform.machine()


def _accelerator() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "gpu": "none", "gpu_name": None}
    gpu, gpu_name = "cpu", None
    try:
        if torch.cuda.is_available():
            gpu, gpu_name = "cuda", torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            gpu, gpu_name = "mps", "Apple Silicon (MPS)"
    except Exception:  # noqa: BLE001
        pass
    return {"torch": torch.__version__, "gpu": gpu, "gpu_name": gpu_name}


def hardware_info() -> dict:
    """A JSON-serializable snapshot of the machine + runtime."""
    acc = _accelerator()
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _mem_gb(),
        "python": platform.python_version(),
        **acc,
    }


def hardware_label(info: dict | None = None) -> str:
    """A compact one-line label, e.g. 'Apple Silicon (MPS) · 16 cores · 64 GB · py3.14'."""
    info = info or hardware_info()
    dev = info.get("gpu_name") or (info.get("gpu") or "cpu")
    mem = f"{info['memory_gb']} GB" if info.get("memory_gb") else "?"
    return f"{dev} · {info.get('cpu_count', '?')} cores · {mem} · py{info.get('python', '?')}"
