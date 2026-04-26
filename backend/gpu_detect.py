"""GPU detection — returns the pod's first visible GPU as a normalized dict.

Output shape: {"type": "nvidia"|"amd"|"cpu", "name": str, "vram_gb": int, "driver": str}
"""

import shutil
import subprocess


def _try_nvidia():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=4,
        )
        line = out.strip().splitlines()[0]
        name, mem_mib, driver = [x.strip() for x in line.split(",")]
        return {
            "type": "nvidia",
            "name": name,
            "vram_gb": round(int(mem_mib) / 1024),
            "driver": driver,
        }
    except Exception:
        return None


def _try_rocm():
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"],
            text=True,
            timeout=4,
        )
        # rocm-smi CSV is non-trivial; do a best-effort parse of the first GPU.
        lines = [l for l in out.splitlines() if l and not l.startswith("device")]
        # e.g.: card0,gfx1100,Radeon RX 7900 XTX,...,vram total memory(B),25753026560
        name = "AMD GPU"
        vram_gb = 0
        for l in lines:
            cells = [c.strip() for c in l.split(",")]
            if len(cells) >= 3 and cells[2]:
                name = cells[2]
            if "vram" in l.lower():
                for c in cells:
                    if c.isdigit() and int(c) > 1024 ** 3:
                        vram_gb = round(int(c) / 1024 ** 3)
                        break
        return {"type": "amd", "name": name, "vram_gb": vram_gb, "driver": "rocm"}
    except Exception:
        return None


def detect_gpu():
    return _try_nvidia() or _try_rocm() or {
        "type": "cpu",
        "name": "CPU only",
        "vram_gb": 0,
        "driver": "",
    }
