from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
PROBE_SCRIPT = ROOT / "scripts" / "probe.sh"

@dataclass
class CPUInfo:
    model: str
    logical_cores: int
    avx: bool
    avx2: bool
    avx512: bool
    amx: bool
    neon: bool

@dataclass
class MemoryInfo:
    ram_bytes: int
    gpu_vram_bytes: int
    unified_memory: bool

@dataclass
class GPUInfo:
    vendor: str
    name: str

@dataclass
class HardwareProfile:
    os: str
    arch: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: GPUInfo
    backends: list[str]

    def to_dict(self):
        return asdict(self)

def run_probe() -> HardwareProfile:
    result = subprocess.run(
        [str(PROBE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    data = json.loads(result.stdout)

    if data["memory"]["ram_bytes"] <= 0:
        raise RuntimeError("shell probe returned no usable RAM value")

    return HardwareProfile(
        os=data["os"],
        arch=data["arch"],
        cpu=CPUInfo(
            model=data["cpu"]["model"],
            logical_cores=data["cpu"]["logical_cores"],
            avx=data["cpu"]["features"]["avx"],
            avx2=data["cpu"]["features"]["avx2"],
            avx512=data["cpu"]["features"]["avx512"],
            amx=data["cpu"]["features"]["amx"],
            neon=data["cpu"]["features"]["neon"],
        ),
        memory=MemoryInfo(
            ram_bytes=data["memory"]["ram_bytes"],
            gpu_vram_bytes=data["memory"]["gpu_vram_bytes"],
            unified_memory=data["memory"]["unified_memory"],
        ),
        gpu=GPUInfo(
            vendor=data["gpu"]["vendor"],
            name=data["gpu"]["name"],
        ),
        backends=data["backends"],
    )

def fallback_probe() -> HardwareProfile:
    """
    Pure-Python fallback if the shell probe cannot execute.
    """

    ram = psutil.virtual_memory().total

    arch = platform.machine().lower()

    if arch in ("x86_64", "amd64"):
        arch = "x86_64"
    elif arch in ("arm64", "aarch64"):
        arch = "aarch64"

    return HardwareProfile(
        os=platform.system().lower(),
        arch=arch,
        cpu=CPUInfo(
            model=platform.processor(),
            logical_cores=os.cpu_count() or 1,
            avx=False,
            avx2=False,
            avx512=False,
            amx=False,
            neon=arch == "aarch64",
        ),
        memory=MemoryInfo(
            ram_bytes=ram,
            gpu_vram_bytes=0,
            unified_memory=False,
        ),
        gpu=GPUInfo(
            vendor="",
            name="",
        ),
        backends=["cpu"],
    )

def probe() -> HardwareProfile:
    try:
        return run_probe()
    except Exception:
        return fallback_probe()
