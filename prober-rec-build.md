# Hardware Prober/LLM recommend

**separate hardware detection from model selection**:

 1. Reuse the important parts of your installer’s CUDA/ROCm/Vulkan/Metal/CPU probing.
2. Produce a normalized hardware profile:
   - OS / architecture
   - CPU cores and instruction sets
   - system RAM
   - GPU vendor/backend
   - VRAM or unified memory
   - conservative memory budget
3. Have FastAPI expose `/hardware`, `/models`, `/recommendations`, and `/models/{id}/command`.
4. Keep a model catalog with **estimated GGUF sizes per quantization**, capabilities, and workload tags.
5. Never recommend a model whose estimated runtime footprint exceeds the calculated budget.
6. Return several choices such as **Safe / Balanced / Maximum**, rather than simply saying "you can run an 8B model."

 This fits well with `llama.cpp`, which supports CPU, CUDA, HIP/ROCm, Metal, Vulkan and other backends, and supports CPU/GPU hybrid inference. Its server also exposes OpenAI-compatible APIs.  GitHub+1

 Below is a working starting implementation.

## Project layout

```
llm-hardware-probe/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── probe.py
│   ├── recommender.py
│   └── models.json
├── scripts/
│   └── probe.sh
├── requirements.txt
└── README.md
```

 ### `requirements.txt`

```
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.8
psutil>=6.0
```

---

 # 1\. Hardware probe

 The shell portion below is intentionally based on your original installer. It retains the architecture/OS detection and CUDA/ROCm/Vulkan/CPU/Metal concepts, but instead of installing anything it reports what it finds.

 ### `scripts/probe.sh`

```
#!/usr/bin/env bash

set -u

LLAMA_BUCKET="${LLAMA_BUCKET:-ggml-org/install.sh}"
REPO="https://huggingface.co/buckets/$LLAMA_BUCKET/resolve"

WORKDIR="${TMPDIR:-/tmp}/llm-hardware-probe"
mkdir -p "$WORKDIR"

die() {
    printf '%s\n' "$*" >&2
    exit 111
}

info() {
    printf '%s\n' "$*" >&2
}

check_bin() {
    command -v "$1" >/dev/null 2>&1
}

curl() {
    if [ -n "${HF_TOKEN:-}" ]; then
        command curl -H "Authorization: Bearer $HF_TOKEN" "$@"
    else
        command curl "$@"
    fi
}

dl_bin() {
    local destination="$1"
    local source="$2"

    [ -x "$destination" ] && return 0

    check_bin curl || return 1

    case "$source" in
        (*.zst)
            check_bin zstd || return 1
            curl -fsSL "$REPO/${LLAMA_VERSION:-latest}/$source" |
                zstd -d > "$destination.tmp"
            ;;
        (*)
            curl -fsSL "$REPO/${LLAMA_VERSION:-latest}/$source" >
                "$destination.tmp"
            ;;
    esac

    chmod +x "$destination.tmp" &&
        mv "$destination.tmp" "$destination"
}

json_escape() {
    printf '%s' "$1" |
        sed 's/\\/\\\\/g; s/"/\\"/g'
}

ARCH=""
OS=""

case "$(uname -m)" in
    arm64|aarch64)
        ARCH="aarch64"
        ;;
    amd64|x86_64)
        ARCH="x86_64"
        ;;
    *)
        ARCH="unknown"
        ;;
esac

case "$(uname -s)" in
    Linux)
        OS="linux"
        ;;
    FreeBSD)
        OS="freebsd"
        ;;
    Darwin)
        OS="macos"
        ;;
    *)
        OS="unknown"
        ;;
esac

CPU_MODEL=""
CPU_CORES=""
RAM_BYTES=""
CPU_FEATURES=""

if check_bin python3; then
    eval "$(
        python3 - <<'PY'
import os
import platform

try:
    import psutil

    print("RAM_BYTES=%s" % psutil.virtual_memory().total)
    print("CPU_CORES=%s" % (psutil.cpu_count(logical=True) or 1))
except Exception:
    print("RAM_BYTES=0")
    print("CPU_CORES=0")

print("CPU_MODEL=%s" % platform.processor().replace(" ", "_"))
PY
    )"
fi

if [ "$OS" = "linux" ]; then

    if [ -r /proc/cpuinfo ]; then
        CPU_MODEL="${CPU_MODEL:-$(awk -F': ' '/model name/ {print $2; exit}' /proc/cpuinfo)}"
        CPU_FEATURES="$(awk -F': ' '/flags/ {print $2; exit}' /proc/cpuinfo)"
    fi

elif [ "$OS" = "macos" ]; then

    CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"

    if [ -z "$CPU_MODEL" ]; then
        CPU_MODEL="$(sysctl -n hw.model 2>/dev/null || true)"
    fi

    CPU_FEATURES="$(sysctl -n machdep.cpu.features 2>/dev/null || true)"

    if [ -z "$RAM_BYTES" ]; then
        RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    fi

    if [ -z "$CPU_CORES" ]; then
        CPU_CORES="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 0)"
    fi
fi

BACKENDS=""
GPU_VENDOR=""
GPU_NAME=""
VRAM_BYTES=0
UNIFIED_MEMORY=false

# ------------------------------------------------------------
# CUDA
# ------------------------------------------------------------

if [ "$OS" = "linux" ] && check_bin nvidia-smi; then

    GPU_VENDOR="NVIDIA"

    GPU_NAME="$(
        nvidia-smi \
            --query-gpu=name \
            --format=csv,noheader 2>/dev/null |
        head -n1
    )"

    VRAM_BYTES="$(
        nvidia-smi \
            --query-gpu=memory.total \
            --format=csv,noheader,nounits 2>/dev/null |
        head -n1 |
        awk '{printf "%.0f", $1 * 1024 * 1024}'
    )"

    BACKENDS="${BACKENDS}cuda,"

fi

# ------------------------------------------------------------
# ROCm / AMD
# ------------------------------------------------------------

if [ "$OS" = "linux" ] && check_bin rocminfo; then

    if [ -z "$GPU_VENDOR" ]; then
        GPU_VENDOR="AMD"
    fi

    if [ -z "$GPU_NAME" ]; then
        GPU_NAME="$(
            rocminfo 2>/dev/null |
            awk -F': ' '/Marketing Name/ {print $2; exit}'
        )"
    fi

    BACKENDS="${BACKENDS}rocm,"

fi

# ------------------------------------------------------------
# Vulkan
# ------------------------------------------------------------

if check_bin vulkaninfo; then

    if vulkaninfo --summary >/dev/null 2>&1; then
        BACKENDS="${BACKENDS}vulkan,"

        if [ -z "$GPU_NAME" ]; then
            GPU_NAME="$(
                vulkaninfo --summary 2>/dev/null |
                awk -F': ' '/deviceName/ {print $2; exit}'
            )"
        fi
    fi

fi

# ------------------------------------------------------------
# Apple Metal
# ------------------------------------------------------------

if [ "$OS" = "macos" ]; then

    if [ -n "$CPU_MODEL" ]; then

        case "$CPU_MODEL" in
            *Apple*)
                BACKENDS="${BACKENDS}metal,"
                GPU_VENDOR="Apple"
                GPU_NAME="$CPU_MODEL"
                UNIFIED_MEMORY=true
                VRAM_BYTES="$RAM_BYTES"
                ;;
        esac

    fi

fi

# ------------------------------------------------------------
# CPU always exists as fallback
# ------------------------------------------------------------

BACKENDS="${BACKENDS}cpu,"

# Remove trailing comma
BACKENDS="${BACKENDS%,}"

# Determine useful CPU features
HAS_AVX=false
HAS_AVX2=false
HAS_AVX512=false
HAS_AMX=false
HAS_NEON=false

case " $CPU_FEATURES " in
    *" avx "*)
        HAS_AVX=true
        ;;
esac

case " $CPU_FEATURES " in
    *" avx2 "*)
        HAS_AVX2=true
        ;;
esac

case " $CPU_FEATURES " in
    *" avx512"*)
        HAS_AVX512=true
        ;;
esac

case " $CPU_FEATURES " in
    *" amx_"*)
        HAS_AMX=true
        ;;
esac

case "$ARCH" in
    aarch64)
        HAS_NEON=true
        ;;
esac

cat <<EOF
{
  "os": "$(json_escape "$OS")",
  "arch": "$(json_escape "$ARCH")",
  "cpu": {
    "model": "$(json_escape "$CPU_MODEL")",
    "logical_cores": ${CPU_CORES:-0},
    "features": {
      "avx": ${HAS_AVX},
      "avx2": ${HAS_AVX2},
      "avx512": ${HAS_AVX512},
      "amx": ${HAS_AMX},
      "neon": ${HAS_NEON}
    }
  },
  "memory": {
    "ram_bytes": ${RAM_BYTES:-0},
    "gpu_vram_bytes": ${VRAM_BYTES:-0},
    "unified_memory": ${UNIFIED_MEMORY}
  },
  "gpu": {
    "vendor": "$(json_escape "$GPU_VENDOR")",
    "name": "$(json_escape "$GPU_NAME")"
  },
  "backends": [
    "$(echo "$BACKENDS" | sed 's/,/","/g')"
  ]
}
EOF
```

 Make it executable:

```
chmod +x scripts/probe.sh
```

---

 # 2\. Python hardware abstraction

 The Python layer is useful because FastAPI shouldn't have to understand shell output.

 ### `app/probe.py`

```
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
```

---

 # 3\. Model catalog

 The important design decision is **not to assume that parameter count equals memory consumption**.

 Instead, give every model/quantization combination an estimated weight size and let the recommender add:

 - model weights
- KV cache
- runtime overhead
- safety margin

 For example, `Q4_K_M` is generally substantially smaller than F16, so the same model can be suitable or unsuitable depending on quantization.

 `llama.cpp` supports 1.5-, 2-, 3-, 4-, 5-, 6- and 8-bit quantization and can also split workloads between CPU and GPU.  GitHub

 ### `app/models.json`

 This is deliberately a **catalog you can expand**, rather than something hard-coded into the application.

```
[
  {
    "id": "qwen3.5-0.8b",
    "name": "Qwen3.5 0.8B",
    "family": "Qwen",
    "repo": "ggml-org/Qwen3.5-0.8B-GGUF",
    "type": "chat",
    "parameters_b": 0.8,
    "context": 32768,
    "quantizations": {
      "Q4_K_M": 600,
      "Q8_0": 1000
    },
    "uses": [
      "general",
      "personal",
      "assistant",
      "fast",
      "low-memory"
    ]
  },
  {
    "id": "qwen3.5-4b",
    "name": "Qwen3.5 4B",
    "family": "Qwen",
    "repo": "Qwen/Qwen3.5-4B-GGUF",
    "type": "chat",
    "parameters_b": 4.0,
    "context": 262144,
    "quantizations": {
      "Q4_K_M": 2800,
      "Q5_K_M": 3300,
      "Q8_0": 4600
    },
    "uses": [
      "general",
      "coding",
      "personal",
      "assistant",
      "reasoning",
      "tools"
    ]
  },
  {
    "id": "qwen3-embedding-4b",
    "name": "Qwen3-Embedding 4B",
    "family": "Qwen",
    "repo": "Qwen/Qwen3-Embedding-4B-GGUF",
    "type": "embedding",
    "parameters_b": 4.0,
    "context": 32768,
    "quantizations": {
      "Q4_K_M": 2500,
      "Q5_K_M": 3000,
      "Q8_0": 4200
    },
    "uses": [
      "embedding",
      "rag",
      "semantic-search",
      "retrieval"
    ]
  }
]
```

 The Qwen3-Embedding GGUF repository explicitly provides llama.cpp usage and quantized variants such as Q4\_K\_M, so it is a good example of how an embedding candidate can live in the same catalog.  Hugging Face+1

 Likewise, current Qwen3.5 4B GGUF builds are around the 2.8 GB range for Q4\_K\_M, making that sort of model a useful candidate for modest machines.  Hugging Face

---

 # 4\. Recommendation engine

 This is where the "don't exceed current hardware" requirement gets enforced.

 ### `app/recommender.py`

```
from __future__ import annotations

import json
from pathlib import Path

from .probe import HardwareProfile

CATALOG = Path(__file__).resolve().parent / "models.json"

# Extra memory beyond the GGUF file itself.
#
# These are intentionally conservative. Actual requirements depend
# on context size, batch size, KV cache, architecture, and backend.
RUNTIME_OVERHEAD_MB = {
    "chat": 900,
    "embedding": 500,
}

SAFETY_FACTOR = 0.80

def load_models() -> list[dict]:
    with CATALOG.open() as f:
        return json.load(f)

def mb(value: int) -> int:
    return value * 1024 * 1024

def available_memory_mb(hw: HardwareProfile) -> int:
    """
    Determine the memory budget available to the model.

    Discrete GPU:
        Prefer VRAM, with CPU fallback.

    Apple unified memory:
        Use a conservative portion of system memory.

    CPU:
        Use conservative portion of system RAM.
    """

    ram_mb = hw.memory.ram_bytes // (1024 * 1024)
    vram_mb = hw.memory.gpu_vram_bytes // (1024 * 1024)

    if hw.memory.unified_memory:
        # Leave substantial memory to the OS and applications.
        return int(ram_mb * SAFETY_FACTOR)

    if vram_mb > 0:
        # Don't consume all VRAM.
        gpu_budget = int(vram_mb * SAFETY_FACTOR)

        # Hybrid CPU/GPU inference can use RAM too.
        cpu_budget = int(ram_mb * 0.35)

        return gpu_budget + cpu_budget

    return int(ram_mb * 0.60)

def workload_score(model: dict, workload: str) -> int:
    uses = set(model.get("uses", []))

    scores = {
        "general": 0,
        "coding": 0,
        "personal": 0,
        "embedding": 0,
        "rag": 0,
    }

    if workload in uses:
        scores[workload] += 100

    if workload == "coding" and "reasoning" in uses:
        scores[workload] += 10

    if workload == "personal" and "assistant" in uses:
        scores[workload] += 20

    if workload == "rag" and "embedding" in uses:
        scores[workload] += 100

    return scores.get(workload, 0)

def backend_score(model: dict, hw: HardwareProfile) -> int:
    score = 0

    if "cuda" in hw.backends:
        score += 20

    if "rocm" in hw.backends:
        score += 20

    if "metal" in hw.backends:
        score += 20

    if "vulkan" in hw.backends:
        score += 10

    if "cpu" in hw.backends:
        score += 2

    return score

def recommend(
    hw: HardwareProfile,
    workload: str = "general",
    limit: int = 10,
) -> list[dict]:

    budget = available_memory_mb(hw)

    results = []

    for model in load_models():

        if workload not in model["uses"]:
            continue

        overhead = RUNTIME_OVERHEAD_MB.get(
            model["type"],
            1000,
        )

        for quant, model_mb in model["quantizations"].items():

            required_mb = model_mb + overhead

            if required_mb > budget:
                continue

            fit_ratio = required_mb / max(budget, 1)

            score = workload_score(model, workload)

            # Prefer models that make reasonable use of available memory
            # without being dangerously close to the limit.
            score += int(min(fit_ratio, 1.0) * 50)

            score += backend_score(model, hw)

            if fit_ratio <= 0.60:
                safety = "safe"
            elif fit_ratio <= 0.80:
                safety = "balanced"
            else:
                safety = "tight"

            results.append({
                "model_id": model["id"],
                "name": model["name"],
                "family": model["family"],
                "repo": model["repo"],
                "type": model["type"],
                "quantization": quant,
                "parameters_b": model["parameters_b"],
                "context": model["context"],
                "estimated_model_mb": model_mb,
                "estimated_runtime_mb": required_mb,
                "memory_budget_mb": budget,
                "fit_ratio": round(fit_ratio, 3),
                "safety": safety,
                "score": score,
                "reason": (
                    f"{quant} estimated at {required_mb} MB "
                    f"against a conservative {budget} MB budget."
                ),
            })

    results.sort(
        key=lambda x: (
            x["score"],
            -x["estimated_runtime_mb"],
        ),
        reverse=True,
    )

    return results[:limit]
```

---

 # 5\. FastAPI server

 The API gives you both a machine-readable service and a very simple browser UI.

 ### `app/main.py`

```
from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .probe import probe
from .recommender import load_models, recommend

app = FastAPI(
    title="LLM Hardware Probe",
    version="0.1.0",
    description=(
        "Hardware-aware local LLM model recommender "
        "for llama.cpp."
    ),
)

@app.get("/", response_class=HTMLResponse)
def index():

    return """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>LLM Hardware Probe</title>

    <style>
        body {
            font-family: system-ui, sans-serif;
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
            background: #111827;
            color: #f9fafb;
        }

        .card {
            background: #1f2937;
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
        }

        button {
            background: #2563eb;
            color: white;
            border: 0;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            margin-right: 8px;
        }

        pre {
            white-space: pre-wrap;
        }

        .model {
            border-left: 4px solid #2563eb;
            padding-left: 15px;
            margin: 20px 0;
        }

        .safe {
            color: #34d399;
        }

        .balanced {
            color: #fbbf24;
        }

        .tight {
            color: #fb7185;
        }
    </style>
</head>

<body>

<h1>LLM Hardware Probe</h1>

<div class="card">
    <h2>Hardware</h2>
    <pre id="hardware">Loading...</pre>
</div>

<div class="card">

    <h2>Intended workload</h2>

    <button onclick="recommend('general')">
        General AI
    </button>

    <button onclick="recommend('coding')">
        Coding
    </button>

    <button onclick="recommend('personal')">
        Personal Assistant
    </button>

    <button onclick="recommend('rag')">
        RAG / Embeddings
    </button>

</div>

<div class="card">
    <h2>Recommendations</h2>
    <div id="results"></div>
</div>

<script>

async function loadHardware() {

    const response = await fetch('/hardware');
    const data = await response.json();

    document.getElementById('hardware').textContent =
        JSON.stringify(data, null, 2);
}

async function recommend(workload) {

    const response = await fetch(
        '/recommendations?workload=' + workload
    );

    const data = await response.json();

    const results = document.getElementById('results');

    results.innerHTML = '';

    for (const model of data.recommendations) {

        const div = document.createElement('div');

        div.className = 'model';

        div.innerHTML = `
            <h3>
                ${model.name}
                — ${model.quantization}
            </h3>

            <p>
                ${model.type} |
                ${model.parameters_b}B parameters |
                ${model.safety}
            </p>

            <p>
                Estimated memory:
                <strong>
                    ${model.estimated_runtime_mb} MB
                </strong>
                /
                ${model.memory_budget_mb} MB available
            </p>

            <p>
                ${model.reason}
            </p>

            <p>
                <code>
                    llama serve -hf
                    ${model.repo}:${model.quantization}
                </code>
            </p>
        `;

        results.appendChild(div);
    }
}

loadHardware();
recommend('general');

</script>

</body>
</html>
"""

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.get("/hardware")
def hardware():

    hw = probe()

    return hw.to_dict()

@app.get("/models")
def models():

    return {
        "models": load_models()
    }

@app.get("/recommendations")
def recommendations(
    workload: str = Query(
        default="general",
        pattern="^(general|coding|personal|embedding|rag)$",
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
):

    hw = probe()

    results = recommend(
        hw,
        workload=workload,
        limit=limit,
    )

    return {
        "workload": workload,
        "hardware": hw.to_dict(),
        "recommendations": results,
    }

@app.get("/models/{model_id}/command")
def model_command(
    model_id: str,
    quantization: str = "Q4_K_M",
):

    models = load_models()

    model = next(
        (m for m in models if m["id"] == model_id),
        None,
    )

    if not model:
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        )

    hw = probe()

    candidates = [
        x
        for x in recommend(
            hw,
            workload=model["uses"][0],
            limit=100,
        )
        if x["model_id"] == model_id
        and x["quantization"] == quantization
    ]

    if not candidates:
        raise HTTPException(
            status_code=409,
            detail=(
                "This model/quantization does not fit "
                "the detected hardware."
            ),
        )

    result = candidates[0]

    return {
        "model": model["name"],
        "quantization": quantization,
        "command": (
            f"llama serve "
            f"-hf {model['repo']}:{quantization}"
        ),
        "fit": result,
    }
```

---

 # 6\. Run it

```
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

chmod +x scripts/probe.sh

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

 Then open:

```
http://localhost:8000
```

 Or query it programmatically:

```
curl http://localhost:8000/hardware
```

```
curl 'http://localhost:8000/recommendations?workload=coding'
```

```
curl 'http://localhost:8000/recommendations?workload=rag'
```

---

 # What the API would return

 For example, conceptually:

```
{
  "workload": "coding",
  "hardware": {
    "os": "linux",
    "arch": "x86_64",
    "cpu": {
      "model": "AMD Ryzen",
      "logical_cores": 16,
      "features": {
        "avx": true,
        "avx2": true,
        "avx512": true,
        "amx": false,
        "neon": false
      }
    },
    "memory": {
      "ram_bytes": 34359738368,
      "gpu_vram_bytes": 12884901888,
      "unified_memory": false
    },
    "gpu": {
      "vendor": "NVIDIA",
      "name": "RTX ..."
    },
    "backends": [
      "cuda",
      "vulkan",
      "cpu"
    ]
  },
  "recommendations": [
    {
      "model_id": "qwen3.5-4b",
      "name": "Qwen3.5 4B",
      "quantization": "Q4_K_M",
      "parameters_b": 4.0,
      "estimated_runtime_mb": 3700,
      "memory_budget_mb": 14000,
      "fit_ratio": 0.264,
      "safety": "safe"
    }
  ]
}
```

 The important part is that **the recommendation is based on the actual machine**, not merely "4B models usually run on 8 GB."

---

 # One improvement I strongly recommend

 I'd make the next version somewhat more sophisticated than simply:

```
model size < available memory
```

 The actual decision should be:

```
available memory
        │
        ├── GPU VRAM
        │
        ├── system RAM
        │
        └── unified memory
               │
               ▼
       runtime memory budget
               │
               ├── GGUF weights
               ├── KV cache
               ├── compute buffers
               ├── context length
               └── safety margin
               │
               ▼
        model compatibility
               │
               ▼
        workload scoring
               │
               ▼
     ┌─────────┼─────────┐
     ▼         ▼         ▼
    Safe    Balanced   Maximum
```

 That matters particularly for long-context models. A model whose GGUF is only 4 GB can require considerably more memory when you give it a very large context.

 Also, `llama.cpp` can use CPU+GPU hybrid inference when the model exceeds available VRAM, so the probe should distinguish **"fits entirely in VRAM"** from **"fits using hybrid CPU/GPU inference."**  GitHub

 ## I would therefore make the API eventually look like this

```
GET /hardware
GET /hardware/backends

GET /models
GET /models/{id}

GET /recommendations?use=general
GET /recommendations?use=coding
GET /recommendations?use=personal
GET /recommendations?use=embedding
GET /recommendations?use=rag

GET /recommendations?use=coding&context=32768
GET /recommendations?use=coding&context=131072

GET /models/{id}/command

POST /probe
POST /recommendations
```

 And the response can explicitly explain **why** something was selected:

```
{
  "model": "Qwen3.5 4B",
  "quantization": "Q4_K_M",
  "fit": "safe",
  "deployment": "GPU",
  "reason": [
    "CUDA detected",
    "12 GB VRAM detected",
    "estimated model footprint: 2.8 GB",
    "estimated runtime footprint: 3.7 GB",
    "conservative available budget: 9.6 GB",
    "coding capability: strong",
    "4B model leaves substantial headroom for context"
  ]
}
```

 That makes the system much more useful than a hardware benchmark: **it becomes a local LLM deployment advisor.**

 The underlying `llama.cpp` server is already OpenAI-compatible and has `/v1/models`, chat-completions, embeddings, health monitoring, continuous batching, and other server functionality, so FastAPI can remain the **hardware/model orchestration layer** rather than trying to become the inference engine itself.  GitHub+1

 llama.cpp project  ·  Qwen3-Embedding-4B-GGUF

 If this is intended to become a real deployable project, the next step I'd make is **dynamic GGUF discovery from Hugging Face + actual VRAM/RAM measurement + model-fit calculation based on context size**, rather than maintaining `models.json` manually.

 That is actually a better architecture because the Hub can filter models by library, task, parameter count, tags, downloads, etc., and its API exposes GGUF metadata and repository files. Hugging Face specifically provides a GGUF filter at `hf.co/models?library=gguf`.  Hugging Face+1

 ## Revised architecture

```
                       ┌──────────────────────┐
                       │     Local Machine     │
                       └──────────┬───────────┘
                                  │
                         hardware probe
                                  │
                 ┌────────────────▼────────────────┐
                 │       Hardware Profile          │
                 │                                 │
                 │ RAM / VRAM / CPU / GPU / OS    │
                 │ CUDA / ROCm / Vulkan / Metal   │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────┐
                    │     FastAPI Service     │
                    └────────────┬───────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
        ┌──────────────────┐          ┌──────────────────┐
        │ Hugging Face API │          │ Fit/Scoring      │
        │                  │          │ Engine           │
        │ models           │          │                  │
        │ GGUF repos       │          │ RAM/VRAM        │
        │ metadata         │          │ quantization    │
        │ downloads        │          │ context         │
        │ tasks            │          │ workload        │
        └────────┬─────────┘          └────────┬─────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                     ┌──────────────────────┐
                     │ Recommendations      │
                     │                      │
                     │ General AI           │
                     │ Coding               │
                     │ Personal Assistant   │
                     │ Embeddings / RAG     │
                     └──────────────────────┘
```

 The important distinction is that **Hugging Face tells us what exists; our hardware probe determines what is appropriate.**

---

 # 1\. Replace `models.json` with Hugging Face

 I'd remove:

```
app/models.json
```

 and replace it with an `HfModelProvider`.

 Install:

```
pip install huggingface_hub
```

 Hugging Face's Python API supports model search, filtering by parameter count, pipeline, library, tags, sorting and limiting results. It can also request expanded `gguf` and `siblings` information.  Hugging Face+1

 ### `app/huggingface.py`

```
from __future__ import annotations

import re
from functools import lru_cache

from huggingface_hub import HfApi

api = HfApi()

GGUF_EXTENSIONS = (".gguf", ".gguf2")

def bytes_to_mb(value: int) -> int:
    return int(value / 1024 / 1024)

def parse_quantization(filename: str) -> str | None:
    """
    Extract common GGUF quantization names from filenames.

    Examples:

        model.Q4_K_M.gguf
        model.Q8_0.gguf
        model.F16.gguf
    """

    name = filename.upper()

    patterns = [
        r"(IQ[1-4]_[A-Z0-9]+)",
        r"(Q[2-8]_[A-Z0-9]+)",
        r"(Q[2-8])",
        r"(F16)",
        r"(F32)",
        r"(BF16)",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)

        if match:
            return match.group(1)

    return None

def get_gguf_files(model_id: str) -> list[dict]:

    info = api.model_info(
        model_id,
        expand=["siblings", "gguf", "downloads", "likes"],
    )

    files = []

    for sibling in info.siblings or []:

        filename = sibling.rfilename

        if not filename.lower().endswith(GGUF_EXTENSIONS):
            continue

        size = sibling.size

        # Some Hub responses expose LFS metadata instead of
        # sibling.size.
        if not size and sibling.lfs:
            size = sibling.lfs.size

        if not size:
            continue

        files.append({
            "filename": filename,
            "size_bytes": size,
            "size_mb": bytes_to_mb(size),
            "quantization": parse_quantization(filename),
        })

    return files

def model_summary(info) -> dict:

    return {
        "id": info.id,
        "author": getattr(info, "author", None),
        "downloads": getattr(info, "downloads", 0),
        "likes": getattr(info, "likes", 0),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": getattr(info, "tags", []),
        "library": getattr(info, "library_name", None),
        "last_modified": str(
            getattr(info, "last_modified", "")
        ),
    }

@lru_cache(maxsize=128)
def search_gguf_models(
    query: str | None = None,
    limit: int = 50,
    sort: str = "downloads",
) -> list[dict]:

    models = api.list_models(
        search=query,
        filter="gguf",
        sort=sort,
        direction=-1,
        limit=limit,
    )

    results = []

    for model in models:

        try:
            files = get_gguf_files(model.id)
        except Exception:
            continue

        if not files:
            continue

        results.append({
            **model_summary(model),
            "files": files,
        })

    return results
```

 This is much better than assuming that every Hugging Face model can run under `llama.cpp`.

 The Hub explicitly supports GGUF and has a viewer for GGUF metadata/tensor information.  Hugging Face

---

 # 2\. Search Hugging Face according to the user's goal

 Now we can make the user's request meaningful.

 Instead of:

```
/recommendations?workload=coding
```

 internally translating that to our own static model list, translate it into **Hugging Face searches**.

 For example:

```
WORKLOAD_SEARCHES = {

    "general": [
        "instruct",
        "chat",
        "general",
    ],

    "coding": [
        "coder",
        "code",
        "coding",
        "instruct",
    ],

    "personal": [
        "assistant",
        "instruct",
        "chat",
    ],

    "embedding": [
        "embedding",
        "embed",
    ],

    "rag": [
        "embedding",
        "reranker",
        "instruct",
    ],
}
```

 However, I would **not rely exclusively on model names**.

 Hugging Face's API lets us use tags, pipeline tasks, parameter counts and other metadata, so we should combine those signals.  Hugging Face

---

 # 3\. A better model discovery layer

 I'd change the previous recommendation engine to this:

```
from .huggingface import search_gguf_models

WORKLOAD_TERMS = {
    "general": [
        "instruct",
        "chat",
    ],

    "coding": [
        "coder",
        "code",
        "coding",
    ],

    "personal": [
        "assistant",
        "instruct",
        "chat",
    ],

    "embedding": [
        "embedding",
        "embed",
    ],

    "rag": [
        "embedding",
        "reranker",
    ],
}

def discover_models(workload: str):

    terms = WORKLOAD_TERMS.get(
        workload,
        WORKLOAD_TERMS["general"],
    )

    found = {}

    for term in terms:

        models = search_gguf_models(
            query=term,
            limit=30,
        )

        for model in models:
            found[model["id"]] = model

    return list(found.values())
```

 So a coding request could discover:

```
Qwen
DeepSeek
Devstral
Qwen Coder
Granite Code
StarCoder
etc.
```

 depending on what is actually present and discoverable on Hugging Face **at the time the request is made**.

 That is a major advantage over the static catalog.

---

 # 4\. Then apply the hardware filter

 This is where the project becomes interesting.

 Suppose the Hub returns:

```
Model A
  Q4_K_M = 3.1 GB
  Q5_K_M = 3.7 GB
  Q8_0   = 5.2 GB

Model B
  Q4_K_M = 7.2 GB
  Q5_K_M = 8.4 GB
  Q8_0   = 11.1 GB

Model C
  Q4_K_M = 15.2 GB
```

 And our probe reports:

```
RAM:       32 GB
VRAM:      8 GB
Backend:   CUDA
CPU:       16 cores
```

 We can say:

```
Model A Q8_0
    ✓ fits

Model B Q4_K_M
    ✓ fits with GPU/CPU hybrid

Model B Q8_0
    ✗ too close to memory limit

Model C Q4_K_M
    ✗ exceeds conservative memory budget
```

 This means **we don't need to know the model beforehand**.

---

 # 5\. Don't use GGUF file size as the complete memory calculation

 This is an important change I'd make from the earlier implementation.

 Instead of:

```
required = gguf_size + 900
```

 use:

```
required memory =
    GGUF weights
  + KV cache
  + compute buffers
  + context allocation
  + runtime overhead
  + safety margin
```

 For example:

```
def estimate_runtime_memory(
    model_size_mb: int,
    context: int,
    parameters_b: float | None = None,
) -> int:

    # Base runtime overhead.
    overhead = 700

    # Conservative KV/cache allowance.
    context_memory = max(
        256,
        int(context / 4096) * 256,
    )

    return (
        model_size_mb
        + overhead
        + context_memory
    )
```

 Then expose context to the API:

```
/recommendations?workload=coding&context=32768
```

 versus:

```
/recommendations?workload=coding&context=131072
```

 A model that is appropriate at 16K context may no longer be appropriate at 128K.

---

 # 6\. Get model information from the Hub

 For each candidate, I'd request the detailed model information.

 Hugging Face's `model_info` supports expanded properties including `gguf`, `siblings`, `config`, `cardData`, `tags`, `pipeline_tag`, and other metadata.  Hugging Face+1

 So the internal pipeline becomes:

```
info = api.model_info(
    model_id,
    expand=[
        "gguf",
        "siblings",
        "config",
        "cardData",
        "downloads",
        "likes",
        "tags",
        "pipeline_tag",
    ],
)
```

 Then extract:

```
repo
files
GGUF sizes
quantization
parameters
architecture
context
task
tags
downloads
likes
```

---

 # 7\. Use Hugging Face popularity as one scoring factor

 I'd also incorporate:

```
downloads
likes
last modified
```

 but **not let popularity override hardware compatibility**.

 For example:

```
def popularity_score(model):

    downloads = model.get("downloads", 0)
    likes = model.get("likes", 0)

    return (
        min(downloads / 100_000, 20)
        + min(likes / 1_000, 10)
    )
```

 Then:

```
final_score = (
    hardware_fit_score
    + workload_score
    + quantization_score
    + popularity_score
)
```

 Hardware fit should dominate.

---

 # 8\. Quantization should become a first-class recommendation

 Instead of returning:

```
Qwen 4B
```

 return:

```
Qwen 4B
├── Q4_K_M  ← Recommended
├── Q5_K_M
├── Q6_K
└── Q8_0
```

 For example:

```
{
  "model": "some/model-GGUF",
  "recommendations": [
    {
      "file": "model.Q4_K_M.gguf",
      "quantization": "Q4_K_M",
      "size_gb": 2.8,
      "fit": "safe"
    },
    {
      "file": "model.Q5_K_M.gguf",
      "quantization": "Q5_K_M",
      "size_gb": 3.4,
      "fit": "safe"
    },
    {
      "file": "model.Q8_0.gguf",
      "quantization": "Q8_0",
      "size_gb": 5.1,
      "fit": "tight"
    }
  ]
}
```

 That gives the user an actual choice between:

 **Fast / Safe**

 and

 **Higher quality / More memory**

---

 # 9\. Hugging Face should also provide the download/run command

 This is another advantage.

 The response can include:

```
{
  "model": "Qwen/SomeModel-GGUF",
  "file": "SomeModel-Q4_K_M.gguf",
  "huggingface": "https://huggingface.co/Qwen/SomeModel-GGUF",
  "command": "llama-cli -hf Qwen/SomeModel-GGUF:Q4_K_M"
}
```

 Your original installer already uses the Hugging Face bucket mechanism, so this fits naturally into the project.

 The current Hugging Face CLI also supports filtering models by `llama.cpp`, searching, parameter count, pipeline tag, and sorting by downloads, which reinforces the idea of using the Hub itself as the dynamic catalog.  Hugging Face

---

 # 10\. I would change the API

 I'd make the public API:

```
GET /hardware
```

 Returns detected machine capabilities.

```
GET /models/search?q=qwen
```

 Searches Hugging Face.

```
GET /models/discover?use=general
```

 Discovers candidate GGUF models.

```
GET /recommendations?use=general
```

 Hardware-aware recommendations.

```
GET /recommendations?use=coding
```

 Coding recommendations.

```
GET /recommendations?use=personal
```

 Personal assistant recommendations.

```
GET /recommendations?use=embedding
```

 Embedding recommendations.

```
GET /recommendations?use=rag
```

 RAG recommendations.

 And importantly:

```
GET /recommendations?use=coding&context=32768
```

 or:

```
GET /recommendations?use=coding&context=131072
```

---

 # 11\. The UI could become much nicer

 Instead of exposing raw hardware numbers, I'd have the web UI say:

```
┌────────────────────────────────────────────────────┐
│              LOCAL LLM ADVISOR                     │
├────────────────────────────────────────────────────┤
│                                                    │
│  Hardware                                          │
│                                                    │
│  CPU       AMD Ryzen 9 7950X       16 cores       │
│  RAM       64 GB                                  │
│  GPU       NVIDIA RTX 4080         16 GB VRAM     │
│  Backend   CUDA                                   │
│                                                    │
├────────────────────────────────────────────────────┤
│                                                    │
│  What do you want to do?                           │
│                                                    │
│  [ General AI ] [ Coding ] [ Personal Assistant ] │
│                                                    │
│  [ Embeddings ] [ RAG ]                            │
│                                                    │
├────────────────────────────────────────────────────┤
│                                                    │
│  Recommended Models                                │
│                                                    │
│  ★ BEST MATCH                                      │
│  ┌──────────────────────────────────────────────┐ │
│  │ Qwen ...                                     │ │
│  │ 7B • Q4_K_M • 4.8 GB                         │ │
│  │                                              │ │
│  │ ███████████░░░  Memory utilization: 61%     │ │
│  │                                              │ │
│  │ ✓ CUDA                                       │ │
│  │ ✓ Fits VRAM                                  │ │
│  │ ✓ Coding                                     │ │
│  │                                              │ │
│  │ [Use this model] [View on Hugging Face]      │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ALTERNATIVES                                     │
│                                                    │
│  Model A       3B Q8      Very safe               │
│  Model B       7B Q5      Balanced                │
│  Model C       14B Q4     Hybrid CPU/GPU         │
│                                                    │
└────────────────────────────────────────────────────┘
```

 ## One other thing I'd change

 I would **not automatically recommend the most downloaded Hugging Face model**.

 The ranking should be roughly:

```
                 HARDWARE FIT
                     │
                     │
              ┌──────▼──────┐
              │              │
              │  Can it run? │──── NO ────> discard
              │              │
              └──────┬───────┘
                     │ YES
                     ▼
               WORKLOAD FIT
                     │
                     ▼
              QUANTIZATION
                     │
                     ▼
              CONTEXT FIT
                     │
                     ▼
             BACKEND OPTIMIZATION
                     │
                     ▼
                POPULARITY
                     │
                     ▼
               FINAL RANK
```

 That makes the service genuinely **hardware-aware**, rather than just being a Hugging Face search frontend.

 ### The key change

 So I would replace the previous static:

```
models.json
      ↓
recommend()
```

 with:

```
hardware probe
      ↓
Hugging Face model discovery
      ↓
GGUF repository inspection
      ↓
quantization/file-size extraction
      ↓
hardware memory calculation
      ↓
context/KV-cache calculation
      ↓
workload classification
      ↓
fit filtering
      ↓
quality/popularity ranking
      ↓
recommendations
```

 That also means the system can discover **new models and new GGUF quantizations without you having to update the application**. Hugging Face explicitly supports searching/filtering GGUF models and exposes GGUF repository metadata through its Hub APIs.  Hugging Face+1

 If you want, I can next turn the earlier prototype into a **complete FastAPI project that dynamically queries Hugging Face, parses the actual GGUF files, detects CUDA/ROCm/Vulkan/Metal, calculates VRAM/RAM fit, and presents the recommendations in a web dashboard** rather than leaving the model catalog static.
