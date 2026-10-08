from __future__ import annotations

import json
from pathlib import Path

from . import huggingface as hf
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

DEFAULT_OVERHEAD_MB = 1000
DEFAULT_CONTEXT = 4096
CONTEXT_CHUNK_TOKENS = 4096
CONTEXT_CHUNK_MB = 256
MIN_CONTEXT_MB = 256
SAFETY_FACTOR = 0.80


def load_models() -> list[dict]:
    with CATALOG.open() as f:
        return json.load(f)

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

def context_memory_mb(context: int) -> int:
    context = max(int(context), 1)

    return max(MIN_CONTEXT_MB, (context // CONTEXT_CHUNK_TOKENS) * CONTEXT_CHUNK_MB)

def estimate_runtime_mb(
    model_mb: int,
    model_type: str,
    context: int,
) -> int:
    overhead = RUNTIME_OVERHEAD_MB.get(model_type, DEFAULT_OVERHEAD_MB)

    return model_mb + overhead + context_memory_mb(context)

def workload_score(model: dict, workload: str) -> int:
    uses = set(model.get("uses", []))

    score = 0

    if workload in uses:
        score += 100

    if workload == "coding" and "reasoning" in uses:
        score += 10

    if workload == "personal" and "assistant" in uses:
        score += 20

    if workload == "rag" and "embedding" in uses:
        score += 100

    return score

def backend_score(hw: HardwareProfile) -> int:
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

def popularity_score(model: dict) -> int:
    """
    Capped low so popularity can only break ties, never
    override hardware fit or workload fit.
    """

    downloads = model.get("downloads") or 0
    likes = model.get("likes") or 0

    return min(int(downloads / 100_000), 8) + min(int(likes / 1_000), 2)

def safety_label(fit_ratio: float) -> str:
    if fit_ratio <= 0.60:
        return "safe"

    if fit_ratio <= 0.80:
        return "balanced"

    return "tight"

def candidates(
    workload: str | None = None,
) -> tuple[list[dict], str]:
    """
    Hugging Face discovery with the static catalog as fallback.
    """

    try:
        models = hf.discover_all() if workload is None else hf.discover_models(workload)

        if models:
            return models, "huggingface"
    except Exception:
        pass

    return load_models(), "static"

def evaluate(
    model: dict,
    quantization: str,
    model_mb: int,
    hw: HardwareProfile,
    workload: str,
    budget: int,
    context: int,
) -> dict | None:

    required_mb = estimate_runtime_mb(model_mb, model.get("type", "chat"), context)

    if required_mb > budget:
        return None

    fit_ratio = required_mb / max(budget, 1)

    score = workload_score(model, workload)

    # Prefer models that make reasonable use of available memory
    # without being dangerously close to the limit.
    score += int(min(fit_ratio, 1.0) * 50)

    score += backend_score(hw)
    score += popularity_score(model)

    return {
        "model_id": model["id"],
        "name": model["name"],
        "family": model["family"],
        "repo": model["repo"],
        "type": model["type"],
        "quantization": quantization,
        "parameters_b": model.get("parameters_b"),
        "context": context,
        "estimated_model_mb": model_mb,
        "estimated_runtime_mb": required_mb,
        "memory_budget_mb": budget,
        "fit_ratio": round(fit_ratio, 3),
        "safety": safety_label(fit_ratio),
        "score": score,
        "reason": (
            f"{quantization} estimated at {required_mb} MB "
            f"against a conservative {budget} MB budget."
        ),
    }

def recommend(
    hw: HardwareProfile,
    workload: str = "general",
    limit: int = 10,
    models: list[dict] | None = None,
    context: int = DEFAULT_CONTEXT,
) -> list[dict]:

    budget = available_memory_mb(hw)

    if models is None:
        models, _ = candidates(workload)

    results = []

    for model in models:

        if workload not in model.get("uses", []):
            continue

        model_context = model.get("context")

        if model_context is not None and model_context < context:
            continue

        for quant, model_mb in model.get("quantizations", {}).items():

            row = evaluate(
                model,
                quant,
                model_mb,
                hw,
                workload,
                budget,
                context,
            )

            if row is not None:
                results.append(row)

    results.sort(
        key=lambda x: (
            x["score"],
            -x["estimated_runtime_mb"],
        ),
        reverse=True,
    )

    return results[:limit]

def model_fit(
    hw: HardwareProfile,
    model: dict,
    quantization: str,
    context: int = DEFAULT_CONTEXT,
) -> dict | None:

    budget = available_memory_mb(hw)

    model_mb = model.get("quantizations", {}).get(quantization)

    if model_mb is None:
        return None

    model_context = model.get("context")

    if model_context is not None and model_context < context:
        return None

    row = evaluate(
        model,
        quantization,
        model_mb,
        hw,
        (model.get("uses") or ["general"])[0],
        budget,
        context,
    )

    if row is None:
        return None

    return {
        "fit": row["safety"],
        "estimated_model_mb": row["estimated_model_mb"],
        "estimated_runtime_mb": row["estimated_runtime_mb"],
        "memory_budget_mb": row["memory_budget_mb"],
        "fit_ratio": row["fit_ratio"],
        "reason": row["reason"],
    }
