from __future__ import annotations

import os
import re
import socket
import time
from functools import lru_cache
from urllib.parse import urlparse

from huggingface_hub import HfApi
from huggingface_hub.constants import ENDPOINT

api = HfApi()

MODEL_TIMEOUT = float(os.environ.get("HF_TIMEOUT", "10"))

GGUF_EXTENSIONS = (".gguf", ".gguf2")

WORKLOADS = ("general", "coding", "personal", "embedding", "rag")

WORKLOAD_TERMS = {
    "general": ["instruct", "chat"],
    "coding": ["coder", "code", "coding"],
    "personal": ["assistant", "instruct"],
    "embedding": ["embedding", "embed"],
    "rag": ["embedding", "reranker"],
}

EMBEDDING_PIPELINE_TAGS = (
    "embedding",
    "sentence-similarity",
    "feature-extraction",
)

_hf_state = {"fail_until": 0.0}


class HfUnavailable(Exception):
    pass


def _reachable(timeout: float = 3.0) -> bool:
    """
    Cheap TCP preflight. huggingface_hub retries failed calls 5 times
    with backoff (~23s), so a dead endpoint must be detected before
    any API request is made.
    """

    parsed = urlparse(ENDPOINT)
    host = parsed.hostname or "huggingface.co"
    port = parsed.port or (443 if parsed.scheme != "http" else 80)

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _ensure_available() -> None:
    if time.time() < _hf_state["fail_until"]:
        raise HfUnavailable("huggingface hub disabled after an earlier failure")

    if not _reachable():
        mark_failed()
        raise HfUnavailable("huggingface hub endpoint unreachable")


def mark_failed() -> None:
    _hf_state["fail_until"] = time.time() + 60.0


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
        r"(IQ[1-4](?:_[A-Z0-9]+)+)",
        r"(Q[2-8](?:_[A-Z0-9]+)+)",
        r"(Q[2-8])(?![A-Z0-9_])",
        r"(BF16)(?![A-Z0-9_])",
        r"(F16)(?![A-Z0-9_])",
        r"(F32)(?![A-Z0-9_])",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)

        if match:
            return match.group(1)

    return None


def parse_parameters_b(model_id: str) -> float | None:
    name = model_id.split("/")[-1]

    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None

def _extract_context(info) -> int | None:
    config = getattr(info, "config", None)

    if isinstance(config, dict):
        for key in ("context_length", "max_position_embeddings", "n_positions", "n_ctx"):
            value = config.get(key)

            if isinstance(value, int) and 0 < value <= 4_000_000:
                return value

    return None

def get_gguf_detail(model_id: str) -> dict:

    info = api.model_info(model_id, files_metadata=True, timeout=MODEL_TIMEOUT)

    files = []

    for sibling in info.siblings or []:

        filename = sibling.rfilename

        if not filename.lower().endswith(GGUF_EXTENSIONS):
            continue

        size = getattr(sibling, "size", None)

        if not size:
            lfs = getattr(sibling, "lfs", None)
            size = getattr(lfs, "size", None)

        if not size:
            continue

        files.append({
            "filename": filename,
            "size_bytes": size,
            "size_mb": bytes_to_mb(size),
            "quantization": parse_quantization(filename),
        })

    return {
        "files": files,
        "context": _extract_context(info),
    }

def model_summary(info) -> dict:

    return {
        "id": info.id,
        "author": getattr(info, "author", None),
        "downloads": getattr(info, "downloads", 0) or 0,
        "likes": getattr(info, "likes", 0) or 0,
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": getattr(info, "tags", []) or [],
        "library": getattr(info, "library_name", None),
        "last_modified": str(
            getattr(info, "last_modified", "")
        ),
    }

@lru_cache(maxsize=128)
def search_gguf_models(
    query: str | None = None,
    limit: int = 10,
    sort: str = "downloads",
) -> tuple[dict, ...]:

    models = api.list_models(
        search=query,
        filter="gguf",
        sort=sort,
        limit=limit,
    )

    results = []

    for model in models:

        try:
            detail = get_gguf_detail(model.id)
        except Exception:
            continue

        if not detail["files"]:
            continue

        results.append({
            **model_summary(model),
            **detail,
        })

    return tuple(results)

def derive_uses(model_id: str, tags: list[str], model_type: str) -> list[str]:
    text = f"{model_id} {' '.join(tags)}".lower()

    uses: set[str] = set()

    if model_type == "embedding":
        uses.update(["embedding", "rag", "semantic-search", "retrieval"])
    else:
        uses.add("general")

    if "coder" in text or "coding" in text or re.search(r"(?<![a-z])code(?![a-z])", text):
        uses.update(["coding", "reasoning"])

    if "reasoning" in text:
        uses.add("reasoning")

    if "assistant" in text or "instruct" in text or "chat" in text:
        uses.update(["personal", "assistant"])

    return sorted(uses)

def _model_type(model: dict) -> str:
    pipeline_tag = model.get("pipeline_tag")

    if pipeline_tag in EMBEDDING_PIPELINE_TAGS:
        return "embedding"

    text = f"{model['id']} {' '.join(model.get('tags') or [])}".lower()

    if "embedding" in text or "reranker" in text:
        return "embedding"

    return "chat"

def to_catalog_entry(model: dict, workload: str | None = None) -> dict | None:

    quantizations: dict[str, int] = {}

    for item in model["files"]:
        quant = item["quantization"]

        if not quant:
            continue

        quantizations[quant] = max(quantizations.get(quant, 0), item["size_mb"])

    if not quantizations:
        return None

    repo = model["id"]
    model_type = _model_type(model)

    uses = set(derive_uses(repo, model.get("tags") or [], model_type))

    if workload:
        uses.add(workload)

    return {
        "id": repo,
        "name": repo.split("/")[-1],
        "family": model.get("author") or repo.split("/")[0],
        "repo": repo,
        "type": model_type,
        "parameters_b": parse_parameters_b(repo),
        "context": model.get("context"),
        "quantizations": quantizations,
        "uses": sorted(uses),
        "downloads": model.get("downloads") or 0,
        "likes": model.get("likes") or 0,
    }

def _max_context(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b

    if b is None:
        return a

    return max(a, b)

def _merge(merged: dict[str, dict], entry: dict) -> None:
    existing = merged.get(entry["id"])

    if existing is None:
        merged[entry["id"]] = entry
        return

    existing["uses"] = sorted(set(existing["uses"]) | set(entry["uses"]))

    for quant, size in entry["quantizations"].items():
        existing["quantizations"][quant] = max(
            existing["quantizations"].get(quant, 0), size
        )

    existing["context"] = _max_context(existing.get("context"), entry.get("context"))
    existing["downloads"] = max(existing["downloads"], entry["downloads"])
    existing["likes"] = max(existing["likes"], entry["likes"])

def fetch_entry(repo_id: str) -> dict | None:
    """
    Build a catalog entry for an arbitrary GGUF repo id, without
    going through workload discovery. Returns None if the repo
    has no usable GGUF files.
    """

    try:
        _ensure_available()
        detail = get_gguf_detail(repo_id)
    except Exception:
        return None

    if not detail["files"]:
        return None

    model = {
        "id": repo_id,
        "tags": [],
        "pipeline_tag": None,
        "author": None,
        "downloads": 0,
        "likes": 0,
        **detail,
    }

    return to_catalog_entry(model)

def discover_models(workload: str) -> list[dict]:
    _ensure_available()

    terms = WORKLOAD_TERMS.get(workload, WORKLOAD_TERMS["general"])

    merged: dict[str, dict] = {}

    for term in terms:
        try:
            models = search_gguf_models(query=term)
        except Exception as exc:
            mark_failed()
            raise HfUnavailable(str(exc)) from exc

        for model in models:
            entry = to_catalog_entry(model, workload=workload)

            if entry:
                _merge(merged, entry)

    return sorted(merged.values(), key=lambda m: m["downloads"], reverse=True)

def discover_all() -> list[dict]:
    merged: dict[str, dict] = {}

    for workload in WORKLOADS:
        for entry in discover_models(workload):
            _merge(merged, entry)

    return sorted(merged.values(), key=lambda m: m["downloads"], reverse=True)
