# AGENTS.md

## Repo state

- Implemented FastAPI service per `prober-rec-build.md` (the design doc). Layout:
  `scripts/probe.sh` → `app/probe.py` → `app/recommender.py` + `app/huggingface.py`
  → `app/main.py` (routes + inline HTML UI), static fallback catalog in
  `app/models.json`, unit tests in `tests/`.
- The design doc is the source of intent, **but its ready-to-paste code has known
  bugs — the repo copies supersede it. Do not re-copy from the doc blindly**:
  - `probe.sh`: a bare `>` at end of line is a bash syntax error (script won't parse).
  - quantization regex truncates `Q4_K_M` to `Q4_K`; `BF16` matches `F16`.
  - `workload_score()` KeyErrors on workloads outside its fixed dict (HF-derived
    `uses` contains e.g. `"assistant"` → HTTP 500).
  - `requirements.txt` omits `huggingface_hub`.
- No CI, lint, or formatter exists. If you add tooling, add its config and a
  runnable command in the same change.

## Commands (verified)

```bash
python3 -m venv .venv && source .venv/bin/activate   # needs python3-venv: sudo apt install python3-venv python3-pip
pip install -r requirements.txt
python3 -m unittest discover -s tests -t .           # unit tests (16 tests)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Smoke check: `curl http://localhost:8000/hardware` and
`curl 'http://localhost:8000/recommendations?workload=coding'` (response includes
`"source"`: `huggingface` or `static`).

## Architecture

- **HF discovery is the primary catalog; `models.json` is only the fallback** when
  the hub is unreachable (doc's "Revised architecture" supersedes the static phase).
- Hard requirement, enforced by tests: never recommend a model whose estimated
  runtime footprint (weights + type overhead + per-4096-token context allowance)
  exceeds the conservative budget. Popularity is capped at 10 points so it can
  only break ties.
- `app/huggingface.py` targets **huggingface_hub 2.x**: `HfApi(timeout=)` and
  `list_models(direction=)` no longer exist; `model_info(expand=...)` cannot be
  combined with `files_metadata=True` (sizes only come from the latter), and
  expanding nullifies other fields — so sizes/metadata come from one
  `model_info(..., files_metadata=True)` call and summaries from `list_models`.
- A dead endpoint is caught by a TCP preflight before any API call: hf_hub's own
  retries cost ~23s per request otherwise. Failures set a 60s negative cache
  (`mark_failed()`); tune with `HF_TIMEOUT` / `HF_ENDPOINT` env vars.
- Memory budget (`available_memory_mb`): unified → 0.8×RAM; discrete GPU →
  0.8×VRAM + 0.35×RAM; CPU-only → 0.6×RAM. The `context` query param is the
  *requested* context (default 4096); models whose max context is below it are
  excluded, and footprint scales with it.
- `/models/{id}/command`: single-segment ids resolve against `models.json` first,
  then discovery; HF repo ids (`owner/name`) need the two-segment route
  `/models/{owner}/{name}/command`, or any unlisted GGUF repo is fetched live.

## Gotchas

- `scripts/probe.sh` must be executable, or `run_probe()` raises and the app falls
  back to the CPU-only Python probe — GPU/VRAM detection disappears silently.
  `run_probe()` also raises if the shell probe reports `ram_bytes: 0` (no psutil
  in the calling `python3`), because budget 0 would return zero recommendations.
- The shell probe writes unquoted JSON; empty `nvidia-smi`/`rocminfo` output can
  yield invalid JSON, which also silently falls back. Validate `/hardware` when
  touching the probe.
- Endpoint vocabulary: implemented code uses `?workload=` everywhere (validated
  against `general|coding|personal|embedding|rag`); the doc's future-API sections
  say `?use=` — keep it as `workload`.
- `probe.sh` honors `HF_TOKEN`, `LLAMA_BUCKET`, `LLAMA_VERSION` and needs `curl`
  (plus `zstd` for `.zst` payloads).
