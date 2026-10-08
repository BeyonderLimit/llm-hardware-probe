# AGENTS.md

## Repo state

- Greenfield repo: no code, no commits, no lint/test/typecheck tooling yet.
- The only source of truth is `prober-rec-build.md` — a complete design doc containing
  ready-to-use code for the entire project. Read it before writing anything.
- No CI, no pre-commit, no package manifest exists. If you add tooling, add its
  config and a runnable command in the same change.

## Architecture (from the design doc)

- Target: FastAPI service that probes local hardware and recommends llama.cpp GGUF
  models that fit the machine's memory budget.
- Flow: `scripts/probe.sh` (shell probe, emits JSON) → `app/probe.py` (parses it,
  with a pure-Python psutil fallback) → `app/recommender.py` (fit/scoring) →
  `app/main.py` (FastAPI routes + inline HTML UI).
- **The doc's second half supersedes the first half.** The static `app/models.json`
  catalog (sections 1–6) is explicitly replaced by dynamic Hugging Face GGUF
  discovery (`app/huggingface.py`, "Revised architecture"). Implement the HF
  version; treat `models.json` as the prototype fallback, not the goal.
- Hard requirement from the doc: never recommend a model whose estimated runtime
  footprint (weights + KV cache + context + overhead + safety margin) exceeds the
  conservative memory budget. Hardware fit must dominate ranking; popularity is
  only a tiebreaker.

## Commands (as specified in the doc; verify they work once code exists)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install huggingface_hub        # needed for the HF discovery path, not in requirements.txt
chmod +x scripts/probe.sh          # probe.py executes this file directly
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Smoke check: `curl http://localhost:8000/hardware` and
`curl 'http://localhost:8000/recommendations?workload=coding'`.

## Gotchas

- `scripts/probe.sh` must be executable or `run_probe()` fails (it falls back to the
  CPU-only Python probe, silently hiding GPU/VRAM detection — check for that).
- The shell probe writes real numbers into JSON with no quoting; empty `nvidia-smi` /
  `rocminfo` output can produce invalid JSON, which also triggers the silent
  fallback. Validate `/hardware` output when touching the probe.
- The doc's `requirements.txt` is missing `huggingface_hub` and lists nothing for
  testing — don't assume it is complete.
- `probe.sh` downloads from a Hugging Face bucket and honors `HF_TOKEN`,
  `LLAMA_BUCKET`, and `LLAMA_VERSION` env vars; it needs `curl` (and `zstd` for
  `.zst` payloads).
- Endpoint vocabulary is inconsistent in the doc (`?workload=` vs `?use=`). The
  implemented code uses `workload`; the "future API" sections use `use`. Pick one
  and keep it consistent across routes, UI, and docs.
