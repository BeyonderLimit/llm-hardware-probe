# LLM Hardware Probe

Hardware-aware local LLM recommender for `llama.cpp`. Probes the machine
(CPU, RAM, GPU/VRAM, backends), then recommends GGUF models and quantizations
whose estimated runtime footprint fits a conservative memory budget.

Model discovery queries the Hugging Face Hub live (GGUF repos, per-workload
search); a small static catalog (`app/models.json`) is used when the Hub is
unreachable. Every response includes a `source` field telling you which one
was used.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> for the web UI, or run the tests:

```bash
python3 -m unittest discover -s tests -t .
```

## API

| Endpoint | Description |
| --- | --- |
| `GET /` | Web UI |
| `GET /health` | Liveness check |
| `GET /hardware` | Detected hardware profile |
| `GET /models?limit=50` | Model catalog (HF discovery, static fallback) |
| `GET /recommendations` | Hardware-fit recommendations |
| `GET /models/{id}/command` | Run command for a catalog id |
| `GET /models/{owner}/{name}/command` | Run command for any HF repo id |

`/recommendations` query parameters:

- `workload` — `general` \| `coding` \| `personal` \| `embedding` \| `rag` (default `general`)
- `context` — requested context length in tokens, 4096–1048576 (default 4096)
- `limit` — 1–50 (default 10)

```bash
curl http://localhost:8000/hardware
curl 'http://localhost:8000/recommendations?workload=coding'
curl 'http://localhost:8000/recommendations?workload=rag&context=32768'
curl 'http://localhost:8000/models/qwen3.5-4b/command?quantization=Q4_K_M'
curl 'http://localhost:8000/models/bartowski/Llama-3.2-1B-Instruct-GGUF/command?quantization=Q4_K_M'
```

## How it works

1. **Probe** — `scripts/probe.sh` emits a JSON hardware profile (OS, arch,
   CPU features, RAM, VRAM, backends). `app/probe.py` parses it and falls back
   to a pure-Python psutil probe if the shell script cannot run.
2. **Budget** — conservative memory budget: unified memory → 0.8 × RAM;
   discrete GPU → 0.8 × VRAM + 0.35 × RAM; CPU only → 0.6 × RAM.
3. **Discovery** — workload-specific searches over GGUF repos on the Hugging
   Face Hub, extracting per-quantization file sizes. On failure the static
   catalog is used (failures are negative-cached for 60s).
4. **Fit & scoring** — estimated footprint = weights + type overhead +
   context allowance. Anything over budget is discarded, period. The rest is
   ranked by workload match, how well the fit uses the budget, backend
   support, and popularity (capped so it only breaks ties). Results carry a
   `safety` label: `safe` (≤60% of budget), `balanced` (≤80%), `tight`.

## Environment variables

| Variable | Effect |
| --- | --- |
| `HF_TIMEOUT` | Timeout in seconds for per-model Hub requests (default 10) |
| `HF_ENDPOINT` | Alternate Hub endpoint |
| `HF_TOKEN` | Token for gated repos (also read by `probe.sh`) |
| `LLAMA_BUCKET`, `LLAMA_VERSION` | Installer bucket used by `probe.sh` |

## Layout

```
scripts/probe.sh        shell hardware probe (must be executable)
app/probe.py            probe wrapper + Python fallback
app/huggingface.py      Hub discovery, GGUF size/quantization parsing
app/recommender.py      memory budget, fit filtering, scoring
app/main.py             FastAPI routes + web UI
app/models.json         static fallback catalog
tests/                  unit tests (budget and fit invariants)
prober-rec-build.md     design document
```
