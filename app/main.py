from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from . import huggingface as hf
from .probe import probe
from .recommender import (
    DEFAULT_CONTEXT,
    candidates,
    load_models,
    model_fit,
    recommend,
)

app = FastAPI(
    title="LLM Hardware Probe",
    version="0.1.0",
    description=(
        "Hardware-aware local LLM model recommender "
        "for llama.cpp."
    ),
)

WORKLOAD_PATTERN = "^(%s)$" % "|".join(hf.WORKLOADS)

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
                ${model.parameters_b != null
                    ? model.parameters_b + 'B parameters'
                    : 'parameter count unknown'} |
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
def models(limit: int = Query(default=50, ge=1, le=200)):

    found, source = candidates(None)

    return {
        "source": source,
        "models": found[:limit],
    }

@app.get("/recommendations")
def recommendations(
    workload: str = Query(
        default="general",
        pattern=WORKLOAD_PATTERN,
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    context: int = Query(
        default=DEFAULT_CONTEXT,
        ge=4096,
        le=1_048_576,
    ),
):

    hw = probe()

    found, source = candidates(workload)

    results = recommend(
        hw,
        workload=workload,
        limit=limit,
        models=found,
        context=context,
    )

    return {
        "workload": workload,
        "context": context,
        "source": source,
        "hardware": hw.to_dict(),
        "recommendations": results,
    }

def _model_command(
    model_id: str,
    quantization: str,
    context: int,
):

    model = next(
        (m for m in load_models() if m["id"] == model_id),
        None,
    )

    if model is None:
        found, _ = candidates(None)

        model = next(
            (m for m in found if m["id"] == model_id or m.get("repo") == model_id),
            None,
        )

    if model is None:
        model = hf.fetch_entry(model_id)

    if not model:
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        )

    if quantization not in model.get("quantizations", {}):
        raise HTTPException(
            status_code=404,
            detail=f"Quantization {quantization} not found for this model",
        )

    hw = probe()

    fit = model_fit(hw, model, quantization, context=context)

    if fit is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This model/quantization does not fit "
                "the detected hardware."
            ),
        )

    return {
        "model": model["name"],
        "quantization": quantization,
        "command": (
            f"llama serve "
            f"-hf {model['repo']}:{quantization}"
        ),
        "fit": fit,
    }

@app.get("/models/{model_id}/command")
def model_command(
    model_id: str,
    quantization: str = "Q4_K_M",
    context: int = Query(
        default=DEFAULT_CONTEXT,
        ge=4096,
        le=1_048_576,
    ),
):

    return _model_command(model_id, quantization, context)

@app.get("/models/{owner}/{name}/command")
def model_command_hf(
    owner: str,
    name: str,
    quantization: str = "Q4_K_M",
    context: int = Query(
        default=DEFAULT_CONTEXT,
        ge=4096,
        le=1_048_576,
    ),
):

    return _model_command(f"{owner}/{name}", quantization, context)
