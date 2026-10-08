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
            curl -fsSL "$REPO/${LLAMA_VERSION:-latest}/$source" \
                > "$destination.tmp"
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

        if [ -z "$CPU_CORES" ] || [ "$CPU_CORES" = "0" ]; then
            CPU_CORES="$(grep -c '^processor' /proc/cpuinfo)"
        fi
    fi

    if { [ -z "$RAM_BYTES" ] || [ "$RAM_BYTES" = "0" ]; } && [ -r /proc/meminfo ]; then
        RAM_BYTES="$(( $(awk '/^MemTotal/ {print $2}' /proc/meminfo) * 1024 ))"
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
