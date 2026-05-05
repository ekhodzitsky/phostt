#!/bin/bash
# Run all benchmarks and print a Markdown summary for README.md.
# Usage: ./scripts/benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== phostt Benchmark Suite ==="
echo ""

# --- 1. Latency (CPU release) ---
echo "Running CPU latency benchmark (release)..."
CPU_OUT=$(cargo test --release --test benchmark_report -- --ignored --nocapture 2>&1)
CPU_LATENCY=$(echo "$CPU_OUT" | grep "Mean latency" | awk '{print $3}')
CPU_RTF=$(echo "$CPU_OUT" | grep "RTF" | awk '{print $4}')
CPU_MEM=$(/usr/bin/time -l cargo test --release --test benchmark_report -- --ignored 2>&1 | grep "maximum resident" | awk '{print $1}')
CPU_MEM_MB=$(echo "scale=1; $CPU_MEM / 1024 / 1024" | bc)

# --- 2. Latency (CoreML release) ---
echo "Running CoreML latency benchmark (release)..."
COREML_OUT=$(cargo test --release --test benchmark_report --features coreml -- --ignored --nocapture 2>&1)
COREML_LATENCY=$(echo "$COREML_OUT" | grep "Mean latency" | awk '{print $3}')
COREML_RTF=$(echo "$COREML_OUT" | grep "RTF" | awk '{print $4}')
COREML_MEM=$(/usr/bin/time -l cargo test --release --test benchmark_report --features coreml -- --ignored 2>&1 | grep "maximum resident" | awk '{print $1}')
COREML_MEM_MB=$(echo "scale=1; $COREML_MEM / 1024 / 1024" | bc)

# --- 3. FLEURS WER ---
echo "Running FLEURS WER benchmark (this takes ~10 min)..."
FLEURS_OUT=$(cargo test --test fleurs_wer -- --ignored --nocapture 2>&1)
FLEURS_WER=$(echo "$FLEURS_OUT" | grep "mean WER" | sed 's/.*mean WER = \([0-9.]*\).*/\1/')
FLEURS_SAMPLES=$(echo "$FLEURS_OUT" | grep "mean WER" | sed 's/.*\([0-9]*\) samples.*/\1/')

echo ""
echo "## Results"
echo ""
echo "### Latency & Throughput"
echo ""
echo "| Backend | Mean Latency | RTF | Peak RSS |"
echo "|---|---|---|---|"
echo "| CPU (Apple Silicon M2 Pro) | ${CPU_LATENCY} ms | ${CPU_RTF}× | ${CPU_MEM_MB} MB |"
echo "| CoreML (Neural Engine) | ${COREML_LATENCY} ms | ${COREML_RTF}× | ${COREML_MEM_MB} MB |"
echo ""
echo "*RTF = real-time factor (higher is better). Audio: 3.74 s Vietnamese speech.*"
echo ""
echo "### WER (FLEURS Vietnamese test split)"
echo ""
echo "| Metric | Value |"
echo "|---|---|"
echo "| Samples | ${FLEURS_SAMPLES} |"
echo "| Mean WER | ${FLEURS_WER} |"
echo ""
echo "*WER is high (>100%) because FLEURS contains many proper names, numbers,*"
echo "*and English terms that the model transcribes phonetically into Vietnamese.*"
echo "*Use this baseline for regression tracking, not absolute quality assessment.*"
