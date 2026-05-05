#!/bin/bash
# Run all benchmarks and print a Markdown summary for README.md.
# Usage: ./scripts/benchmark.sh
#
# Measures latency, RTF, and peak RSS in a single pass per backend
# using /usr/bin/time -l (macOS). On Linux replace -l with -v.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# macOS uses -l, GNU time uses -v. Auto-detect.
if /usr/bin/time -l true >/dev/null 2>&1; then
    TIME_FLAG="-l"
    RSS_KEY="maximum resident set size"
elif /usr/bin/time -v true >/dev/null 2>&1; then
    TIME_FLAG="-v"
    RSS_KEY="Maximum resident set size"
else
    echo "Warning: /usr/bin time not found. RSS will not be measured."
    TIME_FLAG=""
    RSS_KEY=""
fi

run_benchmark() {
    local backend="$1"
    shift
    local features="$1"
    shift

    local out_file="/tmp/phostt_benchmark_${backend}.txt"

    echo "Running ${backend} benchmark (release)..." >&2
    if [ -n "$features" ]; then
        if [ -n "$TIME_FLAG" ]; then
            /usr/bin/time "$TIME_FLAG" cargo test --release --test benchmark_report --features "$features" -- --ignored --nocapture >"$out_file" 2>&1
        else
            cargo test --release --test benchmark_report --features "$features" -- --ignored --nocapture >"$out_file" 2>&1
        fi
    else
        if [ -n "$TIME_FLAG" ]; then
            /usr/bin/time "$TIME_FLAG" cargo test --release --test benchmark_report -- --ignored --nocapture >"$out_file" 2>&1
        else
            cargo test --release --test benchmark_report -- --ignored --nocapture >"$out_file" 2>&1
        fi
    fi

    local latency
    latency=$(grep "Mean latency" "$out_file" | sed -n 's/.*| \([0-9.]*\) ms.*/\1/p')
    local rtf
    rtf=$(grep "RTF" "$out_file" | sed -n 's/.*| \([0-9.]*\).*/\1/p')

    local rss_mb="n/a"
    if [ -n "$RSS_KEY" ]; then
        local rss
        rss=$(grep "$RSS_KEY" "$out_file" | awk '{print $1}')
        if [ -n "$rss" ]; then
            rss_mb=$(echo "scale=1; $rss / 1024 / 1024" | bc)
        fi
    fi

    echo "${latency}:${rtf}:${rss_mb}"
}

echo "=== phostt Benchmark Suite ==="
echo ""

# --- 1. CPU ---
IFS=':' read -r CPU_LATENCY CPU_RTF CPU_MEM <<< "$(run_benchmark CPU "")"

# --- 2. CoreML ---
IFS=':' read -r COREML_LATENCY COREML_RTF COREML_MEM <<< "$(run_benchmark CoreML coreml)"

# --- 3. FLEURS WER (optional, slow) ---
if [ "${RUN_FLEURS:-}" = "1" ]; then
    echo "Running FLEURS WER benchmark (this takes ~10 min)..."
    FLEURS_OUT=$(cargo test --test fleurs_wer -- --ignored --nocapture 2>&1)
    FLEURS_WER=$(echo "$FLEURS_OUT" | grep "mean WER" | sed 's/.*mean WER = \([0-9.]*\).*/\1/')
    FLEURS_SAMPLES=$(echo "$FLEURS_OUT" | grep "mean WER" | sed 's/.*\([0-9]*\) samples.*/\1/')
    FLEURS_MD="### WER (FLEURS Vietnamese test split)

| Metric | Value |
|---|---|
| Samples | ${FLEURS_SAMPLES} |
| Mean WER | ${FLEURS_WER} |

*WER is high (>100%) because FLEURS contains many proper names, numbers,*
*and English terms that the model transcribes phonetically into Vietnamese.*
*Use this baseline for regression tracking, not absolute quality assessment.*"
else
    FLEURS_MD="*Set \`RUN_FLEURS=1\` to include the FLEURS WER benchmark (~10 min).*"
fi

echo ""
echo "## Results"
echo ""
echo "### Latency & Throughput"
echo ""
echo "| Backend | Mean Latency | RTF | Peak RSS |"
echo "|---|---|---|---|"
echo "| CPU (Apple Silicon M2 Pro) | ${CPU_LATENCY} ms | ${CPU_RTF}× | ${CPU_MEM} MB |"
echo "| CoreML (Neural Engine) | ${COREML_LATENCY} ms | ${COREML_RTF}× | ${COREML_MEM} MB |"
echo ""
echo "*RTF = real-time factor (higher is better). Audio: 3.74 s Vietnamese speech.*"
echo ""
echo "$FLEURS_MD"
