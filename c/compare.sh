#!/usr/bin/env bash
# Verify the C engine matches ../gzipt.py byte-for-byte, then time both.
# Usage: ./compare.sh [corpus]   (default data/tinyshakespeare.txt)
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS="${1:-data/tinyshakespeare.txt}"
PROMPT=$'MENENIUS:\n'
C=c/gzipt
PY="python3 gzipt.py"

make -C c >/dev/null

run() { "$@"; }                          # placeholder for readability
check() {  # name, extra-args...
    local name="$1"; shift
    $PY     --corpus "$CORPUS" --prompt "$PROMPT" "$@" >/tmp/cmp_py.txt 2>/dev/null
    $C      --corpus "$CORPUS" --prompt "$PROMPT" "$@" >/tmp/cmp_c.txt
    if diff -q /tmp/cmp_py.txt /tmp/cmp_c.txt >/dev/null; then
        echo "  [OK]   $name"
    else
        echo "  [DIFF] $name"; diff /tmp/cmp_py.txt /tmp/cmp_c.txt | head
    fi
}

echo "Correctness (output must be identical):"
check "greedy bytes"        --length 120 --temperature 0
check "sampled bytes (def)" --length 200
check "spans (sampled)"     --length 200 --mode spans --temperature 0.4
check "spans (greedy)"      --length 160 --mode spans --temperature 0

echo
echo "Speed (length 200, default settings):"
echo -n "  Python (8 workers): "; { /usr/bin/time -f '%e s wall, %U s cpu' \
    $PY --corpus "$CORPUS" --prompt "$PROMPT" --length 200 >/dev/null; } 2>&1
echo -n "  C (OpenMP):         "; { /usr/bin/time -f '%e s wall, %U s cpu' \
    $C  --corpus "$CORPUS" --prompt "$PROMPT" --length 200 >/dev/null; } 2>&1
echo -n "  C (single-thread):  "; { /usr/bin/time -f '%e s wall, %U s cpu' \
    c/gzipt-st --corpus "$CORPUS" --prompt "$PROMPT" --length 200 >/dev/null; } 2>&1
