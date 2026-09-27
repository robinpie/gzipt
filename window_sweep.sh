#!/usr/bin/env bash
# Push the context window far past DEFLATE's 32KB to see if HellaSwag ever
# "groks". zstd long-distance matching sees up to 8MB, so the prime size is the
# real variable. topic = relevant-but-limited context; leading = one huge prime
# from the whole corpus (reaches multi-MB).
set -euo pipefail
cd "$(dirname "$0")"

echo "########## TOPIC + exclude-source (relevant context) ##########"
for W in 256000 1000000 4000000; do
  python3 hellaswag.py --prime-mode topic --exclude-source --algo zstd --window "$W"
  echo
done

echo "########## LEADING (one giant generic prime) ##########"
for W in 64000 256000 1000000 4000000; do
  python3 hellaswag.py --prime-mode leading --algo zstd --window "$W"
  echo
done
echo "DONE"
