"""Retrieval-by-compression for the HellaSwag prime.

Instead of selecting prime examples by the gold `activity_label`, retrieve the
peers whose context is *cheapest to encode given the test context* — i.e. the
most compression-similar ones (an asymmetric NCD, computed with zlib's clone
trick so it's one cheap op per peer). Then score the 4 endings with the
leakage-free winner from §3b: zstd + per_char. Always excludes same-source peers.

Compares, on the same subsample: retrieval prime vs gold-activity prime.
"""

from __future__ import annotations

import argparse
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import hellaswag as H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/hellaswag_val.jsonl")
    p.add_argument("--window", type=int, default=8000)
    p.add_argument("--sub", type=int, default=1000, help="evenly-spaced test subsample")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    rows = H.load(args.data)
    N = len(rows)
    ctxb = [H.context_of(r).encode("utf-8", "replace") for r in rows]
    lineb = [H.example_text(r) for r in rows]
    src = [r["source_id"] for r in rows]
    act = [r["activity_label"] for r in rows]
    by_act = H.build_topic_index(rows)

    step = max(1, N // args.sub)
    test_idx = list(range(0, N, step))[: args.sub]
    zstd = H.SCORERS["zstd"]

    def fill(order, skip_i):
        """Pack peers (in given index order) into a prime up to --window."""
        out = bytearray()
        used = []
        for j in order:
            if j == skip_i or src[j] == src[skip_i]:
                continue
            if len(out) + len(lineb[j]) > args.window:
                continue
            out += lineb[j]
            used.append(j)
            if len(out) >= args.window - 40:
                break
        return bytes(out), used

    def retrieve_order(i):
        """Peer indices sorted by ascending *bits-per-char* to encode given ctx i.

        Per-char normalization removes the bias toward short peer contexts, so
        this ranks by similarity rather than brevity.
        """
        base = zlib.compressobj(6)
        base.compress(ctxb[i])
        costs = []
        for j in range(N):
            c = base.copy()
            m = len(c.compress(ctxb[j]) + c.flush(zlib.Z_FINISH))
            costs.append((m / max(1, len(ctxb[j])), j))
        costs.sort()
        return [j for _, j in costs]

    def per_char_pred(prime, r):
        raw, lens = H.score_example(prime, r, zstd)
        return min(range(4), key=lambda k: raw[k] / lens[k])

    def eval_retrieval(i):
        prime, used = fill(retrieve_order(i), i)
        # how often does compression retrieval rediscover the gold activity?
        same = sum(act[j] == act[i] for j in used)
        ok = per_char_pred(prime, rows[i]) == rows[i]["label"]
        return ok, (same / len(used) if used else 0.0)

    def eval_topic(i):
        prime, used = fill(by_act[act[i]], i)
        return per_char_pred(prime, rows[i]) == rows[i]["label"]

    print(f"subsample: {len(test_idx)} / {N} | window {args.window} | zstd+per_char, source-excluded")

    t = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        topic = list(pool.map(eval_topic, test_idx))
    print(f"  gold-activity prime : {sum(topic) / len(topic) * 100:5.2f}%   ({time.perf_counter() - t:.0f}s)")

    t = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        res = list(pool.map(eval_retrieval, test_idx))
    acc = sum(ok for ok, _ in res) / len(res)
    overlap = sum(o for _, o in res) / len(res)
    print(f"  retrieval prime     : {acc * 100:5.2f}%   ({time.perf_counter() - t:.0f}s)")
    print(f"  (retrieved peers sharing test's gold activity: {overlap * 100:.0f}%)")


if __name__ == "__main__":
    main()
