"""Can gzipt's compression-as-prediction beat chance (25%) on HellaSwag?

No generation: for each of the 4 endings, measure the marginal compressed cost
of the ending given the context (zlib's clone trick), and predict the cheapest
one. Optionally prime the compressor with a disjoint block of examples first.
"""

from __future__ import annotations

import argparse
import json
import lzma
import time
import zlib
from compression import zstd
from concurrent.futures import ThreadPoolExecutor


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def context_of(r: dict) -> str:
    return f"{r['activity_label']}: {r['ctx']} "


def example_text(r: dict) -> bytes:
    """The (context + correct ending) line used to prime the compressor."""
    return (context_of(r) + r["endings"][r["label"]] + "\n").encode("utf-8", "replace")


def build_prime(rows: list[dict], window: int) -> tuple[bytes, int]:
    """LEADING mode: one shared prime from the first rows up to `window`.

    Returns the prime and how many leading rows it consumed (those rows are then
    held out of the test set to avoid leakage).
    """
    out = bytearray()
    n = 0
    for r in rows:
        t = example_text(r)
        if len(out) + len(t) > window:
            break
        out += t
        n += 1
    return bytes(out), n


def build_topic_index(rows: list[dict]) -> dict[str, list[int]]:
    """Map each activity_label to the indices of its examples."""
    idx: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        idx.setdefault(r["activity_label"], []).append(i)
    return idx


def topic_prime(rows: list[dict], peers: list[int], skip: int, window: int,
                skip_source: str | None = None) -> bytes:
    """TOPIC mode: per-example prime from OTHER same-activity examples.

    The scored example (``skip``) is always excluded. HellaSwag items are derived
    from shared source paragraphs, so when ``skip_source`` is given we also drop
    every peer with the same ``source_id`` — otherwise a sibling item can leak the
    test item's actual continuation into the prime.
    """
    out = bytearray()
    for j in peers:
        if j == skip or (skip_source is not None and rows[j]["source_id"] == skip_source):
            continue
        t = example_text(rows[j])
        if len(out) + len(t) > window:
            break
        out += t
    return bytes(out)


# --- marginal-cost scorers per compressor ---------------------------------
# All return raw_marginal[4]: the added compressed cost of each ending given the
# context. zlib's 32KB match window can't see far-back topic prime; lzma (large
# dictionary) and zstd (long-distance matching) can, which is the whole point.

def _zlib_scorer(ctx: bytes, endings: list[bytes], level: int = 9):
    base = zlib.compressobj(level)
    base.compress(ctx)
    out = []
    for eb in endings:
        clone = base.copy()
        out.append(len(clone.compress(eb) + clone.flush(zlib.Z_FINISH)))
    return out


def _lzma_scorer(ctx: bytes, endings: list[bytes], level: int = 9):
    # preset 1 + an 8MB dictionary: fast, but easily large enough to see the
    # whole (<=2MB) topic prime — so far-back matches stay reachable.
    filt = [{"id": lzma.FILTER_LZMA2, "preset": 1, "dict_size": 1 << 23}]
    comp = lambda d: lzma.compress(d, format=lzma.FORMAT_RAW, filters=filt)
    head = len(comp(ctx))
    return [len(comp(ctx + eb)) - head for eb in endings]


def _zstd_scorer(ctx: bytes, endings: list[bytes], level: int = 9):
    CP = zstd.CompressionParameter
    opts = {CP.compression_level: level, CP.enable_long_distance_matching: 1,
            CP.window_log: 23}
    comp = lambda d: zstd.compress(d, options=opts)
    head = len(comp(ctx))
    return [len(comp(ctx + eb)) - head for eb in endings]


SCORERS = {"zlib": _zlib_scorer, "lzma": _lzma_scorer, "zstd": _zstd_scorer}


def score_example(prime: bytes, r: dict, scorer, level: int = 9):
    """Return (raw_marginal[4], char_lens[4]) for the four endings."""
    ctx = prime + context_of(r).encode("utf-8", "replace")
    endings = [e.encode("utf-8", "replace") for e in r["endings"]]
    raw = scorer(ctx, endings, level)
    char_lens = [max(1, len(eb)) for eb in endings]
    return raw, char_lens


SCHEMES = {
    "raw": lambda raw, lens: raw,
    "per_char": lambda raw, lens: [r / l for r, l in zip(raw, lens)],
}


def evaluate(test: list[tuple[bytes, dict]], scorer, level: int, workers: int):
    """`test` is a list of (per-example prime, row) pairs."""
    def work(item):
        prime, r = item
        raw, lens = score_example(prime, r, scorer, level)
        return {name: min(range(4), key=fn(raw, lens).__getitem__)
                for name, fn in SCHEMES.items()}, r["label"]

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(work, test))

    acc = {name: 0 for name in SCHEMES}
    for preds, label in results:
        for name, p in preds.items():
            acc[name] += (p == label)
    return {name: c / len(test) for name, c in acc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/hellaswag_val.jsonl")
    p.add_argument("--prime-mode", choices=("leading", "topic"), default="leading",
                   help="leading = one shared prime from the first examples; "
                        "topic = per-example prime from other same-activity "
                        "examples (the strong setting, ~35%%)")
    p.add_argument("--algo", choices=tuple(SCORERS), default="zlib",
                   help="scoring compressor; zlib has a 32KB match window, lzma "
                        "(big dict) and zstd (long-distance matching) can reach "
                        "far-back topic prime (default zlib)")
    p.add_argument("--window", type=int, default=16000, help="prime size in bytes (0 = none)")
    p.add_argument("--exclude-source", action="store_true",
                   help="[topic mode] drop peers sharing the test item's source_id "
                        "(prevents sibling-paragraph leakage)")
    p.add_argument("--level", type=int, default=9)
    p.add_argument("--limit", type=int, default=0, help="cap #test examples (0 = all)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    rows = load(args.data)

    if args.prime_mode == "topic":
        by_act = build_topic_index(rows)
        items = [(topic_prime(rows, by_act[r["activity_label"]], i, args.window,
                              r["source_id"] if args.exclude_source else None), r)
                 for i, r in enumerate(rows)]
        desc = (f"topic-matched prime <= {args.window}B/example"
                + (" [source-excluded]" if args.exclude_source else ""))
    else:
        prime, n_prime = build_prime(rows, args.window) if args.window else (b"", 0)
        items = [(prime, r) for r in rows[n_prime:]]
        desc = f"leading prime {len(prime)}B / {n_prime} examples"

    if args.limit:
        items = items[: args.limit]

    print(f"algo={args.algo} | {desc} | test: {len(items)} examples")
    t = time.perf_counter()
    acc = evaluate(items, SCORERS[args.algo], args.level, args.workers)
    dt = time.perf_counter() - t
    print(f"({dt:.1f}s, chance = 25.0%)")
    for name, a in acc.items():
        print(f"  {name:9} {a * 100:5.2f}%")


if __name__ == "__main__":
    main()
