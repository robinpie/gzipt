"""PPM as a fractional-bit code-length oracle for gzipt.

The whole project uses ``len(compress(x))`` as a stand-in for -log P(x). gzip's
problem is that the output is an integer number of *bytes*: sub-byte preferences
round away, so the beam goes blind (the "quantization floor") and only zlib's
clone trick scrapes by. But we never decompress here -- we only read a code
length -- so we are free to replace gzip with the model it was secretly
approximating: an adaptive context model whose code length is the exact
-sum log2 P, a real number with no quantization.

This implements PPM-C (Cleary/Witten, Moffat) over bytes: an order-N context
model with escape-based back-off. ``make_scorer`` returns a drop-in scorer for
``gzipt.generate`` (via its ``scorer=`` hook); ``hellaswag_scorer`` plugs into
``hellaswag.py``. Counts are taken from the priming context only and frozen
(not updated by the candidate bytes), mirroring how the corpus -- not the
model's own output -- is the only "knowledge".
"""

from __future__ import annotations

import argparse
import math
import sys


class Model:
    __slots__ = ("counts", "order")

    def __init__(self, counts, order):
        # counts[k]: dict[bytes k-gram -> (dict[int sym -> int count], total)]
        self.counts = counts
        self.order = order


def build(data: bytes, order: int) -> Model:
    """Count, for every context up to length ``order``, the bytes that follow."""
    raw = [dict() for _ in range(order + 1)]
    n = len(data)
    for i in range(n):
        sym = data[i]
        kmax = order if i >= order else i
        for k in range(kmax + 1):
            ctx = data[i - k:i]
            d = raw[k].get(ctx)
            if d is None:
                raw[k][ctx] = {sym: 1}
            else:
                d[sym] = d.get(sym, 0) + 1
    counts = [{ctx: (d, sum(d.values())) for ctx, d in level.items()} for level in raw]
    return Model(counts, order)


_LOG2 = 1.0 / math.log(2.0)


def _sym_bits(model: Model, hist: bytes, sym: int) -> float:
    """PPM-C code length (bits) for ``sym`` given history ``hist``."""
    counts = model.counts
    total = 0.0
    kmax = model.order if len(hist) >= model.order else len(hist)
    for k in range(kmax, -1, -1):
        ctx = hist[len(hist) - k:] if k else b""
        entry = counts[k].get(ctx)
        if entry is None:
            continue  # context never seen at this order -> escape is free
        d, T = entry
        D = len(d)
        denom = T + D
        c = d.get(sym)
        if c is not None:
            return total - math.log(c / denom) * _LOG2
        total -= math.log(D / denom) * _LOG2  # escape down one order
    return total - math.log(1.0 / 256) * _LOG2  # order -1: uniform over bytes


def code_bits(model: Model, history: bytes, seq: bytes) -> float:
    """Total bits to code ``seq`` after ``history`` (frozen counts)."""
    order = model.order
    hist = history[-order:] if order else b""
    bits = 0.0
    for sym in seq:
        bits += _sym_bits(model, hist, sym)
        if order:
            hist = (hist + bytes((sym,)))[-order:]
    return bits


def make_scorer(corpus_window: bytes, order: int = 6):
    """Drop-in scorer for ``gzipt.generate`` (built once from the corpus window).

    ``gzipt`` calls ``score(corpus_window + recent, candidates, pool)``; we model
    on the fixed corpus window and condition only on ``recent`` (the visible
    tail), so the model never trains on its own generated output.
    """
    model = build(corpus_window, order)
    base = len(corpus_window)

    def score(context, sequences, pool=None):
        recent = context[base:]
        return [code_bits(model, recent, seq) for seq in sequences]

    return score


def hellaswag_scorer(ctx: bytes, endings, level: int = 9, order: int = 6):
    """Scorer matching hellaswag.py's interface: marginal bits per ending."""
    model = build(ctx, order)
    return [code_bits(model, ctx, eb) for eb in endings]


def main(argv=None) -> int:
    import gzipt

    p = argparse.ArgumentParser(
        prog="ppm",
        description="Generate text with gzipt's beam search but a PPM "
                    "fractional-bit code length instead of a byte compressor.")
    p.add_argument("--corpus", "--prime", dest="corpus", metavar="FILE")
    p.add_argument("--prompt", default="")
    p.add_argument("--length", type=int, default=200)
    p.add_argument("--order", type=int, default=6, help="PPM context order (default 6)")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--beam-width", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy. Note: costs are in bits, so a useful "
                        "sampling temperature is larger than gzip's (try 1-4).")
    p.add_argument("--tail", type=int, default=80)
    p.add_argument("--window", type=int, default=gzipt.DEFAULT_WINDOW)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args(argv)

    corpus = b""
    if args.corpus:
        with open(args.corpus, "rb") as fh:
            corpus = fh.read()
    prompt = args.prompt.encode("utf-8", errors="replace")
    corpus_window = corpus[:args.window]
    scorer = make_scorer(corpus_window, args.order)

    out = gzipt.generate(
        corpus, prompt, args.length,
        window=args.window, horizon=args.horizon, beam_width=args.beam_width,
        temperature=args.temperature, tail=args.tail, workers=1,
        seed=args.seed, scorer=scorer,
    )
    text = (prompt + out).decode("utf-8", errors="replace")
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
