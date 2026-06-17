"""Experiments on making non-zlib compressors usable as the model.

Idea 2: measure only the *marginal* compressed cost of a candidate by loading
the context as a zstd prefix, instead of recompressing context+candidate whole
(which buries the 1-byte signal under whole-stream rounding).

Idea 1: score multi-byte *spans* instead of single bytes, so the length delta
between candidates is big enough to clear the quantization floor. Candidate
spans are the byte sequences that actually follow the current context in the
corpus (an n-gram lookup); the compressor's only job is to rank them.
"""

from __future__ import annotations

import math
import random
import time
import zlib
from compression import zstd

import gzipt


# --- Idea 2: marginal-cost scorers -----------------------------------------

def zstd_prefix_scorer(level: int = 9):
    """Compress only the candidate against the context-as-prefix.

    The returned length is the candidate's marginal cost given the context,
    not len(compress(context+candidate)). That is the zstd equivalent of zlib's
    compressobj.copy() trick.
    """
    def score(context, sequences, pool):
        prefix = zstd.ZstdDict(context, is_raw=True).as_prefix

        def length_for(seq):
            c = zstd.ZstdCompressor(zstd_dict=prefix)
            return len(c.compress(seq) + c.flush())

        return [length_for(s) for s in sequences]

    return score


def zstd_naive_scorer(level: int = 9):
    """Baseline: recompress the whole context+candidate (the failing path)."""
    def score(context, sequences, pool):
        return [len(zstd.compress(context + s, level)) for s in sequences]

    return score


# --- Idea 1: span generation -----------------------------------------------

def build_ngram_index(corpus: bytes, key: int) -> dict[bytes, list[int]]:
    idx: dict[bytes, list[int]] = {}
    for i in range(len(corpus) - key):
        idx.setdefault(corpus[i : i + key], []).append(i + key)
    return idx


def generate_spans(
    corpus: bytes,
    prompt: bytes,
    length: int,
    *,
    score,
    window: int = 30000,
    span_len: int = 8,
    key: int = 8,
    max_cands: int = 96,
    tail: int = 80,
    temperature: float = 0.4,
    seed: int = 3,
) -> bytes:
    """Generate by ranking corpus-derived continuation spans with `score`."""
    rng = random.Random(seed)
    cw = corpus[:window]
    indexes = {k: build_ngram_index(cw, k) for k in range(1, key + 1)}
    out = bytearray()
    while len(out) < length:
        recent = (prompt + bytes(out))[-tail:]
        ctx = cw + recent
        # Back off from the longest key until we find corpus continuations.
        cands: list[bytes] = []
        for k in range(min(key, len(recent)), 0, -1):
            starts = indexes[k].get(recent[-k:], [])
            seen = set()
            for s in starts:
                span = cw[s : s + span_len]
                if len(span) == span_len and span not in seen:
                    seen.add(span)
                    cands.append(span)
                if len(cands) >= max_cands:
                    break
            if cands:
                break
        if not cands:
            break
        lens = score(ctx, cands, None)
        best = min(lens)
        weights = [math.exp(-(L - best) / temperature) for L in lens]
        out += rng.choices(cands, weights=weights, k=1)[0]
    return bytes(out[:length])


# --- driver ----------------------------------------------------------------

def show(title: str, fn):
    t = time.perf_counter()
    text = fn()
    dt = time.perf_counter() - t
    print(f"\n===== {title}   ({dt:.1f}s) =====")
    print(text.decode("utf-8", errors="replace"))


def main():
    corpus = open("data/tinyshakespeare.txt", "rb").read()
    prompt = b"MENENIUS:\n"
    N = 120

    # Idea 2: zstd, naive recompress vs prefix-marginal, same beam search.
    show("IDEA 2  zstd NAIVE recompress (baseline)", lambda: gzipt.generate(
        corpus, prompt, N, window=8000, horizon=12, beam_width=16,
        temperature=0.5, scorer=zstd_naive_scorer()))
    show("IDEA 2  zstd PREFIX marginal", lambda: gzipt.generate(
        corpus, prompt, N, window=8000, horizon=12, beam_width=16,
        temperature=0.5, scorer=zstd_prefix_scorer()))
    show("IDEA 2  zlib clone (reference)", lambda: gzipt.generate(
        corpus, prompt, N, window=8000, horizon=12, beam_width=16,
        temperature=0.5, algo="zlib"))

    # Idea 1: score multi-byte spans; even naive recompress should rank well now.
    import bz2 as _bz2
    bz2_naive = lambda ctx, seqs, pool: [len(_bz2.compress(ctx + s, 9)) for s in seqs]
    show("IDEA 1  span-rank, bz2 naive recompress", lambda: generate_spans(
        corpus, prompt, N, score=bz2_naive, window=30000, span_len=8))
    show("IDEA 1  span-rank, zstd naive recompress", lambda: generate_spans(
        corpus, prompt, N, score=zstd_naive_scorer(), window=30000, span_len=8))


if __name__ == "__main__":
    main()
