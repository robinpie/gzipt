"""gzip as a language model: beam-search text generation by compression."""

from __future__ import annotations

import argparse
import math
import random
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

# DEFLATE's match window: context older than this is invisible to the matcher.
# Default just under it (leaving room for the recent-output tail), since a fuller
# window gives gzip more corpus to draw from and visibly less degenerate looping.
GZIP_WINDOW = 32768
DEFAULT_WINDOW = 30000


def corpus_alphabet(data: bytes) -> tuple[int, ...]:
    """Byte values that occur in ``data`` — the only ones worth generating.

    A byte absent from the corpus can never extend a match, so it could only ever
    be chosen on a tie; dropping the other ~180 values is a free speedup. Falls
    back to all 256 bytes for empty input.
    """
    return tuple(sorted(set(data))) or tuple(range(256))


# Each algorithm is a different implicit model. zlib (DEFLATE: LZ77 + Huffman,
# 32KB window) is the only one whose encoder state can be cloned mid-stream, so
# it gets a fast path. The rest expose no state-cloning API, so we must
# recompress ``context + seq`` per candidate — correct but far slower, which is
# why non-zlib backends want a smaller --window.
ALGORITHMS = ("zlib", "bz2", "lzma", "zstd", "brotli")

Scorer = "callable: (context, sequences, pool) -> list[int]"


def _one_shot_compressor(algo: str, level: int):
    """Return a ``bytes -> bytes`` compressor for a backend lacking state copy."""
    if algo == "bz2":
        import bz2
        lvl = min(max(level, 1), 9)
        return lambda data: bz2.compress(data, lvl)
    if algo == "lzma":
        import lzma
        return lambda data: lzma.compress(data, preset=min(max(level, 0), 9))
    if algo == "zstd":
        from compression import zstd
        return lambda data: zstd.compress(data, level)
    if algo == "brotli":
        import brotli
        q = min(max(level, 0), 11)
        return lambda data: brotli.compress(data, quality=q)
    raise ValueError(f"unknown algo: {algo!r} (choose from {', '.join(ALGORITHMS)})")


def make_scorer(algo: str = "zlib", level: int = 9):
    """Build a scorer: compressed length of ``context + seq`` for each candidate.

    For ``zlib`` the context is compressed once and the encoder state is cloned
    per candidate, so the match search over the context happens a single time.
    Every other backend recompresses the whole ``context + seq`` per candidate.
    Lower compressed length == better predicted, regardless of backend.
    """
    if algo == "zlib":
        def score(context, sequences, pool):
            base = zlib.compressobj(level)
            head = len(base.compress(context))

            def length_for(seq: bytes) -> int:
                clone = base.copy()
                return head + len(clone.compress(seq) + clone.flush(zlib.Z_FINISH))

            if pool is not None:
                return list(pool.map(length_for, sequences))
            return [length_for(seq) for seq in sequences]

        return score

    compress = _one_shot_compressor(algo, level)

    def score(context, sequences, pool):
        def length_for(seq: bytes) -> int:
            return len(compress(context + seq))

        if pool is not None:
            return list(pool.map(length_for, sequences))
        return [length_for(seq) for seq in sequences]

    return score


def generate(
    corpus: bytes,
    prompt: bytes,
    length: int,
    *,
    window: int = DEFAULT_WINDOW,
    horizon: int = 24,
    beam_width: int = 32,
    temperature: float = 0.5,
    tail: int = 80,
    level: int = 9,
    algo: str = "zlib",
    workers: int = 1,
    alphabet: tuple[int, ...] | None = None,
    seed: int | None = None,
    scorer=None,
) -> bytes:
    """Generate ``length`` bytes continuing ``prompt``, primed by ``corpus``.

    Each step beam-searches ``horizon`` bytes deep and commits that span, then
    re-plans. Candidates are scored purely by compressed length. ``temperature
    == 0`` commits the most-compressible span; a positive temperature samples the
    final beams. Only the last ``tail`` bytes of output stay in the scoring
    context, so gzip cannot directly copy its own older history.
    """
    rng = random.Random(seed)
    if alphabet is None:
        alphabet = corpus_alphabet(corpus + prompt)
    corpus_window = corpus[:window]
    score = scorer if scorer is not None else make_scorer(algo, level)
    pool = ThreadPoolExecutor(workers) if workers > 1 else None

    out = bytearray()
    try:
        while len(out) < length:
            # gzip sees the corpus window plus only the recent tail of output.
            recent = (bytes(prompt) + bytes(out))[-tail:]
            ctx = corpus_window + recent

            beams: list[bytes] = [b""]
            beam_lens: list[int] = [0]
            for _ in range(horizon):
                cand = [h + bytes([b]) for h in beams for b in alphabet]
                lens = score(ctx, cand, pool)
                order = sorted(range(len(cand)), key=lens.__getitem__)[:beam_width]
                beams = [cand[i] for i in order]
                beam_lens = [lens[i] for i in order]

            # beams are sorted ascending by length: beams[0] is most compressible.
            if temperature <= 0:
                span = beams[0]
            else:
                best = beam_lens[0]
                weights = [math.exp(-(L - best) / temperature) for L in beam_lens]
                span = rng.choices(beams, weights=weights, k=1)[0]
            out += span
    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    return bytes(out[:length])


def _ngram_index(corpus: bytes, key: int) -> dict[bytes, list[int]]:
    """Map every ``key``-byte substring to the offsets where its sequel begins."""
    idx: dict[bytes, list[int]] = {}
    for i in range(len(corpus) - key):
        idx.setdefault(corpus[i : i + key], []).append(i + key)
    return idx


def generate_spans(
    corpus: bytes,
    prompt: bytes,
    length: int,
    *,
    window: int = DEFAULT_WINDOW,
    span_len: int = 8,
    key: int = 8,
    max_cands: int = 96,
    temperature: float = 0.4,
    tail: int = 80,
    level: int = 9,
    algo: str = "zlib",
    workers: int = 1,
    seed: int | None = None,
) -> bytes:
    """Secondary mode: rank corpus-derived continuation *spans* by compression.

    Unlike the primary byte-level beam (:func:`generate`), candidates here are
    whole ``span_len``-byte sequences that actually follow the recent context
    somewhere in the corpus (found by backing off from a ``key``-byte n-gram).
    Scoring multi-byte spans clears the quantization floor that blinds coarse
    compressors on single-byte differences, so backends like ``bz2``/``lzma``
    that degenerate in byte mode produce coherent text here. The trade-off: the
    compressor only *re-ranks* real corpus text rather than inventing bytes, so
    output copies more and emerges less.
    """
    rng = random.Random(seed)
    corpus_window = corpus[:window]
    score = make_scorer(algo, level)
    indexes = {k: _ngram_index(corpus_window, k) for k in range(1, key + 1)}
    pool = ThreadPoolExecutor(workers) if workers > 1 else None

    out = bytearray()
    try:
        while len(out) < length:
            recent = (bytes(prompt) + bytes(out))[-tail:]
            ctx = corpus_window + recent

            # Back off from the longest key until the corpus offers continuations.
            cands: list[bytes] = []
            for k in range(min(key, len(recent)), 0, -1):
                seen: set[bytes] = set()
                for s in indexes[k].get(recent[-k:], []):
                    span = corpus_window[s : s + span_len]
                    if len(span) == span_len and span not in seen:
                        seen.add(span)
                        cands.append(span)
                    if len(cands) >= max_cands:
                        break
                if cands:
                    break
            if not cands:
                break

            lens = score(ctx, cands, pool)
            if temperature <= 0:
                span = cands[min(range(len(cands)), key=lens.__getitem__)]
            else:
                best = min(lens)
                weights = [math.exp(-(L - best) / temperature) for L in lens]
                span = rng.choices(cands, weights=weights, k=1)[0]
            out += span
    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    return bytes(out[:length])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="gzipt",
        description="Generate text using gzip as the language model: beam search "
                    "over a corpus that primes the compressor's window.",
    )
    p.add_argument("--corpus", "--prime", dest="corpus", metavar="FILE",
                   help="text file that primes gzip's window — the model's only "
                        "knowledge (generation copies/recombines from it)")
    p.add_argument("--prompt", default="", help="seed text to continue")
    p.add_argument("--length", type=int, default=200, help="bytes to generate (default 200)")
    p.add_argument("--mode", choices=("bytes", "spans"), default="bytes",
                   help="bytes = primary byte-level beam search (default); spans = "
                        "secondary mode that re-ranks corpus n-gram continuations, "
                        "letting coarse compressors (bz2/lzma) produce coherent text")

    # Beam-search knobs.
    p.add_argument("--horizon", type=int, default=24,
                   help="bytes looked ahead and committed per beam search")
    p.add_argument("--beam-width", type=int, default=32,
                   help="partial continuations kept each step (default 32)")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="0 = most-compressible span; >0 samples the final beams "
                        "(default 0.5)")
    p.add_argument("--tail", type=int, default=80,
                   help="generated bytes kept in the scoring context; older output "
                        "is hidden to stop gzip copying itself (default 80)")

    # Secondary --mode spans knobs (ignored in the default bytes mode).
    p.add_argument("--span-len", type=int, default=8,
                   help="[spans mode] bytes committed per ranked span; higher copies "
                        "more verbatim, lower recombines more (default 8)")
    p.add_argument("--key", type=int, default=8,
                   help="[spans mode] longest n-gram used to find corpus "
                        "continuations, backing off when absent (default 8)")

    # Compression / performance knobs.
    p.add_argument("--algo", choices=ALGORITHMS, default="zlib",
                   help="compressor used as the model; zlib has a fast path, the "
                        "rest recompress per candidate so use a smaller --window "
                        "(default zlib)")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"corpus bytes shown to gzip, <=32768 (default {DEFAULT_WINDOW})")
    p.add_argument("--level", type=int, default=9, help="gzip level 0-9 (default 9)")
    p.add_argument("--workers", type=int, default=8,
                   help="threads for candidate scoring; zlib releases the GIL (default 8)")
    p.add_argument("--seed", type=int, default=3, help="RNG seed (for --temperature > 0)")

    args = p.parse_args(argv)

    corpus = b""
    if args.corpus:
        with open(args.corpus, "rb") as fh:
            corpus = fh.read()
    prompt = args.prompt.encode("utf-8", errors="replace")

    if args.mode == "spans":
        out = generate_spans(
            corpus, prompt, args.length,
            window=args.window, span_len=args.span_len, key=args.key,
            temperature=args.temperature, tail=args.tail, level=args.level,
            algo=args.algo, workers=args.workers, seed=args.seed,
        )
    else:
        out = generate(
            corpus, prompt, args.length,
            window=args.window, horizon=args.horizon, beam_width=args.beam_width,
            temperature=args.temperature, tail=args.tail, level=args.level,
            algo=args.algo, workers=args.workers, seed=args.seed,
        )

    text = (prompt + out).decode("utf-8", errors="replace")
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
