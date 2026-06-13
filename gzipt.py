"""gzip as a language model: beam-search text generation by compression.

Compression is prediction. A continuation that gzip "expected" — because it
echoes text already in its window — compresses to almost nothing, so *compressed
length is a score*. We generate by beam search over byte sequences: keep the
``beam_width`` most-compressible partial continuations, extend each by one byte,
prune, and after ``horizon`` bytes commit the best span and repeat.

The corpus primes gzip's 32 KiB window and is the model's only knowledge; gzip is
just a length oracle over it. Mechanically this makes gzip a fuzzy n-gram over the
corpus: DEFLATE predicts the next bytes by matching the recent context against the
window, which is exactly what an n-gram does. So generation reads like recombined
corpus fragments — fluent locally, a collage globally.

Two things make it practical:

* **encoder-state cloning** — the shared context is compressed once and
  ``.copy()``-ed per candidate, so each pays only for its own few bytes. This is
  byte-for-byte identical to ``len(zlib.compress(context + seq))`` but far cheaper
  (the match search over the context runs once, not once per candidate). zlib is
  the only stdlib compressor that exposes this.
* the candidate byte set is restricted to bytes that occur in the corpus, and
  candidate scoring is optionally threaded (``zlib`` releases the GIL).

Run ``python gzipt.py --help`` (or ``uv run gzipt --help``) for the CLI.
"""

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


def candidate_lengths(
    context: bytes,
    sequences: list[bytes],
    *,
    level: int = 9,
    pool: ThreadPoolExecutor | None = None,
) -> list[int]:
    """Compressed length of ``context + seq`` for each seq, sharing the context.

    Compresses ``context`` once into a ``compressobj``, then clones its encoder
    state per candidate and feeds only that candidate. Identical to
    ``len(zlib.compress(context + seq, level))`` for each seq, but the expensive
    match search over ``context`` happens a single time.
    """
    base = zlib.compressobj(level)
    head = len(base.compress(context))

    def length_for(seq: bytes) -> int:
        clone = base.copy()
        return head + len(clone.compress(seq) + clone.flush(zlib.Z_FINISH))

    if pool is not None:
        return list(pool.map(length_for, sequences))
    return [length_for(seq) for seq in sequences]


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
    workers: int = 1,
    alphabet: tuple[int, ...] | None = None,
    seed: int | None = None,
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
                lens = candidate_lengths(ctx, cand, level=level, pool=pool)
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

    # Beam-search knobs.
    p.add_argument("--horizon", type=int, default=24,
                   help="bytes looked ahead and committed per beam search "
                        "(default 24; too small collapses, too large just copies)")
    p.add_argument("--beam-width", type=int, default=32,
                   help="partial continuations kept each step (default 32)")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="0 = most-compressible span; >0 samples the final beams "
                        "(default 0.5)")
    p.add_argument("--tail", type=int, default=80,
                   help="generated bytes kept in the scoring context; older output "
                        "is hidden to stop gzip copying itself (default 80)")

    # Compression / performance knobs.
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"corpus bytes shown to gzip, <=32768 (default {DEFAULT_WINDOW})")
    p.add_argument("--level", type=int, default=9, help="gzip level 0-9 (default 9)")
    p.add_argument("--workers", type=int, default=8,
                   help="threads for candidate scoring; zlib releases the GIL (default 8)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (for --temperature > 0)")

    args = p.parse_args(argv)

    corpus = b""
    if args.corpus:
        with open(args.corpus, "rb") as fh:
            corpus = fh.read()
    prompt = args.prompt.encode("utf-8", errors="replace")

    out = generate(
        corpus, prompt, args.length,
        window=args.window, horizon=args.horizon, beam_width=args.beam_width,
        temperature=args.temperature, tail=args.tail, level=args.level,
        workers=args.workers, seed=args.seed,
    )

    text = (prompt + out).decode("utf-8", errors="replace")
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
