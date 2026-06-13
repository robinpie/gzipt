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
"""

from __future__ import annotations

import math
import random
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


def _repeats(s: bytes, seen: set[bytes], n: int) -> bool:
    """Whether the final ``n``-gram of ``s`` already occurred.

    True if it is in ``seen`` (committed output) or appears earlier within ``s``
    itself (an in-progress self-loop). This is the signal the beam is penalized
    by, so looping continuations get pruned during the search.
    """
    if n <= 1 or len(s) < n:
        return False
    ng = s[-n:]
    return ng in seen or s[:-1].find(ng) != -1


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
    no_repeat: int = 8,
    repeat_penalty: float = 4.0,
    level: int = 9,
    workers: int = 1,
    alphabet: tuple[int, ...] | None = None,
    seed: int | None = None,
) -> bytes:
    """Generate ``length`` bytes continuing ``prompt``, primed by ``corpus``.

    Each step beam-searches ``horizon`` bytes deep and commits that span, then
    re-plans. Candidates are scored by compressed length plus ``repeat_penalty``
    bytes for every ``no_repeat``-gram they would repeat (from earlier output or
    from themselves) — this is what dissolves the verbatim loops, by making the
    beam prefer continuations it has not used. ``temperature == 0`` commits the
    best-scoring span; a positive temperature samples the final beams. Only the
    last ``tail`` bytes of output stay in the scoring context, so gzip cannot
    directly copy its own older history.
    """
    rng = random.Random(seed)
    if alphabet is None:
        alphabet = corpus_alphabet(corpus + prompt)
    corpus_window = corpus[:window]
    pool = ThreadPoolExecutor(workers) if workers > 1 else None

    out = bytearray()
    seen: set[bytes] = set()           # committed n-grams, for the loop guard
    try:
        while len(out) < length:
            # gzip sees the corpus window plus only the recent tail of output.
            recent = (bytes(prompt) + bytes(out))[-tail:]
            ctx = corpus_window + recent
            # n-grams may span the boundary between committed output and the span.
            edge = bytes(out[-(no_repeat - 1):]) if no_repeat > 1 else b""

            # Each beam carries its accumulated repeat penalty alongside the bytes.
            beams: list[tuple[bytes, float]] = [(b"", 0.0)]
            beam_scores: list[float] = [0.0]
            for _ in range(horizon):
                seqs: list[bytes] = []
                penalties: list[float] = []
                for seq, pen in beams:
                    for b in alphabet:
                        nxt = seq + bytes([b])
                        extra = repeat_penalty if _repeats(edge + nxt, seen, no_repeat) else 0.0
                        seqs.append(nxt)
                        penalties.append(pen + extra)
                lens = candidate_lengths(ctx, seqs, level=level, pool=pool)
                scores = [lens[i] + penalties[i] for i in range(len(seqs))]
                order = sorted(range(len(seqs)), key=scores.__getitem__)[:beam_width]
                beams = [(seqs[i], penalties[i]) for i in order]
                beam_scores = [scores[i] for i in order]

            # beams are sorted ascending by score: beams[0] is best.
            if temperature <= 0:
                span = beams[0][0]
            else:
                best = beam_scores[0]
                weights = [math.exp(-(s - best) / temperature) for s in beam_scores]
                span = rng.choices([b for b, _ in beams], weights=weights, k=1)[0]

            if no_repeat > 1:
                stream = edge + span
                for i in range(len(stream) - no_repeat + 1):
                    seen.add(stream[i:i + no_repeat])
            out += span
    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    return bytes(out[:length])
