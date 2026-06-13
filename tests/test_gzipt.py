"""Tests for gzipt: clone scoring, the corpus alphabet, and beam generation."""

from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor

from gzipt import candidate_lengths, corpus_alphabet, generate

CORPUS = b"the cat sat on the mat. the dog sat on the log. " * 40


# -- corpus alphabet --------------------------------------------------------

def test_corpus_alphabet_only_occurring_bytes():
    assert corpus_alphabet(b"abcabc") == (97, 98, 99)


def test_corpus_alphabet_empty_falls_back_to_all_bytes():
    assert corpus_alphabet(b"") == tuple(range(256))


# -- clone scoring (the speedup) --------------------------------------------

def test_candidate_lengths_match_zlib_exactly():
    ctx = CORPUS[:500]
    seqs = [b"x", b"the ", b"cat", b"\n", b"the cat"]
    got = candidate_lengths(ctx, seqs, level=9)
    for seq, g in zip(seqs, got):
        assert g == len(zlib.compress(ctx + seq, 9))


def test_candidate_lengths_threaded_matches_serial():
    ctx = CORPUS[:800]
    seqs = [bytes([b]) for b in range(80)]
    serial = candidate_lengths(ctx, seqs, pool=None)
    with ThreadPoolExecutor(4) as pool:
        threaded = candidate_lengths(ctx, seqs, pool=pool)
    assert serial == threaded


# -- generation -------------------------------------------------------------

_KW = dict(window=2048, horizon=8, beam_width=8)


def test_generate_exact_length():
    out = generate(CORPUS, b"the ", 50, **_KW)
    assert len(out) == 50


def test_generate_only_uses_corpus_alphabet():
    out = generate(CORPUS, b"the ", 40, **_KW)
    assert set(out) <= set(CORPUS + b"the ")


def test_generate_temperature0_is_deterministic():
    a = generate(CORPUS, b"the ", 40, temperature=0.0, **_KW)
    b = generate(CORPUS, b"the ", 40, temperature=0.0, **_KW)
    assert a == b


def test_generate_threading_matches_serial():
    a = generate(CORPUS, b"the ", 40, temperature=0.0, workers=1, **_KW)
    b = generate(CORPUS, b"the ", 40, temperature=0.0, workers=4, **_KW)
    assert a == b


def test_generate_sampling_is_seed_reproducible():
    a = generate(CORPUS, b"the ", 40, temperature=0.8, seed=3, **_KW)
    b = generate(CORPUS, b"the ", 40, temperature=0.8, seed=3, **_KW)
    assert a == b


def test_generate_copies_real_corpus_fragments():
    # gzip generation is recombination/copying, so a chunk of output should
    # appear verbatim somewhere in the corpus.
    out = generate(CORPUS, b"the ", 40, temperature=0.0, **_KW)
    assert any(out[i:i + 6] in CORPUS for i in range(len(out) - 6))
