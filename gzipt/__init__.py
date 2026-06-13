"""gzipt — gzip as a language model, via beam-search generation.

Public API:
    generate           beam-search text generation primed by a corpus
    candidate_lengths  clone-accelerated compressed lengths of context+sequences
    corpus_alphabet    the byte values that occur in a corpus
"""

from __future__ import annotations

from .model import (
    DEFAULT_WINDOW,
    GZIP_WINDOW,
    candidate_lengths,
    corpus_alphabet,
    generate,
)

__version__ = "0.3.0"

__all__ = [
    "generate",
    "candidate_lengths",
    "corpus_alphabet",
    "DEFAULT_WINDOW",
    "GZIP_WINDOW",
    "__version__",
]
