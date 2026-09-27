# gzipt — C engine

A C port of the **zlib engine** in `../gzipt.py`: byte-level beam search
(`generate`) and corpus n-gram span re-ranking (`generate_spans`). The Python
version stays as the reference and keeps the analysis scripts (HellaSwag,
retrieval, SQL, CSS) that depend on Python libraries.

## Build & run

```sh
make                 # -> ./gzipt        (OpenMP, parallel candidate scoring)
make gzipt-st        # -> ./gzipt-st     (single-threaded, for clean timing)

./gzipt --corpus ../data/tinyshakespeare.txt --prompt $'MENENIUS:\n' --length 200
```

The CLI mirrors the Python flags: `--corpus/--prime --prompt --length --mode
{bytes,spans} --horizon --beam-width --temperature --tail --span-len --key
--window --level --workers --seed`. `--algo` accepts only `zlib` (the other
backends never escaped the quantization floor in byte mode anyway).
`--workers` sets the OpenMP thread count (or use `OMP_NUM_THREADS`).

## Bit-exact with the Python version

`compare.sh` checks this and then times both:

```sh
./compare.sh
```

Output is **byte-for-byte identical** to `../gzipt.py`, in every mode:

- Same system `libz` (1.3.2) → identical compressed lengths → identical scores.
- Stable `(length, index)` ordering matches Python's stable sort.
- For `--temperature > 0`, the sampler reproduces CPython's MT19937 +
  `random.choices` (seeded), so even sampled output matches.

## Speed

Default settings, `--length 200`, this 18-core box:

| build               | wall   | cpu    |
|---------------------|--------|--------|
| Python (8 workers)  | ~14 s  | ~73 s  |
| C, single-thread    | ~34 s  | ~34 s  |
| C, OpenMP (8 thr.)  | ~11 s  | ~69 s  |

The headline: **C is only ~2× faster per-CPU, not the 20–100× you'd expect from
"port Python to C."** Why — the workload is not interpreter-bound. The clone
trick (`compressobj.copy()` / `deflateCopy`) duplicates zlib's full ~300 KB
encoder state (32 KB window + hash tables) **for every candidate**, then
compresses a handful of bytes. That memcpy dominates, and it is the *same*
`libz` call from both languages. Python's per-candidate overhead (object
churn, thread-pool dispatch) is the only thing C removes, hence ~2×.

Two C-specific tunings that mattered:

- **Per-thread bump arena for zlib's allocations.** A naive port mallocs/frees
  the encoder state per candidate; under threads the allocator lock serialises
  everything and OpenMP actually ran *slower* than Python. Routing
  `zalloc`/`zfree` through a per-thread arena (free = no-op, reset between
  candidates) removed all per-candidate malloc traffic and fixed scaling.
- **`-O2` vs `-O3` is a wash** (19.24 s vs 19.72 s single-thread, within noise),
  because the hot loop lives in the precompiled `libz.so`, not in our code.

So the real lever isn't the language — it's the algorithm. The deflateCopy
memcpy is inherent to the clone trick; beating it would mean a different
scoring scheme (e.g. a custom DEFLATE that can checkpoint/rewind cheaply),
not just C.
