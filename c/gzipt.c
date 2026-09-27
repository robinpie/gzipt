/* gzipt: gzip as a language model — C port of the zlib engine in ../gzipt.py.
 *
 * Faithful reimplementation of generate() (byte-level beam search) and
 * generate_spans() (corpus n-gram re-ranking) for the zlib backend only.
 *
 * The whole trick is the "clone": compress the context once, then for each
 * candidate fork the encoder state (Python's compressobj.copy(); here zlib's
 * deflateCopy) and feed only the candidate bytes. The match search over the
 * (large) context happens a single time per beam step.
 *
 * Bit-exactness vs the Python version:
 *   - The scorer uses the same system zlib, so compressed *lengths* are
 *     identical, and stable (length, index) ordering matches Python's stable
 *     sort. With --temperature 0 the output is byte-for-byte identical.
 *   - For --temperature > 0 we reproduce CPython's MT19937 + random.choices,
 *     so seeded sampled output also matches (assuming identical libm exp()).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <zlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* CPython-compatible Mersenne Twister (random.Random) for sampling.   */
/* ------------------------------------------------------------------ */

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL
#define MT_UPPER 0x80000000UL
#define MT_LOWER 0x7fffffffUL

typedef struct { uint32_t mt[MT_N]; int mti; } MT;

static void mt_init_genrand(MT *r, uint32_t s) {
    r->mt[0] = s;
    for (r->mti = 1; r->mti < MT_N; r->mti++) {
        r->mt[r->mti] = (1812433253UL *
            (r->mt[r->mti - 1] ^ (r->mt[r->mti - 1] >> 30)) + (uint32_t)r->mti);
    }
}

static void mt_init_by_array(MT *r, const uint32_t *key, int key_len) {
    mt_init_genrand(r, 19650218UL);
    int i = 1, j = 0;
    int k = (MT_N > key_len) ? MT_N : key_len;
    for (; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525UL))
                   + key[j] + (uint32_t)j;
        i++; j++;
        if (i >= MT_N) { r->mt[0] = r->mt[MT_N - 1]; i = 1; }
        if (j >= key_len) j = 0;
    }
    for (k = MT_N - 1; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941UL))
                   - (uint32_t)i;
        i++;
        if (i >= MT_N) { r->mt[0] = r->mt[MT_N - 1]; i = 1; }
    }
    r->mt[0] = 0x80000000UL;
}

/* random.seed(int) for a non-negative int splits abs(seed) into 32-bit words. */
static void mt_seed(MT *r, unsigned long long seed) {
    uint32_t key[2];
    int n = 0;
    do { key[n++] = (uint32_t)(seed & 0xffffffffULL); seed >>= 32; } while (seed && n < 2);
    mt_init_by_array(r, key, n);
}

static uint32_t mt_genrand_uint32(MT *r) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, MT_MATRIX_A};
    if (r->mti >= MT_N) {
        int kk;
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (r->mt[kk] & MT_UPPER) | (r->mt[kk + 1] & MT_LOWER);
            r->mt[kk] = r->mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (r->mt[kk] & MT_UPPER) | (r->mt[kk + 1] & MT_LOWER);
            r->mt[kk] = r->mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (r->mt[MT_N - 1] & MT_UPPER) | (r->mt[0] & MT_LOWER);
        r->mt[MT_N - 1] = r->mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        r->mti = 0;
    }
    y = r->mt[r->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

/* genrand_res53: 53-bit double in [0, 1), exactly as CPython's random(). */
static double mt_random(MT *r) {
    uint32_t a = mt_genrand_uint32(r) >> 5;
    uint32_t b = mt_genrand_uint32(r) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

/* bisect_right(a, x, lo, hi) — matches Python's bisect used by random.choices. */
static int bisect_right(const double *a, double x, int lo, int hi) {
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (x < a[mid]) hi = mid; else lo = mid + 1;
    }
    return lo;
}

/* random.choices(population, weights, k=1)[0]; returns chosen index. */
static int mt_choice(MT *r, const double *weights, int n) {
    double *cum = malloc((size_t)n * sizeof(double));
    double total = 0.0;
    for (int i = 0; i < n; i++) { total += weights[i]; cum[i] = total; }
    int idx = bisect_right(cum, mt_random(r) * total, 0, n - 1);
    free(cum);
    return idx;
}

/* ------------------------------------------------------------------ */
/* zlib scorer: compressed length of context + candidate via deflateCopy. */
/* ------------------------------------------------------------------ */

#define CHUNK 65536

/* Per-thread bump arena for zlib's allocations.
 *
 * The clone trick's cost is dominated by deflateCopy duplicating the full
 * ~300KB deflate state (window + hash tables) for every candidate. With the
 * default allocator that is a malloc + free per candidate; under threads the
 * allocator lock serialises everything. Routing zlib's ZALLOC/ZFREE through a
 * per-thread bump pointer removes all per-candidate malloc traffic: allocate by
 * advancing an offset, "free" is a no-op, and the whole arena is reset between
 * candidates. The big memcpy inside deflateCopy remains (it is inherent to the
 * trick), but the allocator contention that wrecked OpenMP scaling is gone. */
#define ARENA_CAP (4u << 20)   /* per thread; holds primed base + one clone */
typedef struct { unsigned char *buf; size_t cap, off, mark; } Arena;

static voidpf arena_alloc(voidpf opaque, uInt items, uInt size) {
    Arena *a = (Arena *)opaque;
    size_t need = ((size_t)items * size + 63) & ~(size_t)63; /* 64-byte aligned */
    if (a->off + need > a->cap) { fprintf(stderr, "arena overflow\n"); exit(1); }
    void *p = a->buf + a->off;
    a->off += need;
    return p;
}
static void arena_free(voidpf opaque, voidpf addr) { (void)opaque; (void)addr; }

/* Build a deflate stream primed with `context`, leaving its state ready to
 * clone. Returns bytes emitted while consuming the context (Python's
 * len(base.compress(context))). The stream must be deflateEnd()ed by caller. */
static size_t prime_context(z_stream *s, Arena *arena, int level,
                            const unsigned char *context, size_t ctxlen) {
    s->zalloc = arena_alloc; s->zfree = arena_free; s->opaque = arena;
    if (deflateInit(s, level) != Z_OK) { fprintf(stderr, "deflateInit failed\n"); exit(1); }
    unsigned char buf[CHUNK];
    size_t head = 0;
    s->next_in = (Bytef *)context;
    s->avail_in = (uInt)ctxlen;
    do {
        s->next_out = buf;
        s->avail_out = CHUNK;
        deflate(s, Z_NO_FLUSH);
        head += CHUNK - s->avail_out;
    } while (s->avail_in > 0 || s->avail_out == 0);
    return head;
}

/* Compressed length of context+seq: clone `tbase` (already primed with the
 * context, allocating through `arena`), feed seq, finish. Resets the arena. */
static size_t length_for(z_stream *tbase, Arena *arena, size_t head,
                         const unsigned char *seq, size_t seqlen) {
    z_stream c;
    if (deflateCopy(&c, tbase) != Z_OK) { fprintf(stderr, "deflateCopy failed\n"); exit(1); }
    unsigned char buf[CHUNK];
    size_t extra = 0;
    c.next_in = (Bytef *)seq;
    c.avail_in = (uInt)seqlen;
    do {
        c.next_out = buf; c.avail_out = CHUNK;
        deflate(&c, Z_NO_FLUSH);
        extra += CHUNK - c.avail_out;
    } while (c.avail_in > 0 || c.avail_out == 0);
    int ret;
    do {
        c.next_out = buf; c.avail_out = CHUNK;
        ret = deflate(&c, Z_FINISH);
        extra += CHUNK - c.avail_out;
    } while (ret != Z_STREAM_END);
    deflateEnd(&c);
    arena->off = arena->mark;   /* reclaim this candidate's allocations */
    return head + extra;
}

/* One thread's share of the candidate scoring: prime a context stream into a
 * private arena (so candidate clones allocate with zero malloc traffic), then
 * clone-and-finish each candidate. Recomputing the prime per thread costs a few
 * extra context compressions but buys lock-free, contention-free scaling. */
static void score_chunk(int level, const unsigned char *context, size_t ctxlen,
                        const unsigned char *cands, size_t clen,
                        int lo, int hi, size_t *out_lens) {
    Arena arena = { malloc(ARENA_CAP), ARENA_CAP, 0, 0 };
    z_stream s;
    size_t head = prime_context(&s, &arena, level, context, ctxlen);
    arena.mark = arena.off;   /* primed base lives below the mark */
    for (int i = lo; i < hi; i++)
        out_lens[i] = length_for(&s, &arena, head, cands + (size_t)i * clen, clen);
    free(arena.buf);          /* whole arena, incl. the primed base, freed at once */
}

/* Score every candidate (each `clen` bytes, packed contiguously in `cands`)
 * by compressed length of context+candidate. */
static void score_candidates(int level,
                             const unsigned char *context, size_t ctxlen,
                             const unsigned char *cands, size_t clen, int ncand,
                             size_t *out_lens) {
#ifdef _OPENMP
    #pragma omp parallel
    {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        int lo = (int)((long)ncand * tid / nt);
        int hi = (int)((long)ncand * (tid + 1) / nt);
        score_chunk(level, context, ctxlen, cands, clen, lo, hi, out_lens);
    }
#else
    score_chunk(level, context, ctxlen, cands, clen, 0, ncand, out_lens);
#endif
}

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

typedef struct { int idx; size_t len; } Scored;

/* Stable ascending sort by (len, idx), matching Python's stable sort. */
static int cmp_scored(const void *a, const void *b) {
    const Scored *x = a, *y = b;
    if (x->len < y->len) return -1;
    if (x->len > y->len) return 1;
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return 1;
    return 0;
}

static unsigned char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(sz > 0 ? (size_t)sz : 1);
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    *out_len = n;
    return buf;
}

/* ------------------------------------------------------------------ */
/* generation                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t window, length;
    int horizon, beam_width, tail, level, span_len, key;
    double temperature;
    unsigned long long seed;
    int spans; /* 0 = bytes mode, 1 = spans mode */
} Config;

/* Byte-level beam search (port of generate()). */
static void gen_bytes(const unsigned char *corpus, size_t corpus_len,
                      const unsigned char *prompt, size_t prompt_len,
                      const Config *cfg, unsigned char **out, size_t *out_len) {
    MT rng; mt_seed(&rng, cfg->seed);

    /* alphabet: sorted unique bytes of corpus + prompt */
    int present[256] = {0};
    for (size_t i = 0; i < corpus_len; i++) present[corpus[i]] = 1;
    for (size_t i = 0; i < prompt_len; i++) present[prompt[i]] = 1;
    unsigned char alpha[256]; int nalpha = 0;
    for (int b = 0; b < 256; b++) if (present[b]) alpha[nalpha++] = (unsigned char)b;
    if (nalpha == 0) for (int b = 0; b < 256; b++) alpha[nalpha++] = (unsigned char)b;

    size_t win = corpus_len < cfg->window ? corpus_len : cfg->window;

    int H = cfg->horizon, BW = cfg->beam_width;
    size_t cap = cfg->length + (size_t)H + 16;
    unsigned char *res = malloc(cap);
    size_t rlen = 0;

    /* scratch buffers */
    size_t max_cand = (size_t)BW * nalpha;
    unsigned char *cands = malloc(max_cand * (size_t)H);
    size_t *lens = malloc(max_cand * sizeof(size_t));
    Scored *scored = malloc(max_cand * sizeof(Scored));
    unsigned char *beams = malloc((size_t)BW * H);   /* current beams */
    size_t *beam_lens = malloc((size_t)BW * sizeof(size_t));
    unsigned char *ctx = malloc(win + (size_t)cfg->tail);
    double *weights = malloc((size_t)BW * sizeof(double));

    while (rlen < cfg->length) {
        /* recent = (prompt + out)[-tail:] */
        unsigned char recent[1 << 16];
        size_t total = prompt_len + rlen;
        size_t rstart = total > (size_t)cfg->tail ? total - cfg->tail : 0;
        size_t rn = total - rstart;
        for (size_t i = 0; i < rn; i++) {
            size_t pos = rstart + i;
            recent[i] = pos < prompt_len ? prompt[pos] : res[pos - prompt_len];
        }
        memcpy(ctx, corpus, win);
        memcpy(ctx + win, recent, rn);
        size_t ctxlen = win + rn;

        int nbeams = 1;            /* start: [b""] */
        size_t depth = 0;
        beam_lens[0] = 0;
        for (int step = 0; step < H; step++) {
            int ncand = nbeams * nalpha;
            size_t clen = depth + 1;
            for (int h = 0; h < nbeams; h++) {
                for (int a = 0; a < nalpha; a++) {
                    int ci = h * nalpha + a;
                    unsigned char *dst = cands + (size_t)ci * clen;
                    memcpy(dst, beams + (size_t)h * H, depth);
                    dst[depth] = alpha[a];
                }
            }
            score_candidates(cfg->level, ctx, ctxlen, cands, clen, ncand, lens);
            for (int i = 0; i < ncand; i++) { scored[i].idx = i; scored[i].len = lens[i]; }
            qsort(scored, ncand, sizeof(Scored), cmp_scored);
            int keep = ncand < BW ? ncand : BW;
            /* move kept candidates into beams */
            for (int i = 0; i < keep; i++) {
                memcpy(beams + (size_t)i * H,
                       cands + (size_t)scored[i].idx * clen, clen);
                beam_lens[i] = scored[i].len;
            }
            nbeams = keep;
            depth = clen;
        }

        /* choose span: greedy beams[0] or temperature sample */
        int pick = 0;
        if (cfg->temperature > 0.0) {
            size_t best = beam_lens[0];
            for (int i = 0; i < nbeams; i++)
                weights[i] = exp(-((double)beam_lens[i] - (double)best) / cfg->temperature);
            pick = mt_choice(&rng, weights, nbeams);
        }
        memcpy(res + rlen, beams + (size_t)pick * H, depth);
        rlen += depth;
    }

    free(cands); free(lens); free(scored); free(beams);
    free(beam_lens); free(ctx); free(weights);
    *out = res;
    *out_len = rlen < cfg->length ? rlen : cfg->length;
}

/* Span re-ranking (port of generate_spans()). */
static void gen_spans(const unsigned char *corpus, size_t corpus_len,
                      const unsigned char *prompt, size_t prompt_len,
                      const Config *cfg, unsigned char **out, size_t *out_len) {
    MT rng; mt_seed(&rng, cfg->seed);
    size_t win = corpus_len < cfg->window ? corpus_len : cfg->window;
    int SL = cfg->span_len, KEY = cfg->key;
    int MAXC = 96;

    size_t cap = cfg->length + (size_t)SL + 16;
    unsigned char *res = malloc(cap);
    size_t rlen = 0;

    unsigned char *cands = malloc((size_t)MAXC * SL);
    size_t *lens = malloc((size_t)MAXC * sizeof(size_t));
    double *weights = malloc((size_t)MAXC * sizeof(double));
    unsigned char *ctx = malloc(win + (size_t)cfg->tail);

    while (rlen < cfg->length) {
        unsigned char recent[1 << 16];
        size_t total = prompt_len + rlen;
        size_t rstart = total > (size_t)cfg->tail ? total - cfg->tail : 0;
        size_t rn = total - rstart;
        for (size_t i = 0; i < rn; i++) {
            size_t pos = rstart + i;
            recent[i] = pos < prompt_len ? prompt[pos] : res[pos - prompt_len];
        }
        memcpy(ctx, corpus, win);
        memcpy(ctx + win, recent, rn);
        size_t ctxlen = win + rn;

        /* back off from longest key until the corpus offers continuations */
        int ncand = 0;
        int kmax = (int)rn < KEY ? (int)rn : KEY;
        for (int k = kmax; k >= 1 && ncand == 0; k--) {
            const unsigned char *pat = recent + rn - k;
            /* scan window for occurrences of pat at i in [0, win-key)  */
            if (win <= (size_t)KEY) break;
            for (size_t i = 0; i + (size_t)KEY <= win && i < win - (size_t)KEY; i++) {
                if (memcmp(corpus + i, pat, (size_t)k) != 0) continue;
                size_t s = i + k;
                if (s + (size_t)SL > win) continue;
                const unsigned char *span = corpus + s;
                /* dedup against already-collected candidates */
                int dup = 0;
                for (int j = 0; j < ncand; j++)
                    if (memcmp(cands + (size_t)j * SL, span, SL) == 0) { dup = 1; break; }
                if (dup) continue;
                memcpy(cands + (size_t)ncand * SL, span, SL);
                ncand++;
                if (ncand >= MAXC) break;
            }
        }
        if (ncand == 0) break;

        score_candidates(cfg->level, ctx, ctxlen, cands, (size_t)SL, ncand, lens);

        int pick = 0;
        if (cfg->temperature > 0.0) {
            size_t best = lens[0];
            for (int i = 1; i < ncand; i++) if (lens[i] < best) best = lens[i];
            for (int i = 0; i < ncand; i++)
                weights[i] = exp(-((double)lens[i] - (double)best) / cfg->temperature);
            pick = mt_choice(&rng, weights, ncand);
        } else {
            size_t best = lens[0]; pick = 0;
            for (int i = 1; i < ncand; i++) if (lens[i] < best) { best = lens[i]; pick = i; }
        }
        memcpy(res + rlen, cands + (size_t)pick * SL, (size_t)SL);
        rlen += SL;
    }

    free(cands); free(lens); free(weights); free(ctx);
    *out = res;
    *out_len = rlen < cfg->length ? rlen : cfg->length;
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s --corpus FILE [--prompt STR] [--length N] [--mode bytes|spans]\n"
        "          [--horizon N] [--beam-width N] [--temperature F] [--tail N]\n"
        "          [--span-len N] [--key N] [--window N] [--level N]\n"
        "          [--workers N] [--seed N]\n"
        "  zlib backend only (the C engine port of ../gzipt.py).\n", prog);
}

int main(int argc, char **argv) {
    const char *corpus_path = NULL;
    const char *prompt_str = "";
    Config cfg = {
        .window = 30000, .length = 200, .horizon = 24, .beam_width = 32,
        .tail = 80, .level = 9, .span_len = 8, .key = 8,
        .temperature = 0.5, .seed = 3, .spans = 0,
    };
    int workers = 8;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() (++i < argc ? argv[i] : (usage(argv[0]), exit(1), ""))
        if      (!strcmp(a, "--corpus") || !strcmp(a, "--prime")) corpus_path = NEXT();
        else if (!strcmp(a, "--prompt")) prompt_str = NEXT();
        else if (!strcmp(a, "--length")) cfg.length = strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--mode")) cfg.spans = !strcmp(NEXT(), "spans");
        else if (!strcmp(a, "--horizon")) cfg.horizon = atoi(NEXT());
        else if (!strcmp(a, "--beam-width")) cfg.beam_width = atoi(NEXT());
        else if (!strcmp(a, "--temperature")) cfg.temperature = atof(NEXT());
        else if (!strcmp(a, "--tail")) cfg.tail = atoi(NEXT());
        else if (!strcmp(a, "--span-len")) cfg.span_len = atoi(NEXT());
        else if (!strcmp(a, "--key")) cfg.key = atoi(NEXT());
        else if (!strcmp(a, "--window")) cfg.window = strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--level")) cfg.level = atoi(NEXT());
        else if (!strcmp(a, "--workers")) workers = atoi(NEXT());
        else if (!strcmp(a, "--seed")) cfg.seed = strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--algo")) {
            const char *al = NEXT();
            if (strcmp(al, "zlib")) {
                fprintf(stderr, "C engine supports zlib only (got %s)\n", al); return 1;
            }
        }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 1; }
        #undef NEXT
    }

#ifdef _OPENMP
    if (workers > 0) omp_set_num_threads(workers);
#else
    (void)workers;
#endif

    size_t corpus_len = 0;
    unsigned char *corpus = corpus_path ? read_file(corpus_path, &corpus_len)
                                        : (unsigned char *)"";
    const unsigned char *prompt = (const unsigned char *)prompt_str;
    size_t prompt_len = strlen(prompt_str);

    unsigned char *out = NULL; size_t out_len = 0;
    if (cfg.spans)
        gen_spans(corpus, corpus_len, prompt, prompt_len, &cfg, &out, &out_len);
    else
        gen_bytes(corpus, corpus_len, prompt, prompt_len, &cfg, &out, &out_len);

    fwrite(prompt, 1, prompt_len, stdout);
    fwrite(out, 1, out_len, stdout);
    if (out_len == 0 || out[out_len - 1] != '\n') fputc('\n', stdout);

    free(out);
    if (corpus_path) free(corpus);
    return 0;
}
