CC      ?= cc
# -O2 vs -O3 is a wash here: ~all runtime is inside libz.so, not our loop
# (measured 19.24s vs 19.72s single-thread, within noise). -O2 by default.
CFLAGS  ?= -O2 -march=native -Wall -Wextra -std=c11
LDLIBS  := -lz -lm
# OpenMP parallelises candidate scoring (the Python version uses a thread pool).
OPENMP  ?= -fopenmp

gzipt: gzipt.c
	$(CC) $(CFLAGS) $(OPENMP) -o $@ gzipt.c $(LDLIBS)

# Single-threaded build (no OpenMP) — useful for clean perf comparison.
gzipt-st: gzipt.c
	$(CC) $(CFLAGS) -o $@ gzipt.c $(LDLIBS)

clean:
	rm -f gzipt gzipt-st

.PHONY: clean
