"""Can gzipt produce *working* code? SQL is the best bet: a SELECT is a complete,
self-contained program with no long-range obligations (no lock to release, no
variable to define-then-return) — exactly the dependencies that broke the C
attempt. And we can *execute* the output to verify, instead of eyeballing it.

Build a schema + a corpus of valid SELECTs, prime gzipt, generate, then run every
generated statement against a real SQLite DB. The headline metric is
NOVEL-AND-VALID: runs without error AND was not verbatim in the corpus.
"""

from __future__ import annotations

import random
import re
import sqlite3

import gzipt

SCHEMA = """\
CREATE TABLE employees (id INTEGER, name TEXT, dept_id INTEGER, salary INTEGER, age INTEGER, city TEXT);
CREATE TABLE departments (id INTEGER, name TEXT, budget INTEGER);
CREATE TABLE projects (id INTEGER, name TEXT, dept_id INTEGER, budget INTEGER);
"""

CITIES = ["Austin", "Denver", "Boston", "Seattle", "Portland"]
ECOLS = ["id", "name", "dept_id", "salary", "age", "city"]
NUMCOLS = ["salary", "age", "dept_id"]


def make_corpus(rng: random.Random, n: int = 320) -> list[str]:
    """Generate n valid, varied SELECT statements against the schema."""
    q = []
    for _ in range(n):
        t = rng.randint(0, 6)
        if t == 0:
            cols = ", ".join(rng.sample(ECOLS, rng.randint(1, 3)))
            c = rng.choice(NUMCOLS); op = rng.choice([">", "<", ">=", "<="]); v = rng.choice([25, 30, 40, 1000, 50000, 80000])
            d = rng.choice(["ASC", "DESC"]); o = rng.choice(ECOLS)
            q.append(f"SELECT {cols} FROM employees WHERE {c} {op} {v} ORDER BY {o} {d};")
        elif t == 1:
            q.append(f"SELECT name FROM employees WHERE city = '{rng.choice(CITIES)}';")
        elif t == 2:
            q.append("SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id;")
        elif t == 3:
            f = rng.choice(["AVG", "MAX", "MIN", "SUM"]); c = rng.choice(NUMCOLS)
            q.append(f"SELECT {f}({c}) FROM employees WHERE age > {rng.choice([25, 30, 40])};")
        elif t == 4:
            q.append("SELECT e.name, d.name FROM employees e JOIN departments d ON e.dept_id = d.id "
                     f"WHERE d.budget > {rng.choice([100000, 500000])};")
        elif t == 5:
            q.append(f"SELECT name, salary FROM employees ORDER BY salary DESC LIMIT {rng.choice([5, 10, 20])};")
        else:
            q.append(f"SELECT name FROM projects WHERE budget > {rng.choice([10000, 50000])} ORDER BY budget;")
    return q


def fresh_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    rng = random.Random(0)
    for i in range(60):
        db.execute("INSERT INTO employees VALUES (?,?,?,?,?,?)",
                   (i, f"emp{i}", rng.randint(1, 5), rng.randint(30000, 120000),
                    rng.randint(22, 65), rng.choice(CITIES)))
    for i in range(1, 6):
        db.execute("INSERT INTO departments VALUES (?,?,?)", (i, f"dept{i}", rng.randint(50000, 800000)))
        db.execute("INSERT INTO projects VALUES (?,?,?,?)", (i, f"proj{i}", i, rng.randint(5000, 90000)))
    db.commit()
    return db


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def statements(text: str) -> list[str]:
    """Split generated text into candidate statements; keep ones starting SELECT."""
    out = []
    for part in text.split(";"):
        s = part.strip()
        if s.upper().startswith("SELECT") and len(s) > 12:
            out.append(s + ";")
    return out


def run_trial(label: str, corpus_text: str, corpus_norm: set[str], gen, gen_kwargs: dict):
    out = gen(corpus_text.encode(), b"SELECT ", **gen_kwargs)
    text = "SELECT " + out.decode("utf-8", "replace")
    stmts = statements(text)
    db = fresh_db()
    valid = novel_valid = 0
    examples = []
    for s in dict.fromkeys(stmts):  # dedup, keep order
        try:
            db.execute(s).fetchall()
            ok = True
        except Exception:
            ok = False
        novel = norm(s) not in corpus_norm
        valid += ok
        if ok and novel:
            novel_valid += 1
            if len(examples) < 4:
                examples.append(s)
    n = len(set(stmts))
    print(f"\n--- {label} ---")
    print(f"  {n} distinct candidate statements | {valid} run OK | {novel_valid} NOVEL-and-valid")
    for e in examples:
        print(f"    OK+novel: {e}")
    db.close()


def main():
    rng = random.Random(7)
    corpus = make_corpus(rng)
    corpus_text = SCHEMA + "\n" + "\n".join(corpus) + "\n"
    corpus_norm = {norm(q) for q in corpus}
    print(f"corpus: {len(corpus)} queries, {len(corpus_text)} bytes; schema + SELECTs")

    run_trial("BYTES mode", corpus_text, corpus_norm, gzipt.generate,
              dict(length=800, temperature=0.5, tail=120, seed=1))
    for sl in (8, 12, 16):
        run_trial(f"SPANS mode (span_len={sl})", corpus_text, corpus_norm, gzipt.generate_spans,
                  dict(length=800, span_len=sl, key=8, seed=1))


if __name__ == "__main__":
    main()
