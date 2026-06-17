"""Does a realistic, well-commented SQL corpus get more *sensible* AND valid
output from gzipt? Uses the real Chinook sample DB (executable target) with a
generated corpus of realistic, commented SELECTs over its actual schema/values.

Metrics, all on generated statements that were NOT verbatim in the corpus:
  - valid      : executes without error
  - +returns   : valid AND returns >=1 row  (proxy for *semantic* sensibility —
                 absurd predicates like `age >= 80000` are exactly what return 0)
Compares a commented corpus vs the same corpus with comment lines stripped.
"""

from __future__ import annotations

import random
import re
import sqlite3

import gzipt


def load_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(open("data/sql/chinook.sql", encoding="utf-8", errors="replace").read())
    return db


def vals(db, q):
    return [r[0] for r in db.execute(q).fetchall() if r[0] is not None]


def make_corpus(db, rng: random.Random, n: int = 280) -> list[tuple[str, str]]:
    """(comment, query) pairs — realistic SELECTs over Chinook with sane domains."""
    composers = rng.sample(vals(db, "SELECT DISTINCT Composer FROM Track WHERE Composer IS NOT NULL"), 12)
    artists = rng.sample(vals(db, "SELECT Name FROM Artist"), 12)
    countries = vals(db, "SELECT DISTINCT BillingCountry FROM Invoice")
    genres = vals(db, "SELECT Name FROM Genre")
    out = []
    for _ in range(n):
        t = rng.randint(0, 8)
        if t == 0:
            p = rng.choice([0.99, 1.99]); d = rng.choice(["", " ORDER BY Name"])
            out.append(("-- Tracks at a given price point",
                        f"SELECT Name, UnitPrice FROM Track WHERE UnitPrice = {p}{d};"))
        elif t == 1:
            ms = rng.choice([200000, 300000, 400000])
            out.append(("-- Longer-than-average tracks, longest first",
                        f"SELECT Name, Milliseconds FROM Track WHERE Milliseconds > {ms} ORDER BY Milliseconds DESC;"))
        elif t == 2:
            out.append(("-- Every track written by a specific composer",
                        f"SELECT Name FROM Track WHERE Composer = '{rng.choice(composers)}';"))
        elif t == 3:
            out.append(("-- Albums released by one artist",
                        "SELECT Album.Title, Artist.Name FROM Album JOIN Artist "
                        f"ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = '{rng.choice(artists)}';"))
        elif t == 4:
            tot = rng.choice([5, 10, 15])
            out.append(("-- Invoices above a spending threshold",
                        f"SELECT BillingCountry, Total FROM Invoice WHERE Total > {tot} ORDER BY Total DESC;"))
        elif t == 5:
            out.append(("-- Customers located in one country",
                        f"SELECT FirstName, LastName, City FROM Customer WHERE Country = '{rng.choice(countries)}';"))
        elif t == 6:
            out.append(("-- Total revenue broken down by country",
                        "SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY BillingCountry ORDER BY SUM(Total) DESC;"))
        elif t == 7:
            lim = rng.choice([5, 10, 20])
            out.append(("-- A handful of the longest tracks",
                        f"SELECT Name FROM Track ORDER BY Milliseconds DESC LIMIT {lim};"))
        else:
            out.append(("-- How many tracks fall under each media type",
                        "SELECT MediaTypeId, COUNT(*) FROM Track GROUP BY MediaTypeId;"))
    return out


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"--[^\n]*", "", s)).strip().lower().rstrip(";")


def statements(text):
    text = re.sub(r"--[^\n]*", " ", text)  # strip line comments before splitting
    return [s.strip() + ";" for s in text.split(";")
            if s.strip().upper().startswith("SELECT") and len(s.strip()) > 14]


def trial(label, corpus_text, corpus_norm, db, span_len):
    out = gzipt.generate_spans(corpus_text.encode(), b"SELECT ",
                               length=1200, span_len=span_len, key=8, seed=1)
    stmts = list(dict.fromkeys(statements("SELECT " + out.decode("utf-8", "replace"))))
    valid = novel_valid = novel_valid_rows = 0
    good = []
    for s in stmts:
        try:
            rows = db.execute(s).fetchall()
            ok = True
        except Exception:
            ok = False; rows = []
        novel = norm(s) not in corpus_norm
        valid += ok
        if ok and novel:
            novel_valid += 1
            if rows:
                novel_valid_rows += 1
                if len(good) < 5:
                    good.append((len(rows), s))
    print(f"\n--- {label} (span_len={span_len}) ---")
    print(f"  {len(stmts)} distinct | {valid} valid | {novel_valid} novel+valid | "
          f"{novel_valid_rows} novel+valid+returns-rows")
    for nrows, s in good:
        print(f"    [{nrows:>4} rows] {s}")


def main():
    db = load_db()
    rng = random.Random(11)
    pairs = make_corpus(db, rng)
    commented = "\n".join(f"{c}\n{q}" for c, q in pairs) + "\n"
    plain = "\n".join(q for _, q in pairs) + "\n"
    corpus_norm = {norm(q) for _, q in pairs}
    print(f"corpus: {len(pairs)} queries | commented {len(commented)}B | plain {len(plain)}B "
          f"| chinook {db.execute('SELECT COUNT(*) FROM Track').fetchone()[0]} tracks")

    for sl in (8, 12):
        trial("COMMENTED corpus", commented, corpus_norm, db, sl)
    for sl in (8, 12):
        trial("PLAIN corpus", plain, corpus_norm, db, sl)


if __name__ == "__main__":
    main()
