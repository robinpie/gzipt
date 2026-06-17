"""Hunt for practical, working gzipt-generated code in the flat/statement-local
basin: CSS. Declarations are mutually independent, rules are independent, nesting
is one shallow brace level, and OSS CSS is hugely redundant — the recombination
sweet spot. Validity is checked with a real parser (tinycss2), and we demand the
rule be NOVEL (not verbatim in the corpus), so it's generation, not copying.

Run with the venv that has tinycss2:  .venv/bin/python css_experiment.py
"""

from __future__ import annotations

import re
import sys

import tinycss2

import gzipt


# A solid subset of real, applicable CSS properties. Requiring these (not custom
# `--vars`, whose values are almost unconstrained) makes "valid" mean "real,
# applicable styling," not just "token-balanced."
STANDARD_PROPS = set("""
color background background-color background-image background-size background-position
background-repeat margin margin-top margin-right margin-bottom margin-left padding
padding-top padding-right padding-bottom padding-left border border-top border-right
border-bottom border-left border-color border-width border-style border-radius width
height max-width min-width max-height min-height display position top right bottom left
float clear font font-size font-family font-weight font-style line-height text-align
text-decoration text-transform letter-spacing word-spacing white-space vertical-align
overflow overflow-x overflow-y z-index opacity cursor content box-shadow box-sizing
transition transform flex flex-direction flex-wrap flex-grow flex-shrink justify-content
align-items align-self align-content gap grid-template-columns grid-template-rows
list-style list-style-type outline outline-offset visibility word-wrap overflow-wrap
text-overflow text-indent fill stroke order resize user-select pointer-events
""".split())


def sane_value(decl) -> bool:
    v = tinycss2.serialize(decl.value)
    return bool(v.strip()) and "{" not in v and "}" not in v and ";" not in v \
        and v.count("(") == v.count(")")


def corpus_rule_set(css: str) -> set[str]:
    rules = set()
    for n in tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True):
        if n.type == "qualified-rule":
            rules.add(re.sub(r"\s+", " ", tinycss2.serialize([n])).strip())
    return rules


def valid_rules(css: str):
    """Yield (serialized_rule, selector, n_decls) for well-formed qualified rules."""
    for n in tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True):
        if n.type != "qualified-rule":
            continue
        sel = tinycss2.serialize(n.prelude).strip()
        decls = tinycss2.parse_declaration_list(n.content, skip_comments=True, skip_whitespace=True)
        if not sel or any(d.type == "error" for d in decls):
            continue
        good = [d for d in decls if d.type == "declaration"]
        # Every declaration must be a real, applicable property with a sane value.
        if good and all(d.lower_name in STANDARD_PROPS and sane_value(d) for d in good):
            yield re.sub(r"\s+", " ", tinycss2.serialize([n])).strip(), sel, len(good)


def trial(corpus: str, prompt: bytes, corpus_rules: set[str], span_len: int):
    out = gzipt.generate_spans(corpus.encode(), prompt, length=2000,
                               span_len=span_len, key=8, seed=1)
    gen = prompt.decode("utf-8", "replace") + out.decode("utf-8", "replace")
    seen, novel_valid, examples = set(), 0, []
    total = 0
    for ser, sel, nd in valid_rules(gen):
        if ser in seen:
            continue
        seen.add(ser)
        total += 1
        if ser not in corpus_rules:
            novel_valid += 1
            if len(examples) < 6:
                examples.append(ser)
    print(f"\n--- span_len={span_len} ---")
    print(f"  {total} distinct valid rules | {novel_valid} NOVEL + valid")
    for e in examples:
        print(f"    {e}")
    return [ser for ser in seen if ser not in corpus_rules]


def main():
    corpus = open("data/css/realprops.css", encoding="utf-8", errors="replace").read()
    corpus_rules = corpus_rule_set(corpus)
    prompt = b"button {\n  "
    print(f"corpus: realprops.css {len(corpus)} bytes, {len(corpus_rules)} rules "
          f"(gzipt sees first {gzipt.DEFAULT_WINDOW} bytes)")
    novel = {}
    for sl in (8, 12, 16):
        for ser in trial(corpus, prompt, corpus_rules, sl):
            novel.setdefault(ser, sl)

    # Tangible artifact: write the novel valid rules as a real stylesheet and
    # confirm the *whole file* parses with zero errors.
    if novel:
        sheet = "/* gzipt-generated, novel + valid CSS rules */\n" + "\n".join(novel)
        with open("data/css/gzipt_generated.css", "w") as fh:
            fh.write(sheet + "\n")
        errs = sum(n.type == "error" for n in tinycss2.parse_stylesheet(sheet))
        print(f"\nwrote {len(novel)} novel rules -> data/css/gzipt_generated.css "
              f"| whole-file parse errors: {errs}")


if __name__ == "__main__":
    main()
