"""Stage 4a - Build the evaluation set.

Every eval item has an EVIDENCE string: a short exact span of the source text
that a correct retrieval must contain. A retrieved chunk counts as relevant if
it contains the evidence, so naive and struct chunkings are scored fairly.

    python 05_make_evalset.py synthetic --n 80      # LLM writes questions from chunks
    python 05_make_evalset.py template              # write a manual template
    python 05_make_evalset.py check                 # validate evalset.jsonl

Synthetic questions are a starting point, not the finish line: they tend to
reuse the passage's wording, which flatters BM25. Add 20-30 hand-written
questions in your own words (ideally from real GitHub issues) to
data/eval/manual.jsonl - the eval script merges both files.
"""
import argparse
import json
import random
import re

from config import CHUNKS, EVAL, JUDGE_PROVIDER, JUDGE_MODEL
from llm import generate, parse_json

GEN_SYSTEM = "You write evaluation questions for a documentation search system. Reply with JSON only."
GEN_PROMPT = """Passage from {tool} documentation ({path} :: {section}):
---
{text}
---
Write ONE question a real engineer might ask that this passage answers specifically
(a command, an option, a behaviour, a default value...). Rules:
- Paraphrase: do NOT copy distinctive phrases from the passage into the question,
  except unavoidable command or option names.
- The question must make sense on its own without seeing the passage.
- "evidence": a SHORT span (3-12 words) copied EXACTLY from the passage that contains the key fact.
- "answer": a 1-2 sentence reference answer.
If the passage has no specific answerable fact, return {{"skip": true}}.
JSON: {{"question": "...", "evidence": "...", "answer": "..."}}"""


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def synthetic(n, seed):
    chunks = [json.loads(l) for l in open(CHUNKS / "chunks_struct.jsonl")]
    chunks = [c for c in chunks if c["source"] == "docs" and len(c["text"].split()) > 60]
    random.Random(seed).shuffle(chunks)
    # stratify across tools so one big manual doesn't dominate
    by_tool = {}
    for c in chunks:
        by_tool.setdefault(c["tool"], []).append(c)
    order = [c for group in zip(*by_tool.values()) for c in group] or chunks

    out_path = EVAL / "synthetic.jsonl"
    made = 0
    with open(out_path, "w") as f:
        for c in order:
            if made >= n:
                break
            try:
                j = parse_json(generate(
                    GEN_PROMPT.format(tool=c["tool"], path=c["path"], section=c["section"],
                                      text=c["text"][:2500]),
                    system=GEN_SYSTEM, max_tokens=300,
                    provider=JUDGE_PROVIDER, model=JUDGE_MODEL))
            except Exception as e:
                print("  skip (bad reply):", e); continue
            if j.get("skip") or norm(j.get("evidence", "")) not in norm(c["text"]):
                continue   # evidence must be verbatim, or the item is useless
            item = {"id": f"syn-{made}", "question": j["question"], "evidence": j["evidence"],
                    "answer": j.get("answer", ""), "tool": c["tool"], "gold_path": c["path"],
                    "origin": "synthetic"}
            f.write(json.dumps(item) + "\n"); made += 1
            print(f"  [{made}/{n}] {item['question']}")
    print(f"wrote {made} -> {out_path}")


def template():
    p = EVAL / "manual.jsonl"
    if p.exists():
        raise SystemExit(f"{p} exists; not overwriting")
    examples = [
        {"id": "man-0", "question": "How do I stop the resizer from buffering nets beyond a certain fanout?",
         "evidence": "max_fanout", "answer": "", "tool": "openroad", "origin": "manual"},
        {"id": "man-1", "question": "Which yosys command turns the hierarchy into a single flat module?",
         "evidence": "flatten", "answer": "", "tool": "yosys", "origin": "manual"},
    ]
    with open(p, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"wrote template {p} - edit/extend it. 'evidence' must appear verbatim in the docs.")


def check():
    chunks = [json.loads(l) for l in open(CHUNKS / "chunks_struct.jsonl")]
    corpus = [norm(c["text"]) for c in chunks]
    for name in ("synthetic.jsonl", "manual.jsonl"):
        p = EVAL / name
        if not p.exists():
            continue
        items = [json.loads(l) for l in open(p)]
        bad = [i["id"] for i in items if not any(norm(i["evidence"]) in t for t in corpus)]
        print(f"{name}: {len(items)} items, {len(bad)} with evidence not found {bad[:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synthetic", "template", "check"])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"synthetic": lambda: synthetic(a.n, a.seed), "template": template, "check": check}[a.cmd]()


if __name__ == "__main__":
    main()
