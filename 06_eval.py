"""Stage 4b - Evaluate retrieval (and optionally answers) across configurations.

    python 06_eval.py                                  # all modes x both chunkings
    python 06_eval.py --modes bm25 hybrid --chunks struct
    python 06_eval.py --modes hybrid_rerank agentic --answers   # + LLM-judged answers
    python 06_eval.py --show-failures hybrid_rerank    # where does it miss?

Retrieval metrics (no LLM needed):
  recall@k  fraction of questions whose evidence appears in the top-k chunks
  MRR       mean of 1/rank of the first chunk containing the evidence
Answer metrics (--answers, uses JUDGE model):
  faithful  every claim supported by the retrieved context (no hallucination)
  correct   agrees with the reference answer
"""
import argparse
import json
import re
import time

from config import EVAL, JUDGE_PROVIDER, JUDGE_MODEL
from llm import generate, parse_json, rag_prompt, format_context
from retrieval import Retriever, MODES

KS = (1, 3, 5, 10)
JUDGE_PROMPT = """Question: {q}
Reference answer: {ref}
Context given to the system:
{ctx}
System answer: {ans}

Grade the system answer. JSON only:
{{"faithful": true/false  (every factual claim is supported by the context, or it correctly says not found),
 "correct": true/false   (it agrees with the reference answer on the key fact)}}"""


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def load_items(limit):
    items = []
    for name in ("manual.jsonl", "synthetic.jsonl"):
        p = EVAL / name
        if p.exists():
            items += [json.loads(l) for l in open(p)]
    if not items:
        raise SystemExit("no eval items - run 05_make_evalset.py")
    return items[:limit] if limit else items


def first_hit(hits, evidence):
    ev = norm(evidence or "")
    if not ev:              # empty evidence would "match" every chunk
        return None
    for r, c in enumerate(hits, 1):
        if ev in norm(c["text"]):
            return r
    return None


def run(items, retriever, mode, answers):
    if mode == "agentic":
        from agent import agentic_retrieve
        get = lambda q: agentic_retrieve(q, retriever, k=max(KS))[0]
    else:
        get = lambda q: retriever.search(q, mode, k=max(KS))
    rows, t0 = [], time.time()
    for it in items:
        hits = get(it["question"])
        row = {"id": it["id"], "question": it["question"], "rank": first_hit(hits, it["evidence"]),
               "top": [f"{c['tool']}:{c['path']}::{c['section'][:50]}" for c in hits[:3]]}
        if answers:
            ctx = hits[:5]
            ans = generate(rag_prompt(it["question"], ctx))
            try:
                g = parse_json(generate(JUDGE_PROMPT.format(
                    q=it["question"], ref=it.get("answer") or it["evidence"],
                    ctx=format_context(ctx, 1200), ans=ans),
                    system="You are a strict grader. JSON only.", max_tokens=100,
                    provider=JUDGE_PROVIDER, model=JUDGE_MODEL))
            except Exception:
                g = {"faithful": None, "correct": None}
            row.update(answer=ans, **g)
        rows.append(row)
    secs = (time.time() - t0) / len(items)
    m = {f"R@{k}": sum(1 for r in rows if r["rank"] and r["rank"] <= k) / len(rows) for k in KS}
    m["MRR"] = sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows)
    m["s/q"] = secs
    if answers:
        for key in ("faithful", "correct"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            m[key] = sum(vals) / len(vals) if vals else float("nan")
    return m, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=MODES, help=f"{MODES} + agentic")
    ap.add_argument("--chunks", nargs="*", default=["naive", "struct"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--show-failures", metavar="MODE")
    args = ap.parse_args()
    items = load_items(args.limit)
    print(f"{len(items)} questions\n")

    table, all_rows = [], {}
    for ch in args.chunks:
        retriever = Retriever(ch)
        for mode in args.modes:
            try:
                m, rows = run(items, retriever, mode, args.answers)
            except SystemExit as e:
                print(f"skip {ch}/{mode}: {e}"); continue
            table.append((ch, mode, m)); all_rows[f"{ch}/{mode}"] = rows
            print(f"{ch:6s} {mode:14s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = EVAL / f"results-{stamp}.json"
    out.write_text(json.dumps({"metrics": [(c, mo, m) for c, mo, m in table], "rows": all_rows}, indent=1))
    print(f"\nsaved {out}")

    if args.show_failures:
        for key, rows in all_rows.items():
            if key.endswith("/" + args.show_failures):
                print(f"\nmisses for {key}:")
                for r in rows:
                    if not r["rank"] or r["rank"] > 5:
                        print(f"  - {r['question']}\n      got: {r['top']}")


if __name__ == "__main__":
    main()
