"""Ask a question (Stages 1-3, pick the retrieval mode).

    python 04_ask.py "What does -max_fanout do in repair_design?"
    python 04_ask.py "How do I flatten the design in yosys?" --mode dense --chunks naive
    python 04_ask.py "..." --compare        # side-by-side retrieval for all modes
    LLM_PROVIDER=none python 04_ask.py "..." --show-context   # retrieval only
"""
import argparse
import time

from llm import generate, rag_prompt
from retrieval import Retriever, MODES


def show(hits):
    for n, c in enumerate(hits, 1):
        print(f"  [{n}] {c['score']:.3f}  {c['tool']}:{c['path']}  {c['section'][:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("--chunks", choices=["naive", "struct"], default="struct")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tool", nargs="*", help="restrict to openroad/yosys/opensta")
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    r = Retriever(args.chunks)

    if args.compare:
        for m in MODES:
            t = time.time()
            try:
                hits = r.search(args.question, m, args.k, args.tool)
            except SystemExit as e:
                print(f"\n{m}: skipped ({e})"); continue
            print(f"\n{m}  ({time.time()-t:.2f}s)"); show(hits)
        return

    hits = r.search(args.question, args.mode, args.k, args.tool)
    print(f"Retrieved ({args.mode}, {args.chunks}):"); show(hits)
    if args.show_context:
        for n, c in enumerate(hits, 1):
            print(f"\n----- [{n}] -----\n{c['text'][:1500]}")
    print("\nAnswer:\n" + generate(rag_prompt(args.question, hits)))
    print("\nSources:")
    for n, c in enumerate(hits, 1):
        print(f"  [{n}] {c['url']}")


if __name__ == "__main__":
    main()
