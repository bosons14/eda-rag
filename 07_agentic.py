"""Stage 5 - Ask with the agentic pipeline (prints the agent's trace).

    python 07_agentic.py "my design has huge fanout nets after synthesis, how do I fix that in the flow?"
    python 06_eval.py --modes hybrid_rerank agentic --chunks struct   # the real test

The optional extra - log triage: pass a log file and the agent turns its
errors/warnings into the question.
    python 07_agentic.py --log run.log
"""
import argparse
import re

from agent import answer
from retrieval import Retriever

MSG = re.compile(r"^\s*(\[?(ERROR|WARNING|Error|Warning)\b.*)$", re.M)


def log_question(path, max_msgs=8):
    text = open(path, errors="ignore").read()
    msgs = list(dict.fromkeys(m.group(1).strip() for m in MSG.finditer(text)))[:max_msgs]
    if not msgs:
        raise SystemExit("no ERROR/WARNING lines found")
    return ("My EDA tool run produced these messages. What do they mean and how do I fix them?\n"
            + "\n".join(msgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--log")
    ap.add_argument("--chunks", default="struct")
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    q = log_question(a.log) if a.log else a.question
    if not q:
        ap.error("give a question or --log")
    ans, ctx, _ = answer(q, Retriever(a.chunks), a.k, verbose=True)
    print("\nAnswer:\n" + ans + "\n\nSources:")
    for n, c in enumerate(ctx, 1):
        print(f"  [{n}] {c['url']}  ({c['section'][:60]})")


if __name__ == "__main__":
    main()
