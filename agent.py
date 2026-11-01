"""Stage 5 - Agentic RAG: plan -> retrieve -> judge -> (retry) -> answer.

The agent adds three things plain RAG lacks:
  1. Query rewriting: turns vague questions into tool vocabulary
     ("stop buffering high-fanout nets" -> "repair_design -max_fanout").
  2. Routing: picks which tool's docs to search (metadata filter).
  3. Self-check: judges whether the context actually answers the question
     and, if not, issues follow-up queries for what is missing.
Every LLM call costs latency and money, so Stage 5's real question is:
does this beat hybrid_rerank on YOUR eval set by enough to be worth it?
"""
import json

from llm import generate, parse_json, rag_prompt, format_context
from retrieval import Retriever

TOOLS_DESC = (
    "openroad: physical design flow - floorplan, placement, CTS, routing, resizer "
    "(repair_design, repair_timing), PDN, IO placement, DFT, global/detail route.\n"
    "yosys: RTL synthesis - read_verilog, synth, flatten, opt, abc, techmap, dfflibmap, "
    "passes and scripting.\n"
    "opensta: static timing analysis - SDC (create_clock, set_input_delay), report_checks, "
    "liberty, parasitics, timing exceptions."
)

PLAN = """You route questions to EDA documentation search.
Tools:
{tools}

Question: {q}

Return JSON: {{"tools": [subset of "openroad","yosys","opensta"; empty list if unsure],
 "queries": [2-3 search queries using the exact command/option vocabulary the docs would use]}}"""

JUDGE = """Question: {q}

Retrieved context:
{ctx}

Does the context contain enough to answer the question fully and specifically?
Return JSON: {{"sufficient": true/false, "missing": "what is missing (if any)",
 "new_queries": [up to 2 queries that would find the missing piece]}}"""

AGENT_SYS = "You are a precise retrieval planner. Reply with JSON only."


def agentic_retrieve(q, retriever, k=5, max_rounds=2, mode="hybrid_rerank", verbose=False):
    trace = []
    try:
        plan = parse_json(generate(PLAN.format(tools=TOOLS_DESC, q=q), system=AGENT_SYS, max_tokens=250))
    except Exception:
        plan = {"tools": [], "queries": []}
    tools = [t for t in plan.get("tools", []) if t in ("openroad", "yosys", "opensta")] or None
    queries = [q] + plan.get("queries", [])[:3]
    trace.append({"step": "plan", "tools": tools, "queries": queries})

    fused = {}                       # chunk id -> [chunk, accumulated RRF score]
    for rnd in range(max_rounds + 1):
        # accumulate across rounds: evidence found earlier is never thrown away,
        # and chunks that several queries agree on rise to the top
        for x in queries:
            for rank, c in enumerate(retriever.search(x, mode, k, tools)):
                fused.setdefault(c["id"], [c, 0.0])[1] += 1.0 / (60 + rank + 1)
        ranked = [dict(c, score=s) for c, s in sorted(fused.values(), key=lambda v: -v[1])]
        if rnd == max_rounds:
            break
        try:
            j = parse_json(generate(JUDGE.format(q=q, ctx=format_context(ranked[:k], 1200)),
                                    system=AGENT_SYS, max_tokens=250))
        except Exception:
            break
        trace.append({"step": f"judge{rnd}", **j})
        if j.get("sufficient") or not j.get("new_queries"):
            break
        queries = j["new_queries"][:2]
        tools = None          # widen the search on retry: routing may have been wrong

    if verbose:
        print(json.dumps(trace, indent=2))
    return ranked[:k], trace


def answer(q, retriever, k=5, verbose=True):
    ctx, trace = agentic_retrieve(q, retriever, k, verbose=verbose)
    return generate(rag_prompt(q, ctx)), ctx, trace
