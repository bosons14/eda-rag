# eda-rag: a RAG assistant for OpenROAD, Yosys and OpenSTA

A hands-on RAG project built in stages. Each stage is a script; each one
teaches one idea and gets measured on the same evaluation set.

| Stage | Script | Idea |
|---|---|---|
| 1 | `01_ingest.py`, `03_index.py`, `04_ask.py --mode dense` | Naive RAG baseline |
| 2 | `02_chunk.py` | Structure-aware chunking (headings, whole code blocks, metadata) |
| 3 | `04_ask.py --mode hybrid / hybrid_rerank` | BM25 + dense fused with RRF, cross-encoder rerank |
| 4 | `05_make_evalset.py`, `06_eval.py` | Retrieval metrics (recall@k, MRR) + LLM-judged faithfulness |
| 5 | `agent.py`, `07_agentic.py` | Query rewrite, routing, self-check + retry; log triage |
| 6 | `LLM_PROVIDER=hf` | Swap in a tiny generator and watch where it breaks |

Library modules: `config.py` (all settings), `retrieval.py`, `llm.py`, `agent.py`.

## Setup

Runs on Colab/Kaggle (free T4 is plenty) or any laptop; embeddings on CPU take
a few minutes for this corpus (~190k words, ~1.2k chunks).

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or: LLM_PROVIDER=openai OPENAI_API_KEY=... [OPENAI_BASE_URL=...]
```

On Colab: upload the folder (or push it to your GitHub and `git clone`), then
run the commands below with a leading `!`, and set keys with
`os.environ["ANTHROPIC_API_KEY"] = ...`.

LLM providers (env `LLM_PROVIDER`): `anthropic` (default, `claude-haiku-5-5`),
`openai` (any OpenAI-compatible server: OpenAI, Groq, Together, Ollama, vLLM),
`hf` (local transformers model), `none` (no generation, retrieval only, free).
Use `JUDGE_PROVIDER` / `JUDGE_MODEL` for a stronger model to build and grade the eval set.

## Run it

```bash
# Stage 1: data + baseline
python 01_ingest.py                       # sparse-clones docs only (fast)
python 01_ingest.py --issues 200          # optional: + closed GitHub issues (set GITHUB_TOKEN)
python 02_chunk.py                        # writes BOTH naive and struct chunk sets
python 03_index.py                        # BM25 + FAISS for both
python 04_ask.py "What does -max_fanout do in repair_design?" --mode dense --chunks naive

# Stage 2: look at what chunking did
python 02_chunk.py --inspect "repair_design -max"

# Stage 3: compare retrieval modes on one question
python 04_ask.py "stop the resizer buffering beyond some fanout" --compare
python 04_ask.py "stop the resizer buffering beyond some fanout"     # hybrid_rerank + answer

# Stage 4: build eval set, measure everything
python 05_make_evalset.py synthetic --n 80
# then add 20-30 of YOUR OWN questions to data/eval/manual.jsonl (4 examples included)
python 05_make_evalset.py check
python 06_eval.py                                           # 4 modes x 2 chunkings
python 06_eval.py --show-failures hybrid_rerank --chunks struct
python 06_eval.py --modes hybrid_rerank --chunks struct --answers   # faithfulness/correctness

# Stage 5: agentic
python 07_agentic.py "my netlist has nets driving hundreds of loads, how do I fix it in the flow?"
python 07_agentic.py --log path/to/openroad.log
python 06_eval.py --modes hybrid_rerank agentic --chunks struct     # is the agent worth it?

# Stage 6 (bridge to nanoGPT): tiny generator, same retrieval
LLM_PROVIDER=hf LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct python 06_eval.py --modes hybrid_rerank --chunks struct --answers
```

## What to look for at each stage

1. **Baseline.** Dense retrieval on naive chunks misses exact identifiers:
   `-max_fanout`, `set_input_delay`, error codes like `GRT-0116`. Embeddings
   capture meaning, not exact strings.
2. **Chunking.** Naive windows flatten Markdown tables and separate a command's
   synopsis from its option table. The structural chunker keeps code blocks whole,
   ignores `#` Tcl comments inside code (they aren't headings), merges tiny
   sections and prefixes each chunk with `[tool] path :: Heading > Sub`, so the
   embedding knows where the text came from.
3. **Hybrid.** BM25 wins on identifiers and dense wins on paraphrase; RRF merges
   their rankings without any score normalization. The tokenizer in
   `retrieval.py` keeps `repair_design` whole *and* splits it, so both
   "repair_design" and "repair design" match. The reranker is the slowest part,
   so check how much it adds to `s/q`.
4. **Evaluation.** Items are scored on an *evidence span* (verbatim text), not
   chunk IDs, so different chunkings get a fair comparison. Synthetic questions
   share wording with the passage and flatter BM25, so your manual questions are
   the honest test. Faithfulness and correctness are separate metrics: an answer
   can be faithful to the wrong context.
5. **Agentic.** Look at the trace: did the rewrite produce the docs' vocabulary?
   Did routing send an OpenSTA question to OpenROAD? (A retry widens the search
   for that reason.) Compare recall *and* seconds per question against
   `hybrid_rerank`. The agent makes 2-4 extra LLM calls per question.
6. **Tiny generator.** With the same context, a 0.5B model ignores citations,
   blends passages, and answers from its own weights when the context
   doesn't help. Faithfulness drops while recall stays the same. That gap
   between retrieving the right text and using it is what you'll be studying from the inside with nanoGPT.

## Ideas to extend

- Add `--with-issues` chunks and check whether real-user Q&A improves recall on your manual set.
- Contextual retrieval: have an LLM write a one-line summary of each chunk's document and prepend it before embedding.
- Fine-tune the embedder on your synthetic (question, chunk) pairs (sentence-transformers `MultipleNegativesRankingLoss`).
- Expose `Retriever.search` as an MCP tool so Claude Code can use it from your eda-ops workflow.
