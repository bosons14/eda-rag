"""Central configuration. Override anything with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"          # cloned repos + docs.jsonl / issues.jsonl
CHUNKS = DATA / "chunks"    # chunks_naive.jsonl, chunks_struct.jsonl
INDEX = DATA / "index"      # bm25 / faiss per chunking strategy
EVAL = DATA / "eval"        # evalset.jsonl, results

for d in (RAW, CHUNKS, INDEX, EVAL):
    d.mkdir(parents=True, exist_ok=True)

# --- Corpus -----------------------------------------------------------------
# Sparse-checkout patterns keep clones small (docs only, no source tree).
SOURCES = {
    "openroad": {
        "repo": "https://github.com/The-OpenROAD-Project/OpenROAD",
        "patterns": ["src/*/README.md", "docs/user/*.md", "docs/main/*.md"],
        "issues": "The-OpenROAD-Project/OpenROAD",
    },
    "yosys": {
        "repo": "https://github.com/YosysHQ/yosys",
        "patterns": ["docs/source/**/*.rst", "/README.md"],
        "issues": "YosysHQ/yosys",
    },
    "opensta": {
        "repo": "https://github.com/The-OpenROAD-Project/OpenSTA",
        "patterns": ["/README.md", "/doc/*.txt", "/doc/*.md"],
        "issues": "The-OpenROAD-Project/OpenSTA",
    },
}
DOC_SUFFIXES = {".md", ".rst", ".txt"}

# --- Models -----------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge v1.5 works better with an instruction prefix on queries (not on passages)
QUERY_PREFIX = os.getenv(
    "QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# LLM_PROVIDER: anthropic | openai | hf | none
#   openai -> any OpenAI-compatible server (OpenAI, Groq, Together, Ollama,
#             vLLM...) via OPENAI_BASE_URL + OPENAI_API_KEY
#   hf     -> local transformers model (use a tiny model to see where small
#             generators break: e.g. Qwen/Qwen2.5-0.5B-Instruct)
#   none   -> no generation; prints retrieved context (dry runs, free)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    {
        "anthropic": "claude-haiku-5-5",
        "openai": "gpt-4o-mini",
        "hf": "Qwen/Qwen2.5-0.5B-Instruct",
        "none": "none",
    }.get(LLM_PROVIDER, ""),
)
# Optional separate (stronger) model for building the eval set and judging
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", LLM_PROVIDER)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

# --- Chunking ---------------------------------------------------------------
NAIVE_WORDS = 200
NAIVE_OVERLAP = 40
STRUCT_MAX_WORDS = 350
STRUCT_MIN_WORDS = 60   # smaller sections get merged into the next one
