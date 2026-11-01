"""Thin LLM wrapper + the RAG prompt. Provider chosen in config / env vars."""
import json
import os
import re
from functools import lru_cache

from config import LLM_PROVIDER, LLM_MODEL

SYSTEM = (
    "You are an assistant for open-source EDA tools (OpenROAD, Yosys, OpenSTA). "
    "Answer ONLY from the numbered context passages. Cite passages like [1], [2]. "
    "Quote command names and flags exactly as they appear. If the context does "
    "not contain the answer, say: 'Not found in the provided documentation.' "
    "Do not use outside knowledge."
)


def format_context(chunks, max_chars=1800):
    parts = []
    for n, c in enumerate(chunks, 1):
        loc = f"{c['tool']}:{c['path']}" + (f" :: {c['section']}" if c.get("section") else "")
        parts.append(f"[{n}] ({loc})\n{c['text'][:max_chars]}")
    return "\n\n".join(parts)


def rag_prompt(question, chunks):
    return f"Context:\n{format_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"


@lru_cache(maxsize=4)
def _client(provider):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    return None


@lru_cache(maxsize=2)
def _hf(model):
    from transformers import pipeline
    return pipeline("text-generation", model=model, device_map="auto")


def generate(prompt, system=SYSTEM, max_tokens=700, provider=None, model=None, temperature=0.0):
    provider = provider or LLM_PROVIDER
    model = model or LLM_MODEL
    if provider == "none":
        return "(LLM_PROVIDER=none - generation skipped)"
    if provider == "anthropic":
        r = _client(provider).messages.create(
            model=model, max_tokens=max_tokens, system=system, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if b.type == "text")
    if provider == "openai":
        r = _client(provider).chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    if provider == "hf":
        pipe = _hf(model)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        out = pipe(msgs, max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]
    raise ValueError(f"unknown provider {provider}")


def parse_json(text):
    """Pull the first JSON object out of an LLM reply (tolerates ```json fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in: {text[:200]}")
    return json.loads(m.group(0))
