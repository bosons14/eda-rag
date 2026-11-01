"""Stage 2 - Chunking: naive fixed-window vs. structure-aware.

    python 02_chunk.py                 # docs only
    python 02_chunk.py --with-issues   # include issues.jsonl
    python 02_chunk.py --inspect "repair_design"   # see how each strategy cut it

Output: data/chunks/chunks_naive.jsonl, data/chunks/chunks_struct.jsonl

What to look at: with naive chunks, a Tcl command's synopsis and its option
table often land in different chunks, and '#' comment lines inside code blocks
get mistaken for headings by careless splitters. The structural chunker keeps
code blocks whole, tracks the heading path and prefixes it to every chunk.
"""
import argparse
import json
import re

from config import RAW, CHUNKS, NAIVE_WORDS, NAIVE_OVERLAP, STRUCT_MAX_WORDS, STRUCT_MIN_WORDS

RST_UNDERLINE = re.compile(r'^([=\-~^"\'`*+#:._])\1{2,}\s*$')
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
RST_CODE = re.compile(r"^\s*(\.\. (code-block|code|literalinclude)::|.*::\s*$)")
CMD_LINE = re.compile(r"^\s*([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")  # snake_case commands


def load(paths):
    for p in paths:
        if p.exists():
            with open(p) as f:
                for line in f:
                    yield json.loads(line)


# ---------------------------------------------------------------- naive ----
def naive_chunks(doc):
    words = doc["text"].split()
    step = NAIVE_WORDS - NAIVE_OVERLAP
    for i in range(0, max(1, len(words) - NAIVE_OVERLAP), step):
        piece = words[i:i + NAIVE_WORDS]
        if not piece:
            break
        yield {"text": " ".join(piece), "section": ""}


# ----------------------------------------------------------- structural ----
def split_sections(text, is_rst):
    """Yield (heading_path, [blocks]). Blocks are paragraphs or whole code blocks."""
    lines = text.splitlines()
    path, rst_levels = [], []
    blocks, cur, in_fence, in_rst_code = [], [], False, False

    def flush_para():
        nonlocal cur
        if any(l.strip() for l in cur):
            blocks.append(("\n".join(cur).strip("\n"), in_fence or in_rst_code))
        cur = []

    def emit():
        nonlocal blocks
        flush_para()
        out = (list(path), blocks)
        blocks = []
        return out

    i = 0
    while i < len(lines):
        line = lines[i]
        # --- fenced code (markdown): never interpret '#' inside it
        if FENCE.match(line):
            if not in_fence:
                flush_para(); in_fence = True; cur.append(line)
            else:
                cur.append(line); flush_para(); in_fence = False
            i += 1; continue
        if in_fence:
            cur.append(line); i += 1; continue

        # --- rst literal blocks: '::' then an indented block
        if is_rst and in_rst_code:
            if line.strip() == "" or line.startswith((" ", "\t")):
                cur.append(line); i += 1; continue
            flush_para(); in_rst_code = False

        # --- headings
        md = None if is_rst else MD_HEADING.match(line)
        rst = (is_rst and i + 1 < len(lines) and line.strip()
               and RST_UNDERLINE.match(lines[i + 1])
               and len(lines[i + 1].strip()) >= len(line.strip()))
        if md or rst:
            if md:
                level, title = len(md.group(1)), md.group(2)
                skip = 1
            else:
                ch = lines[i + 1].strip()[0]
                if ch not in rst_levels:
                    rst_levels.append(ch)
                level, title = rst_levels.index(ch) + 1, line.strip()
                skip = 2
            yield emit()
            path[:] = path[: level - 1] + [title]
            i += skip; continue
        if is_rst and RST_UNDERLINE.match(line):   # rst overline / transition
            i += 1; continue

        if line.strip() == "":
            flush_para()
        else:
            cur.append(line)
            if is_rst and RST_CODE.match(line):
                flush_para(); in_rst_code = True
        i += 1
    yield emit()


def pack(blocks, max_words):
    """Greedy-pack blocks; a code block is never split and stays with the
    paragraph that introduces it."""
    out, cur, n = [], [], 0
    expanded = []
    for text, is_code in blocks:          # split only giant blocks, at line breaks
        if len(text.split()) > max_words * 2:
            lines, buf = text.splitlines(), []
            for ln in lines:
                buf.append(ln)
                if len(" ".join(buf).split()) >= max_words:
                    expanded.append(("\n".join(buf), False)); buf = []
            if buf:
                expanded.append(("\n".join(buf), False))
        else:
            expanded.append((text, is_code))
    for text, is_code in expanded:
        w = len(text.split())
        if cur and n + w > max_words and not is_code:
            out.append(cur); cur, n = [], 0
        cur.append(text); n += w
        if n > max_words * 1.5:      # very long section: hard break after code
            out.append(cur); cur, n = [], 0
    if cur:
        out.append(cur)
    return ["\n\n".join(c) for c in out]


def struct_chunks(doc):
    if doc["source"] == "issue":       # issues: keep Q + top answers together
        text = doc["text"]
        for j, part in enumerate(pack([(p, False) for p in text.split("\n\n")], STRUCT_MAX_WORDS)):
            yield {"text": part, "section": doc.get("title", "")}
        return
    is_rst = doc["path"].endswith(".rst")
    pending = None   # tiny section waiting to be merged into the next one
    for path, blocks in split_sections(doc["text"], is_rst):
        if not blocks:
            continue
        section = " > ".join(path)
        for part in pack(blocks, STRUCT_MAX_WORDS):
            if pending:
                psec, ptext = pending
                if psec and psec != section:
                    part = f"## {section}\n{part}"
                    ptext = f"## {psec}\n{ptext}"
                part, pending = ptext + "\n\n" + part, None
            if len(part.split()) < STRUCT_MIN_WORDS:
                pending = (section, part); continue
            yield {"text": part, "section": section}
    if pending:
        yield {"text": pending[1], "section": pending[0]}


def commands_in(text):
    cmds = set()
    for line in text.splitlines():
        m = CMD_LINE.match(line)
        if m:
            cmds.add(m.group(1))
    return sorted(cmds)


def build(docs, strategy):
    fn = naive_chunks if strategy == "naive" else struct_chunks
    out = []
    for doc in docs:
        for j, c in enumerate(fn(doc)):
            header = f"[{doc['tool']}] {doc['path']}" + (f" :: {c['section']}" if c["section"] else "")
            body = c["text"]
            out.append({
                "id": f"{doc['id']}#{j}",
                "doc_id": doc["id"],
                "tool": doc["tool"],
                "source": doc["source"],
                "path": doc["path"],
                "url": doc["url"],
                "section": c["section"],
                "commands": commands_in(body),
                # struct chunks carry their context header INTO the embedded text
                "text": f"{header}\n{body}" if strategy == "struct" else body,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-issues", action="store_true")
    ap.add_argument("--inspect", help="print chunks containing this string")
    args = ap.parse_args()
    files = [RAW / "docs.jsonl"] + ([RAW / "issues.jsonl"] if args.with_issues else [])
    docs = list(load(files))
    if not docs:
        raise SystemExit("no docs - run 01_ingest.py first")

    for strategy in ("naive", "struct"):
        chunks = build(docs, strategy)
        with open(CHUNKS / f"chunks_{strategy}.jsonl", "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        lens = sorted(len(c["text"].split()) for c in chunks)
        print(f"{strategy:6s}: {len(chunks):5d} chunks, median {lens[len(lens)//2]} words, "
              f"max {lens[-1]}")
        if args.inspect:
            hits = [c for c in chunks if args.inspect in c["text"]][:3]
            for c in hits:
                print(f"\n--- {strategy} | {c['id']} | {c['section']}\n{c['text'][:900]}\n")


if __name__ == "__main__":
    main()
