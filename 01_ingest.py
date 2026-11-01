"""Stage 1a - Ingest: sparse-clone EDA docs (and optionally GitHub issues).

    python 01_ingest.py                    # docs only
    python 01_ingest.py --issues 200       # + 200 closed issues per repo
    GITHUB_TOKEN=ghp_xxx python 01_ingest.py --issues 500   # token = 5000 req/h

Output: data/raw/docs.jsonl, data/raw/issues.jsonl
"""
import argparse
import json
import os
import subprocess
import time

import requests

from config import RAW, SOURCES, DOC_SUFFIXES


def sparse_clone(name, spec):
    dest = RAW / name
    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             spec["repo"] + ".git", str(dest)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *spec["patterns"]],
        check=True,
    )
    return dest


def collect_docs(name, spec, dest):
    for f in sorted(dest.rglob("*")):
        if ".git" in f.parts or not f.is_file() or f.suffix not in DOC_SUFFIXES:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text.split()) < 30:          # skip stubs
            continue
        rel = f.relative_to(dest).as_posix()
        yield {
            "id": f"{name}:{rel}",
            "tool": name,
            "source": "docs",
            "path": rel,
            "url": f"{spec['repo']}/blob/HEAD/{rel}",
            "text": text,
        }


def fetch_issues(name, owner_repo, limit, token=None):
    """Closed issues that got discussion: the 'real user questions' of the corpus."""
    s = requests.Session()
    s.headers["Accept"] = "application/vnd.github+json"
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    out, page = [], 1
    while len(out) < limit:
        r = s.get(f"https://api.github.com/repos/{owner_repo}/issues",
                  params={"state": "closed", "per_page": 100, "page": page,
                          "sort": "comments", "direction": "desc"})
        if r.status_code == 403:
            print("  rate-limited; set GITHUB_TOKEN for more"); break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for it in batch:
            if "pull_request" in it or it["comments"] == 0:
                continue
            c = s.get(it["comments_url"], params={"per_page": 30})
            comments = c.json() if c.ok else []
            body = "\n\n".join(
                [f"QUESTION: {it['title']}\n\n{it.get('body') or ''}"]
                + [f"COMMENT by {x['user']['login']}:\n{x.get('body') or ''}" for x in comments]
            )
            out.append({
                "id": f"{name}:issue/{it['number']}",
                "tool": name,
                "source": "issue",
                "path": f"issues/{it['number']}",
                "url": it["html_url"],
                "title": it["title"],
                "text": body,
            })
            if len(out) >= limit:
                break
            time.sleep(0.1 if token else 1.0)
        page += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issues", type=int, default=0, help="closed issues per repo (0 = skip)")
    ap.add_argument("--tools", nargs="*", default=list(SOURCES))
    args = ap.parse_args()

    docs = []
    for name in args.tools:
        print(f"[{name}] cloning docs")
        dest = sparse_clone(name, SOURCES[name])
        got = list(collect_docs(name, SOURCES[name], dest))
        print(f"  {len(got)} doc files")
        docs += got
    with open(RAW / "docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    if args.issues:
        token = os.getenv("GITHUB_TOKEN")
        issues = []
        for name in args.tools:
            print(f"[{name}] fetching issues")
            got = fetch_issues(name, SOURCES[name]["issues"], args.issues, token)
            print(f"  {len(got)} issues")
            issues += got
        with open(RAW / "issues.jsonl", "w") as f:
            for d in issues:
                f.write(json.dumps(d) + "\n")

    words = sum(len(d["text"].split()) for d in docs)
    print(f"done: {len(docs)} docs, ~{words:,} words -> {RAW}")


if __name__ == "__main__":
    main()
