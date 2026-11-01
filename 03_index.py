"""Stage 1b/3 - Build indexes: BM25 (keyword) and FAISS (dense) per chunk set.

    python 03_index.py                     # both chunk sets, both indexes
    python 03_index.py --chunks struct
    python 03_index.py --no-dense          # BM25 only (no GPU / no model download)

Output: data/index/{naive,struct}/bm25.pkl, faiss.index, chunks.jsonl
"""
import argparse
import json
import pickle
import shutil

from config import CHUNKS, INDEX, EMBED_MODEL
from retrieval import tokenize


def build(name, dense):
    src = CHUNKS / f"chunks_{name}.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} missing - run 02_chunk.py")
    out = INDEX / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out / "chunks.jsonl")
    chunks = [json.loads(l) for l in open(src)]
    texts = [c["text"] for c in chunks]

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"[{name}] bm25 over {len(texts)} chunks")

    if dense:
        import faiss
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)
        index = faiss.IndexFlatIP(emb.shape[1])   # cosine (embeddings are normalized)
        index.add(emb.astype("float32"))
        faiss.write_index(index, str(out / "faiss.index"))
        (out / "embed_model.txt").write_text(EMBED_MODEL)
        print(f"[{name}] faiss dim={emb.shape[1]} model={EMBED_MODEL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", choices=["naive", "struct", "both"], default="both")
    ap.add_argument("--no-dense", action="store_true")
    args = ap.parse_args()
    for name in (["naive", "struct"] if args.chunks == "both" else [args.chunks]):
        build(name, dense=not args.no_dense)


if __name__ == "__main__":
    main()
