#!/usr/bin/env python3
import argparse
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from runtime import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_QDRANT_URL,
    default_sentence_transformer_device,
    default_top_k,
)


def build_filter(lang: Optional[str], doc_type: Optional[str]) -> Optional[qm.Filter]:
    must = []
    if lang:
        must.append(qm.FieldCondition(key="language", match=qm.MatchValue(value=lang.upper())))
    if doc_type:
        must.append(qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=doc_type.lower())))
    return qm.Filter(must=must) if must else None


def qdrant_query(
    client: QdrantClient,
    collection_name: str,
    qvec,
    top_k: int,
    flt: Optional[qm.Filter],
):
    """
    Compatibility layer:
    - Newer qdrant-client: query_points()
    - Older qdrant-client: search()
    """
    if hasattr(client, "query_points"):
        # newer API
        res = client.query_points(
            collection_name=collection_name,
            query=qvec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=flt,
        )
        return res.points

    # older API
    return client.search(
        collection_name=collection_name,
        query_vector=qvec,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Query text")
    ap.add_argument("--qdrant", default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Embedding model")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--top_k", type=int, default=default_top_k())
    ap.add_argument("--lang", default=None, help="Language filter e.g. RO/PL/EN/DE")
    ap.add_argument("--doc_type", default=None, help="doc_type filter e.g. policy/guideline/example")
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant)
    device = default_sentence_transformer_device(args.device)
    emb = SentenceTransformer(args.model, device=device)

    qvec = emb.encode([args.query], normalize_embeddings=True)[0].tolist()
    flt = build_filter(args.lang, args.doc_type)

    hits = qdrant_query(client, args.collection, qvec, args.top_k, flt)

    if not hits:
        print("No results.")
        return

    print(f"\nQuery: {args.query}\n")
    for i, h in enumerate(hits, 1):
        # h can be ScoredPoint in both APIs
        p = h.payload or {}
        text = (p.get("text") or "")[:240].replace("\n", " ")
        print(f"{i}. score={h.score:.4f}  lang={p.get('language')}  type={p.get('doc_type')}  title={p.get('title')}")
        print(f"   src={p.get('path')}#chunk={p.get('chunk_index')} v={p.get('version')}")
        print(f"   {text}...\n")


if __name__ == "__main__":
    main()
