#!/usr/bin/env python3
import argparse
import glob
import os
import re
import uuid
from typing import Dict, List, Tuple

import frontmatter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from runtime import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_QDRANT_URL,
    default_chunk_size,
    default_embed_batch_size,
    default_overlap,
    default_sentence_transformer_device,
    default_upsert_batch_size,
)


def simple_chunk(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """Simple char-based chunking.  Can be improved later with smarter sentence-based chunking if needed."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(
            size=vector_size,
            distance=qm.Distance.COSINE,
        ),
    )

    # Useful indexes for filtering
    client.create_payload_index(
        collection_name=collection_name,
        field_name="language",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_type",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )


def load_docs(docs_dir: str) -> List[Tuple[str, Dict, str]]:
    paths = sorted(glob.glob(os.path.join(docs_dir, "**/*.*"), recursive=True))
    out = []
    for p in paths:
        if os.path.isdir(p):
            continue
        try:
            post = frontmatter.load(p)
            meta = dict(post.metadata or {})
            text = (post.content or "").strip()
            if not text:
                continue
            out.append((p, meta, text))
        except Exception:
            # fallback: treat as plain text
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
                if txt:
                    out.append((p, {}, txt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs", help="Docs folder")
    ap.add_argument("--qdrant", default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Embedding model")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--batch_size", type=int, default=default_embed_batch_size(), help="Embedding batch size")
    ap.add_argument("--upsert_batch_size", type=int, default=default_upsert_batch_size(), help="Qdrant upsert batch size")
    ap.add_argument("--chunk_size", type=int, default=default_chunk_size())
    ap.add_argument("--overlap", type=int, default=default_overlap())
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant)
    device = default_sentence_transformer_device(args.device)
    emb = SentenceTransformer(args.model, device=device)
    vector_size = emb.get_sentence_embedding_dimension()

    ensure_collection(client, args.collection, vector_size)

    docs = load_docs(args.docs)
    if not docs:
        raise SystemExit(f"No docs found in {args.docs}. Add markdown/txt files first.")

    total_chunks = 0

    for path, meta, text in docs:
        language = (meta.get("language") or meta.get("lang") or "UNK").upper()
        doc_type = (meta.get("doc_type") or "doc").lower()
        source = meta.get("source") or "local"
        version = meta.get("version") or "0.0"
        title = meta.get("title") or os.path.basename(path)

        chunks = simple_chunk(text, chunk_size=args.chunk_size, overlap=args.overlap)
        if not chunks:
            continue

        vectors = emb.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )

        points: List[qm.PointStruct] = []
        for idx, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
            chunk_id = str(uuid.uuid4())
            payload = {
                "path": path,
                "title": title,
                "chunk_index": idx,
                "language": language,
                "doc_type": doc_type,
                "source": source,
                "version": str(version),
                "text": chunk_text,  # store snippet for Day 1 debugging; we can optimize later
            }
            points.append(qm.PointStruct(id=chunk_id, vector=vec.tolist(), payload=payload))
            total_chunks += 1

            if len(points) >= args.upsert_batch_size:
                client.upsert(collection_name=args.collection, points=points)
                points = []

        if points:
            client.upsert(collection_name=args.collection, points=points)

    print(
        f"Ingested {len(docs)} docs into '{args.collection}' as {total_chunks} chunks "
        f"using {args.model} on {device}."
    )


if __name__ == "__main__":
    main()
