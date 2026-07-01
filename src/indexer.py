"""
Full-corpus indexer.

Streams every article from the CSV, chunks with the chosen config values, embeds
in batches, and upserts into the DEFAULT Pinecone namespace (the one the live app
queries). Run this ONCE at the final config. Re-running overwrites (chunk_ids are
stable) rather than duplicating.

The experiment's cfg_* namespaces are separate and untouched by this.
"""

import time

from data import load_articles
from chunking import chunk_article
from embeddings import embed_batch
from vector_store import upsert_chunks, ensure_index
import config

PATH = r"C:/Users/golde/PycharmProjects/RAG Assignment/medium-english-50mb.csv"
BATCH = 100     # chunks per embedding call (stays under request limits)


def _retry(fn, what, retries=5):
    """Run fn(); on any error, wait (1,2,4,8,16s) and retry. Handles transient network blips."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  {what} failed ({type(e).__name__}); retry {attempt + 1}/{retries} in {wait}s...")
            time.sleep(wait)


def _flush(chunks):
    # Retry embed and upsert separately so an upsert blip never re-pays for embedding.
    vectors = _retry(lambda: embed_batch([c.text for c in chunks]), "embed")
    _retry(lambda: upsert_chunks(chunks, vectors), "upsert")   # default namespace ""


def build_index(limit=None):
    ensure_index()
    batch = []
    total = 0
    for article in load_articles(PATH, limit):
        for chunk in chunk_article(article, config.CHUNK_SIZE, config.OVERLAP_RATIO):
            batch.append(chunk)
            if len(batch) >= BATCH:
                _flush(batch)
                total += len(batch)
                print(f"indexed {total} chunks...")
                batch = []
    if batch:                          # final partial batch
        _flush(batch)
        total += len(batch)
    print(f"DONE — {total} chunks")


if __name__ == "__main__":
    build_index(limit=None)     # None = the FULL corpus (~7,600 articles)
