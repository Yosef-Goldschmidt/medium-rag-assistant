"""
Config comparison experiment.

Indexes the first LIMIT articles into a SEPARATE Pinecone namespace per config,
so different chunk_size/overlap settings can be compared side by side without
their vectors colliding. Then runs the test questions against each config and
prints the retrieved chunks with scores, so we can (a) pick the best config and
(b) eyeball the score distribution to choose a min_score threshold.

USAGE:
  1. First run: leave run_index() enabled -> embeds + upserts all 3 configs.
  2. After that: comment out run_index() and just run run_eval() as many times
     as you like (re-embedding is the costly part; querying is cheap).
"""

from data import load_articles
from chunking import chunk_article
from embeddings import embed_batch
from vector_store import upsert_chunks, ensure_index
from retriever import retrieve

PATH = r"C:/Users/golde/PycharmProjects/RAG Assignment/medium-english-50mb.csv"
LIMIT = 200          # articles to index for the experiment
TOP_K = 5            # fixed across configs so we isolate the chunking effect
BATCH = 100          # chunks per embedding call (stays under request limits)

# The 3 chunk-size configs to compare. Each goes into its own namespace.
CONFIGS = [
    {"name": "cfg_256_010",  "chunk_size": 256,  "overlap": 0.10},
    {"name": "cfg_512_020",  "chunk_size": 512,  "overlap": 0.20},
    {"name": "cfg_1024_015", "chunk_size": 1024, "overlap": 0.15},
]

# Overlap-isolation set: fix chunk_size=512, vary ONLY overlap.
# cfg_512_020 is already indexed from the first run, so we reuse it.
OVERLAP_CONFIGS = [
    {"name": "cfg_512_010", "chunk_size": 512, "overlap": 0.10},
    {"name": "cfg_512_020", "chunk_size": 512, "overlap": 0.20},   # already indexed
    {"name": "cfg_512_030", "chunk_size": 512, "overlap": 0.30},
]

# Only the two NEW overlap variants still need embedding (512_020 is done).
NEW_OVERLAP_CONFIGS = [
    {"name": "cfg_512_010", "chunk_size": 512, "overlap": 0.10},
    {"name": "cfg_512_030", "chunk_size": 512, "overlap": 0.30},
]

# One question per query type (plus a known anchor). Adjust freely.
TEST_QUESTIONS = [
    "Find an article about smell training and the brain. Give the title and author.",
    "List 3 distinct articles about education. Titles only.",
    "Find an article about mental health and summarize its central idea.",
    "I want practical, beginner-friendly advice on building habits that stick. Which article would you recommend, and why?",
    "an article about the coronavirus pandemic and the brain",
]


def _flush(chunks, namespace):
    """Embed one batch of chunks and upsert into the given namespace."""
    vectors = embed_batch([c.text for c in chunks])
    upsert_chunks(chunks, vectors, namespace=namespace)


def index_config(cfg):
    """Stream LIMIT articles, chunk with this config, batch-embed, upsert to its namespace."""
    batch = []
    total = 0
    for article in load_articles(PATH, LIMIT):
        for chunk in chunk_article(article, cfg["chunk_size"], cfg["overlap"]):
            batch.append(chunk)
            if len(batch) >= BATCH:
                _flush(batch, cfg["name"])
                total += len(batch)
                batch = []
    if batch:                      # final partial batch
        _flush(batch, cfg["name"])
        total += len(batch)
    print(f"[{cfg['name']}] indexed {total} chunks")


def run_index(configs):
    ensure_index()
    for cfg in configs:
        index_config(cfg)


def run_eval(configs):
    for q in TEST_QUESTIONS:
        print("\n" + "=" * 90)
        print("Q:", q)
        for cfg in configs:
            print(f"\n--- {cfg['name']} ---")
            for r in retrieve(q, TOP_K, namespace=cfg["name"]):
                print(f"  {round(r.score, 3)} | {r.article_id} | {r.title[:65]}")


if __name__ == "__main__":
    # --- overlap-isolation run (chunk_size fixed at 512, vary overlap) ---
    run_index(NEW_OVERLAP_CONFIGS)   # embed the 2 new overlap variants (512_020 already done)
    run_eval(OVERLAP_CONFIGS)        # compare 0.10 vs 0.20 vs 0.30

    # --- the earlier chunk-size comparison (already indexed) ---
    # run_eval(CONFIGS)
