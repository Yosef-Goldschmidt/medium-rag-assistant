# RAG hyperparameters — single source of truth (also reported by /api/stats).
CHUNK_SIZE = 512        # tokens per chunk (assignment cap: 1024)
OVERLAP_RATIO = 0.2     # fraction of overlap between chunks (assignment cap: 0.3)
TOP_K = 8               # chunks retrieved for a normal question (assignment cap: 30)

# Retrieval filtering (not part of /api/stats).
MIN_SCORE = 0.2         # drop clearly-unrelated chunks; low because broad-topic matches score ~0.33
                        # (0.35 over-filtered valid education/topic queries on the full corpus)

# "List N distinct articles" query handling.
LIST_POOL = 30          # retrieve a wide pool of chunks, then dedup by article
LIST_CANDIDATES = 8     # show up to this many DISTINCT articles to the LLM; it picks the ~3 most on-topic
