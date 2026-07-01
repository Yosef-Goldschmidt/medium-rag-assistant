from dataclasses import dataclass
from embeddings import embed_batch
from vector_store import query
import config

@dataclass
class RetrievedChunk:
    article_id: str
    title: str
    chunk: str
    score: float
    authors: list[str]
    url: str

def retrieve(question, top_k, namespace="", min_score=0.0) -> list[RetrievedChunk]:
    qvec = embed_batch([question])[0]
    matches = query(qvec, top_k, namespace=namespace)
    res = []
    for m in matches:
        if m.score < min_score:
            continue
        res.append(RetrievedChunk(article_id=m.metadata["article_id"], title=m.metadata["title"], chunk=m.metadata["text"], score=m.score, authors=m.metadata["authors"], url=m.metadata["url"]))
    return res

def dedup_by_article(results) -> list[RetrievedChunk]:
    seen = set()
    distinct = []
    for r in results:
        if r.article_id not in seen:
            seen.add(r.article_id)
            distinct.append(r)
    return distinct

# for r in retrieve("smell training and the brain", config.TOP_K):
#     print(round(r.score, 3), r.article_id, r.title)
# print("distinct:", len(dedup_by_article(retrieve("smell training and the brain", config.TOP_K))))