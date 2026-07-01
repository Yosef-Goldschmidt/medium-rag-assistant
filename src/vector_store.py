import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key = api_key)
INDEX_NAME = "medium-rag"

def ensure_index():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name= INDEX_NAME,
            dimension= 1536,
            metric= "cosine",
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
        )

ensure_index()
index = pc.Index(INDEX_NAME)

def upsert_chunks(chunks, vectors, namespace=""):
    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": {
                "article_id": chunk.article_id,
                "title": chunk.title,
                "text": chunk.text,
                "authors": chunk.authors,
                "url": chunk.url,
                "tags": chunk.tags,
                "chunk_index": chunk.chunk_index,
            },
        })
    index.upsert(vectors=records, namespace=namespace)

def query(vector, top_k, namespace=""):
    res = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=namespace)
    return res.matches
