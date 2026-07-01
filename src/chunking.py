import tiktoken
from dataclasses import dataclass
from data import Article, load_articles

enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

@dataclass
class Chunk:
    chunk_id: str
    article_id: str
    chunk_index: int
    text: str
    title: str
    authors: list[str]
    url: str
    tags: list[str]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = enc.encode(text)
    start = 0
    chunks_list = []
    while start < len(tokens):
        chunks_list.append(enc.decode(tokens[start : start + chunk_size]))
        start += chunk_size - overlap

    return chunks_list

def chunk_article(article: Article, chunk_size: int, overlap_ratio: float) -> list[Chunk]:
    overlap = int(chunk_size * overlap_ratio)
    pieces = chunk_text(article.text, chunk_size, overlap)
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(Chunk(f"{article.article_id}-{i}", article.article_id, i, piece, article.title, article.authors, article.url, article.tags))

    return chunks


#tests
# print(count_tokens("hello world"))
# print(Chunk(chunk_id="0-0", article_id="0", chunk_index=0, text="hi", title="t", authors=["a"], url="u", tags=[]))
# print(chunk_text("word "*1000, 100, 20))
# path = r"C:/Users/golde/PycharmProjects/RAG Assignment/medium-english-50mb.csv"
# a = next(load_articles(path, 1))
# for chunk in chunk_article(a, 512, 0.15):
#     print(chunk.chunk_id)
#     print(count_tokens(chunk.text))
#     print("END :", chunk.text[-300:])  # tail of this chunk
#     print("START:", chunk.text[:300])  # head of this chunk