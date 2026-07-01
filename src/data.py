import ast
from collections.abc import Iterator
from dataclasses import dataclass
import csv
import re

@dataclass
class Article:
    article_id: str
    title: str
    text: str
    url: str
    authors: list[str]
    timestamp: str
    tags: list[str]

def parse_list(raw:str) -> list[str]:
    try:
        res = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        res = []
    return res

CAPTION = re.compile(r"^Photo by .+? (on|from) \S+$")


def clean_text(text: str) -> str:
    lines = text.split("\n")
    kept = [line for line in lines if not CAPTION.match(line)]
    return " ".join(" ".join(kept).split())


def load_articles(csv_path: str, limit: int | None = None) -> Iterator[Article]:
    with open(csv_path, encoding= "utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield Article(article_id=f"{i:05d}", title=row["title"], text=clean_text(row["text"]), url=row["url"], authors=parse_list(row["authors"]), timestamp=row["timestamp"], tags=parse_list(row["tags"]))






#tests
# a = Article(article_id="1", title="t", text="x", url="", authors=[], timestamp="", tags=[])
# print(a)
# print(parse_list("['Health', 'Science']"))
# print(parse_list(""))
# path = r"C:/Users/golde/PycharmProjects/RAG Assignment/medium-english-50mb.csv"
# #
# for article in load_articles(path, 2):
# #     print(article.title)
# #     print(article.authors)
# #     print(article.tags)
#     print(article.text[:200])