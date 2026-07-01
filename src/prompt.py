SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data."

Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

When asked to list multiple articles, return distinct articles. When asked for the title and author, provide both. Be concise."""


def build_user_prompt(question, chunks) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        author = ", ".join(chunk.authors) if chunk.authors else "Unknown"
        header = f"[{i}] Title: {chunk.title} | Author: {author} | Article ID: {chunk.article_id}"
        blocks.append(f"{header}\n{chunk.chunk}")
    context = "\n\n".join(blocks)
    return f"Context from Medium articles:\n\n{context}\n\n---\nQuestion: {question}"