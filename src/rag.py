from retriever import retrieve, dedup_by_article
from prompt import SYSTEM_PROMPT, build_user_prompt
from llm import complete
import config


def _is_list_query(question: str) -> bool:
    """Heuristic: is the user asking to LIST several distinct articles?"""
    q = question.lower()
    return "list" in q or "distinct" in q


def _get_context_chunks(question: str):
    """Pick the retrieval strategy based on the query type."""
    if _is_list_query(question):
        # Wider pool -> dedup to distinct articles -> hand the LLM several candidates
        # so it can pick the ones actually on-topic (the prompt tells it to list ~3).
        pool = retrieve(question, config.LIST_POOL, min_score=config.MIN_SCORE)
        return dedup_by_article(pool)[:config.LIST_CANDIDATES]
    # Normal question: top_k chunks (may include several from one article).
    return retrieve(question, config.TOP_K, min_score=config.MIN_SCORE)


def answer_question(question):
    chunks = _get_context_chunks(question)
    user_prompt = build_user_prompt(question, chunks)
    response = complete(SYSTEM_PROMPT, user_prompt)

    context = [
        {
            "article_id": c.article_id,
            "title": c.title,
            "chunk": c.chunk,
            "score": c.score,
        }
        for c in chunks
    ]

    return {
        "response": response,
        "context": context,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }


if __name__ == "__main__":
    for q in [
        # --- assignment's own example questions (one per query type) ---
        "Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author.",
        "List exactly 3 articles about education. Return only the titles.",
        "Find an article that argues past pandemics (such as the bubonic plague) can spur innovation and recovery, and summarise its central argument.",
        "I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?",
        # --- extra coverage ---
        "Find an article about smell training and the brain. Give the title and author.",
        "List 3 distinct articles about writing. Return only the titles.",
        "Summarise the central idea of an article about remote work.",
        # --- guardrail: should refuse (out of corpus) ---
        "What is the capital of France?",
    ]:
        print("Q:", q)
        print(answer_question(q)["response"])
        print("-" * 40)
