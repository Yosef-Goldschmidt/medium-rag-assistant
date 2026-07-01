import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["NBUECSE_API_KEY"]
base_url = os.environ["NBUECSE_BASE_URL"]
client = OpenAI(api_key = api_key, base_url = base_url, timeout=60)
EMBED_MODEL = "NBUECSE-text-embedding-3-small"

def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]

