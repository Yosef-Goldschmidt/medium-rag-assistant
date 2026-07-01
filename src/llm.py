import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["NBUECSE_API_KEY"]
base_url = os.environ["NBUECSE_BASE_URL"]
client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
CHAT_MODEL = "NBUECSE-gpt-5-mini"

def complete(system: str, user: str) -> str:
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])#tpye: ignore
    return resp.choices[0].message.content

# print(complete("You are a helpful assistant.", "Say hello in exactly one word."))