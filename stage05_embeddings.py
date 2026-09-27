"""
Stage 5 — Embeddings.

Goal: turn text into a *vector* (a list of numbers) that captures MEANING, then
measure how "close" two pieces of text are with cosine similarity.

This is the math that makes vector search (Stage 6) and RAG (Stage 7) possible:
similar meaning -> vectors that point in the same direction -> high similarity.
"""

import os
import math

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
EMBED_DEPLOYMENT = os.environ["EMBED_DEPLOYMENT"]

resource_endpoint, separator, _ = PROJECT_ENDPOINT.partition("/api/projects/")
if not separator:
    raise ValueError("FOUNDRY_PROJECT_ENDPOINT must include '/api/projects/<project-name>'")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default",
)
client = OpenAI(
    base_url=f"{resource_endpoint}/openai/v1/",
    api_key=token_provider,
)

print("PROJECT_ENDPOINT:", PROJECT_ENDPOINT)
print("EMBED_DEPLOYMENT:", EMBED_DEPLOYMENT)

def embed(texts: list[str]) -> list[list[float]]:
    """Turn a batch of strings into a list of embedding vectors."""
    # Batching (a list of inputs) is cheaper and faster than one call per string.
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [item.embedding for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """1.0 = same direction (very similar meaning), 0 = unrelated, -1 = opposite."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# --- 1. See what an embedding actually IS ------------------------------------
sample = embed(["Azure AI Foundry"])[0]
print("=== what an embedding looks like ===")
print(f"dimensions      : {len(sample)}  (text-embedding-3-small = 1536 numbers)")
print(f"first 5 numbers : {[round(x, 4) for x in sample[:5]]}")
print()

# --- 2. Similar meaning -> high similarity, even with different words ---------
pairs = [
    ("How do I reset my password?", "I forgot my login credentials"),   # same idea
    ("How do I reset my password?", "What time does the store close?"),  # unrelated
    ("car", "automobile"),                                              # synonyms
    ("car", "banana"),                                                  # unrelated
]
print("=== semantic similarity (cosine) ===")
for t1, t2 in pairs:
    v1, v2 = embed([t1, t2])
    print(f"{cosine_similarity(v1, v2):+.3f}   {t1!r}  vs  {t2!r}")
print()

# --- 3. A tiny 'search': rank a corpus against a query -----------------------
#    This is Stage 6 in miniature, done in memory with brute force.
corpus = [
    "Azure AI Search stores vectors and runs similarity search.",
    "Embeddings convert text into numbers that capture meaning.",
    "Pizza dough needs yeast, flour, water, and salt.",
    "Foundry lets you deploy and manage AI models.",
    "The mitochondria is the powerhouse of the cell.",
]
query = "How does semantic search work?"

corpus_vectors = embed(corpus)
query_vector = embed([query])[0]

ranked = sorted(
    ((cosine_similarity(query_vector, cv), doc) for cv, doc in zip(corpus_vectors, corpus)),
    reverse=True,
)

print(f"=== ranking corpus against query: {query!r} ===")
for score, doc in ranked:
    print(f"{score:+.3f}  {doc}")
