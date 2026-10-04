"""
Stage 7 — Retrieval-Augmented Generation (RAG).

Goal: connect retrieval (Stage 6) to the model (Stage 1-4) so Study Buddy
answers from OUR knowledge base instead of its training data.

The RAG loop:
  1. RETRIEVE  - search the index for chunks relevant to the question.
  2. AUGMENT   - put those chunks into the prompt as "context".
  3. GENERATE  - ask the model to answer ONLY from that context, with citations.

Prerequisite: run stage06_vector_search.py once so the index exists.
"""

import os

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]
EMBED_DEPLOYMENT = os.environ["EMBED_DEPLOYMENT"]
SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_INDEX = os.environ.get("SEARCH_INDEX", "study-buddy")
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY") or ""

search_credential = AzureKeyCredential(SEARCH_API_KEY) if SEARCH_API_KEY else DefaultAzureCredential()
project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
openai_client = project.get_openai_client()
search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, credential=search_credential)

resource_endpoint, separator, _ = PROJECT_ENDPOINT.partition("/api/projects/")
if not separator:
    raise ValueError("FOUNDRY_PROJECT_ENDPOINT must include '/api/projects/<project-name>'")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default",
)
embedding_client = OpenAI(
    base_url=f"{resource_endpoint}/openai/v1/",
    api_key=token_provider,
)

# The grounding rules. "Answer ONLY from context" + "say you don't know" is the
# core anti-hallucination instruction of RAG.
RAG_SYSTEM_PROMPT = """\
You are Study Buddy, a tutor for people new to Azure AI.
Answer the user's question using ONLY the numbered sources in the Context.
- Cite the sources you use like [1], [2].
- If the Context does not contain the answer, say:
  "I don't have that in my knowledge base." Do NOT use outside knowledge.
- Keep the answer under 120 words.
"""


def retrieve(question: str, k: int = 3) -> list[dict]:
    """Step 1: hybrid search for the most relevant chunks."""
    query_vector = embedding_client.embeddings.create(
        model=EMBED_DEPLOYMENT, input=[question]
    ).data[0].embedding

    vq = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="chunk_vector")
    results = search_client.search(
        search_text=question,          # hybrid: keyword + vector
        vector_queries=[vq],
        select=["title", "chunk"],
        top=k,
    )
    return [{"title": r["title"], "chunk": r["chunk"]} for r in results]


def build_context(chunks: list[dict]) -> str:
    """Step 2: format retrieved chunks as numbered sources the model can cite."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c['title']}: {c['chunk']}")
    return "\n".join(lines)


def answer(question: str) -> None:
    """The full RAG pipeline."""
    chunks = retrieve(question)
    context = build_context(chunks)

    # Step 3: generate, grounded in the context.
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    resp = openai_client.responses.create(
        model=CHAT_DEPLOYMENT,
        instructions=RAG_SYSTEM_PROMPT,
        input=user_message,
    )

    print(f"\n{'='*70}\nQ: {question}")
    print(f"\nRetrieved sources:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['title']}")
    print(f"\nAnswer:\n{resp.output_text}")


if __name__ == "__main__":
    # In-knowledge questions -> grounded, cited answers.
    answer("What is an embedding and how many dimensions does Study Buddy use?")
    answer("Why does RAG reduce hallucinations?")
    # Out-of-knowledge question -> should refuse, NOT make something up.
    answer("What is the capital of Australia?")
