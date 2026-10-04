"""
Stage 6 — Vector search with Azure AI Search.

Goal: move from the in-memory brute-force search of Stage 5 to a REAL vector
database. We will:
  1. Create a search index that has a vector field.
  2. Embed our knowledge chunks and upload them.
  3. Run a pure VECTOR query (semantic).
  4. Run a HYBRID query (vector + keyword) and see the difference.

Run this once; the index persists in your Azure AI Search service for Stage 7.
"""

import os

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

from knowledge import KNOWLEDGE

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
EMBED_DEPLOYMENT = os.environ["EMBED_DEPLOYMENT"]
SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_INDEX = os.environ.get("SEARCH_INDEX", "study-buddy")
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY") or ""

EMBED_DIMS = 1536  # text-embedding-3-small

# Keyless (DefaultAzureCredential) if no key is set; otherwise use the admin key.
search_credential = AzureKeyCredential(SEARCH_API_KEY) if SEARCH_API_KEY else DefaultAzureCredential()

# Embeddings use the resource-level OpenAI inference endpoint.
resource_endpoint, separator, _ = PROJECT_ENDPOINT.partition("/api/projects/")
if not separator:
    raise ValueError("FOUNDRY_PROJECT_ENDPOINT must include '/api/projects/<project-name>'")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default",
)
openai_client = OpenAI(
    base_url=f"{resource_endpoint}/openai/v1/",
    api_key=token_provider,
)


def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]


# --- 1. Create the index (schema = fields + how to search the vector) --------
def create_index() -> None:
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=search_credential)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="chunk", type=SearchFieldDataType.String),  # keyword-searchable text
        SearchField(
            name="chunk_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIMS,
            vector_search_profile_name="hnsw-profile",  # links field -> algorithm
        ),
    ]

    # HNSW = the graph algorithm that makes nearest-neighbor search fast at scale.
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-config")],
    )

    index = SearchIndex(name=SEARCH_INDEX, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"index '{SEARCH_INDEX}' ready")


# --- 2. Embed chunks and upload them as documents ----------------------------
def upload_documents() -> None:
    vectors = embed([k["chunk"] for k in KNOWLEDGE])  # batch-embed all chunks
    documents = [
        {"id": k["id"], "title": k["title"], "chunk": k["chunk"], "chunk_vector": v}
        for k, v in zip(KNOWLEDGE, vectors)
    ]
    client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, credential=search_credential)
    result = client.upload_documents(documents=documents)
    print(f"uploaded {sum(1 for r in result if r.succeeded)} documents")


# --- 3. Pure vector query: find chunks nearest to the query's meaning --------
def vector_query(query: str, k: int = 3) -> None:
    client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, credential=search_credential)
    query_vector = embed([query])[0]

    vq = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="chunk_vector")
    results = client.search(search_text=None, vector_queries=[vq], select=["title", "chunk"])

    print(f"\n=== VECTOR search: {query!r} ===")
    for r in results:
        print(f"[{r['@search.score']:.4f}] {r['title']}")


# --- 4. Hybrid query: vector + keyword together ------------------------------
def hybrid_query(query: str, k: int = 3) -> None:
    client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, credential=search_credential)
    query_vector = embed([query])[0]

    vq = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="chunk_vector")
    # Passing BOTH search_text (keyword/BM25) and vector_queries = hybrid search.
    results = client.search(search_text=query, vector_queries=[vq], select=["title", "chunk"], top=k)

    print(f"\n=== HYBRID search: {query!r} ===")
    for r in results:
        print(f"[{r['@search.score']:.4f}] {r['title']}")


if __name__ == "__main__":
    create_index()
    upload_documents()
    vector_query("How does semantic search find relevant text?")
    hybrid_query("What is HNSW in Azure AI Search?")
