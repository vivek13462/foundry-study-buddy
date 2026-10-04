"""
Shared tiny knowledge base for Study Buddy (used from Stage 6 onward).

In a real app these would be chunks of your product docs / FAQ / notes,
produced by splitting large documents into ~300-500 token pieces.
Here we hand-write short chunks so the focus stays on the retrieval mechanics.
"""

# Each item becomes one searchable document (one vector) in the index.
KNOWLEDGE = [
    {
        "id": "1",
        "title": "What is Azure AI Foundry",
        "chunk": (
            "Azure AI Foundry is Microsoft's platform for building, deploying, and "
            "managing AI applications and agents. You create a project, deploy models "
            "like GPT and embedding models, and connect services such as Azure AI Search."
        ),
    },
    {
        "id": "2",
        "title": "Embeddings",
        "chunk": (
            "An embedding converts text into a vector of numbers that captures meaning. "
            "Similar text produces vectors that are close together, which enables "
            "semantic search. Study Buddy uses text-embedding-3-small, which returns "
            "1536 dimensions."
        ),
    },
    {
        "id": "3",
        "title": "Vector search in Azure AI Search",
        "chunk": (
            "Azure AI Search stores vectors and runs similarity search using the HNSW "
            "algorithm. Hybrid search combines vector similarity with traditional "
            "keyword (BM25) search and often beats either method alone."
        ),
    },
    {
        "id": "4",
        "title": "Retrieval-Augmented Generation",
        "chunk": (
            "RAG retrieves relevant chunks from a knowledge source and puts them into "
            "the prompt so the model answers from your data instead of only its "
            "training. It reduces hallucinations and lets you update knowledge without "
            "retraining the model."
        ),
    },
    {
        "id": "5",
        "title": "Tokens and context windows",
        "chunk": (
            "Models read text as tokens, roughly four characters each. The context "
            "window is the maximum number of tokens, counting both input and output. "
            "Because the window is limited, RAG retrieves only the most relevant chunks "
            "instead of sending whole documents."
        ),
    },
    {
        "id": "6",
        "title": "Pricing of coffee",
        "chunk": (
            "A medium latte at the campus cafe costs about four dollars. Refills are "
            "half price before 10 AM. This has nothing to do with Azure."
        ),
    },
]
