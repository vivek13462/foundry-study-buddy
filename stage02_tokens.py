"""
Stage 2 — Tokenization & context windows.

Goals:
  1. See how text becomes *tokens* (the unit models actually read and bill).
  2. Count tokens locally BEFORE calling the API (with tiktoken).
  3. Compare our local count to the API's reported usage.
  4. Do a simple "context budget" check so we never blow the model's limit.
"""

import os

import tiktoken
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]

# gpt-4o / gpt-4o-mini / gpt-5 family all use the "o200k_base" encoding.
# (Older gpt-4 / gpt-3.5 used "cl100k_base".)
encoding = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Return how many tokens `text` becomes for this model family."""
    return len(encoding.encode(text))


def show_tokenization(text: str) -> None:
    """Print each token so you can SEE how the model chops up text."""
    token_ids = encoding.encode(text)
    pieces = [encoding.decode([tid]) for tid in token_ids]
    print(f"text     : {text!r}")
    print(f"tokens   : {len(token_ids)}")
    print(f"pieces   : {pieces}")


# --- 1. Feel what a token is -------------------------------------------------
print("=== how text splits into tokens ===")
show_tokenization("Azure AI Foundry")
print()
show_tokenization("tokenization")          # one word -> multiple tokens
print()
show_tokenization("supercalifragilistic")  # rare word -> many tokens
print()

# --- 2. Predict cost BEFORE calling the API ---------------------------------
prompt = "In two sentences, what is Azure AI Foundry?"
predicted = count_tokens(prompt)
print(f"=== local prediction ===\nprompt tokens (local, text only): {predicted}\n")

# --- 3. Call the model and compare to the API's own count -------------------
project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
client = project.get_openai_client()

response = client.responses.create(model=CHAT_DEPLOYMENT, input=prompt)
print("=== model reply ===")
print(response.output_text, "\n")

print("=== API-reported usage ===")
print(f"input tokens : {response.usage.input_tokens}   (a bit higher than our {predicted}:")
print("               chat requests add a few 'envelope' tokens per message)")
print(f"output tokens: {response.usage.output_tokens}")
print(f"total tokens : {response.usage.total_tokens}\n")

# --- 4. A context-window budget check ---------------------------------------
# Pretend the deployed model allows this many tokens total (input + output).
# Look up the real number for your model on the Foundry models page.
MODEL_CONTEXT_WINDOW = 128_000
MAX_OUTPUT_TOKENS = 800

def fits_in_context(prompt_text: str) -> bool:
    needed = count_tokens(prompt_text) + MAX_OUTPUT_TOKENS
    print(f"budget: need ~{needed} of {MODEL_CONTEXT_WINDOW} tokens")
    return needed <= MODEL_CONTEXT_WINDOW

print("=== context budget check ===")
big_prompt = prompt + ("\n\nExtra context: " + "blah " * 5000)
print("small prompt fits? ", fits_in_context(prompt))
print("huge prompt fits?  ", fits_in_context(big_prompt))
