"""
Stage 3 — Prompt engineering.

Goal: control the model's behavior with a well-crafted *system prompt*
(persona + rules + output format), and see how instructions change the answer.

We compare:
  A) no system prompt   (default, generic model)
  B) a strong system prompt that defines our "Study Buddy" persona
  C) the same, plus a few-shot example that pins the answer style
"""

import os

import tiktoken
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]

encoding = tiktoken.get_encoding("o200k_base")
project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
client = project.get_openai_client()

# This becomes the Study Buddy's permanent "constitution". Reused in later stages.
SYSTEM_PROMPT = """\
You are "Study Buddy", a patient tutor for people new to Azure AI.

Rules:
- Explain like the reader is a smart beginner. Avoid unexplained jargon.
- Keep answers under 120 words.
- Always end with one line starting with "Try this:" giving a tiny exercise.
- If you are not sure or the question is outside Azure/AI, say so honestly
  instead of guessing.
"""

# A few-shot example: show the model ONE ideal Q/A so it copies the style.
FEWSHOT = [
    {"role": "user", "content": "What is a token?"},
    {"role": "assistant", "content": (
        "A token is a small chunk of text (about 4 characters) that the model "
        "reads and bills by. Common words are one token; rare words split into "
        "several.\nTry this: count the tokens in your own name."
    )},
]

QUESTION = "What is an embedding?"


def ask(label: str, *, instructions: str | None, prefix_messages: list | None = None):
    messages = list(prefix_messages or []) + [{"role": "user", "content": QUESTION}]
    kwargs = {"model": CHAT_DEPLOYMENT, "input": messages}
    if instructions:
        kwargs["instructions"] = instructions  # <-- the system prompt goes here
    resp = client.responses.create(**kwargs)
    print(f"\n===== {label} =====")
    print(resp.output_text)
    print(f"[input tokens: {resp.usage.input_tokens}, output: {resp.usage.output_tokens}]")


# A) Baseline: no instructions at all -> generic, unbounded answer.
ask("A. No system prompt", instructions=None)

# B) Strong system prompt -> persona, length limit, and "Try this:" format.
ask("B. Strong system prompt", instructions=SYSTEM_PROMPT)

# C) System prompt + few-shot -> style locked in by example.
ask("C. System prompt + few-shot", instructions=SYSTEM_PROMPT, prefix_messages=FEWSHOT)

# Show the cost of the instructions themselves.
print(f"\nSYSTEM_PROMPT costs {len(encoding.encode(SYSTEM_PROMPT))} tokens on EVERY call.")
print(f"FEWSHOT adds {sum(len(encoding.encode(m['content'])) for m in FEWSHOT)} more tokens.")
