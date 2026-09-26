"""
Stage 1 — Basic LLM interaction with Azure AI Foundry.

Goal: send one message to a deployed model and print its answer.
This is the "hello world" that every later stage builds on.
"""

import os

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

# 1. Load configuration from the .env file into environment variables.
load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]

# 2. Connect to your Foundry project.
#    DefaultAzureCredential uses your `az login` identity — no API keys in code.
project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

# 3. Get an OpenAI-compatible client that is pre-wired to your project.
client = project.get_openai_client()

# 4. Send a single prompt using the Responses API and print the reply.
response = client.responses.create(
    model=CHAT_DEPLOYMENT,
    input="In five sentences, what is Azure Service Bus?",
)

response = client.responses.create(
    model=CHAT_DEPLOYMENT,
    input="In one sentences, what is Azure Event Hub?",
)

print("Model reply:\n")
print(response.output_text)
print(response.model)
print(response.id)

# 5. Peek at token usage — we will care a lot about this in Stage 2.
usage = response.usage
print("\n--- token usage ---")
print(f"input tokens : {usage.input_tokens}")
print(f"output tokens: {usage.output_tokens}")
print(f"total tokens : {usage.total_tokens}")
