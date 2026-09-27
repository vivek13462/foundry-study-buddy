"""
Stage 4 — Structured outputs.

Goal: stop parsing free-form prose. Make the model return JSON that matches a
schema WE define, validated into real Python objects we can use in code.

We ask Study Buddy to turn a topic into a flashcard, and get back a typed
`Flashcard` object instead of a paragraph.
"""

import os
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]

project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
client = project.get_openai_client()


# 1. Define the SHAPE of the answer we want, as a Pydantic model.
#    Field descriptions are sent to the model as hints — treat them as prompts.
class Difficulty(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class Flashcard(BaseModel):
    concept: str = Field(description="The name of the concept being taught")
    definition: str = Field(description="A one-sentence beginner-friendly definition")
    difficulty: Difficulty
    quiz_question: str = Field(description="One question to test understanding")
    keywords: list[str] = Field(description="2-4 related search keywords")


SYSTEM_PROMPT = "You are Study Buddy, a tutor for people new to Azure AI."

topic = "vector embeddings"

# 2. Ask the model and parse the reply DIRECTLY into our model.
#    `text_format=Flashcard` tells the API to return JSON matching the schema.
response = client.responses.parse(
    model=CHAT_DEPLOYMENT,
    instructions=SYSTEM_PROMPT,
    input=f"Create a flashcard for the topic: {topic}",
    text_format=Flashcard,
)

# 3. `output_parsed` is a real, validated Flashcard instance — not a string.
card: Flashcard = response.output_parsed

print("type of result:", type(card).__name__)
print("concept       :", card.concept)
print("definition    :", card.definition)
print("difficulty    :", card.difficulty.value)
print("quiz_question :", card.quiz_question)
print("keywords      :", card.keywords)

# 4. Because it's structured, code can USE it — no fragile text parsing.
print("\n--- proof it is usable data ---")
print("first keyword upper-cased:", card.keywords[0].upper())
print("is advanced topic?       :", card.difficulty == Difficulty.advanced)
print("\nraw JSON we could store in a DB:\n", card.model_dump_json(indent=2))
