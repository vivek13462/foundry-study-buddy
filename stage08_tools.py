"""
Stage 8 — Function / tool calling.

Goal: let the model DECIDE to call our Python functions. We describe some tools;
the model reads the question, picks a tool, and returns the arguments it wants.
WE run the function and hand the result back; the model writes the final answer.

Key idea: the model never runs your code. It only *requests* a call. You stay
in control of what actually executes.
"""

import os
import json

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]

project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
client = project.get_openai_client()


# --- 1. The real Python functions the model may ask us to run ----------------
def calculator(expression: str) -> dict:
    """Safely evaluate a simple arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return {"error": "unsupported characters"}
    try:
        return {"result": eval(expression, {"__builtins__": {}}, {})}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def word_count(text: str) -> dict:
    """Count words in a piece of text."""
    return {"word_count": len(text.split())}


# Map tool name -> function, so we can dispatch by name.
TOOL_IMPLEMENTATIONS = {
    "calculator": calculator,
    "word_count": word_count,
}

# --- 2. Describe the tools to the model (JSON Schema per parameter) -----------
#    The 'description' fields are prompts: they teach the model WHEN to use each.
TOOLS = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Use for any math.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '3 * (4 + 5)'"}
            },
            "required": ["expression"],
        },
    },
    {
        "type": "function",
        "name": "word_count",
        "description": "Count the number of words in a given text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def ask(question: str) -> None:
    print(f"\n{'='*70}\nQ: {question}")

    # --- 3. First call: give the model the question AND the tool menu --------
    response = client.responses.create(
        model=CHAT_DEPLOYMENT,
        tools=TOOLS,
        input=question,
    )

    # --- 4. Did it ask to call any tools? Run them. -------------------------
    tool_outputs = []
    for item in response.output:
        if item.type == "function_call":
            args = json.loads(item.arguments)
            print(f"  model wants: {item.name}({args})")
            result = TOOL_IMPLEMENTATIONS[item.name](**args)  # WE run it
            print(f"  tool result: {result}")
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,         # ties result to the request
                    "output": json.dumps(result),
                }
            )

    # --- 5. No tool needed? The model already answered. ---------------------
    if not tool_outputs:
        print(f"Answer (no tool): {response.output_text}")
        return

    # --- 6. Second call: hand results back so the model writes the answer ----
    final = client.responses.create(
        model=CHAT_DEPLOYMENT,
        previous_response_id=response.id,  # continues the same exchange
        input=tool_outputs,
    )
    print(f"Answer: {final.output_text}")


if __name__ == "__main__":
    ask("What is 1234 * 5678?")                       # -> calculator
    ask("How many words are in 'the quick brown fox'?")  # -> word_count
    ask("Who wrote Romeo and Juliet?")                # -> no tool, direct answer
