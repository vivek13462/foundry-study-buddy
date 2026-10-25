"""Stage 10b: let Agent Framework drive the Stage 10 tool-calling loop."""

import asyncio
import json
import os
from collections.abc import Callable

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv

from stage10_workflow import (
    INSTRUCTIONS,
    MultiplyArguments,
    SearchArguments,
    execute_tool,
)


QUESTION = (
    "Look up how many dimensions Study Buddy's embedding model returns. "
    "Then use the multiplication tool to calculate the total number of "
    "vector components for 1000 document chunks. Cite the dimension source. "
    "Count components, not bytes."
)


def build_agent(
    client: FoundryChatClient,
    retrieve: Callable[[str], list[dict[str, str]]],
) -> Agent:
    @tool(schema=SearchArguments)
    async def search_knowledge(query: str) -> str:
        """Search Study Buddy's knowledge base for project facts."""
        print(f"\nTool request: search_knowledge({query!r})")
        # Stage 7 uses synchronous network clients; don't block the event loop.
        result = await asyncio.to_thread(
            execute_tool, "search_knowledge", json.dumps({"query": query}), retrieve
        )
        print(f"Tool result: {json.dumps(result)}")
        return json.dumps(result)

    @tool(schema=MultiplyArguments)
    def multiply_numbers(left: int, right: int) -> str:
        """Multiply two integers, each with absolute value at most one billion."""
        print(f"\nTool request: multiply_numbers({left}, {right})")
        result = execute_tool(
            "multiply_numbers", json.dumps({"left": left, "right": right}), retrieve
        )
        print(f"Tool result: {json.dumps(result)}")
        return json.dumps(result)

    client.function_invocation_configuration.update(
        {
            "max_iterations": 5,
            "max_function_calls": 4,
            "max_duration_seconds": 90.0,
            "allow_concurrent_invocation": False,
        }
    )
    return Agent(
        client=client,
        name="StudyBuddyFramework",
        instructions=INSTRUCTIONS,
        tools=[search_knowledge, multiply_numbers],
        default_options={"max_tokens": 800, "allow_multiple_tool_calls": False},
    )


async def run_agent(agent: Agent, question: str = QUESTION) -> str:
    print(f"Q: {question}")
    response = await agent.run(question)
    # Frameworks can turn tool exceptions into model-visible error results.
    # Don't report a recovered prose answer as a successful run of failed tools.
    for message in response.messages:
        for content in message.contents:
            if content.exception or content.type == "error":
                raise RuntimeError(
                    f"Agent encountered a tool or response error: {content.exception or content}"
                )
    if response.continuation_token or response.finish_reason in ("length", "content_filter"):
        raise RuntimeError(f"Agent response did not finish normally: {response.finish_reason}")
    if not response.text.strip():
        raise RuntimeError("Agent returned no final text")
    print(f"\nFinal answer:\n{response.text}")
    return response.text


async def main() -> None:
    load_dotenv()
    from stage07_rag import retrieve

    async with DefaultAzureCredential() as credential:
        async with AIProjectClient(
            endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            credential=credential,
        ) as project:
            client = FoundryChatClient(
                project_client=project,
                model=os.environ["CHAT_DEPLOYMENT"],
            )
            async with client.client:
                async with build_agent(client, retrieve) as agent:
                    await run_agent(agent)


if __name__ == "__main__":
    asyncio.run(main())
