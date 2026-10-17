"""Stage 9b: explicitly load a local skill into a Foundry prompt agent."""

import os
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from stage09_agent import AGENT_INSTRUCTIONS


SKILL_PATH = Path(__file__).resolve().parent / "skills" / "quiz-coach" / "SKILL.md"


def build_instructions() -> str:
    skill = SKILL_PATH.read_text(encoding="utf-8").strip()
    if not skill:
        raise ValueError(f"Skill file is empty: {SKILL_PATH}")
    # Explicit injection, not automatic discovery or native Skills API registration.
    return f"{AGENT_INSTRUCTIONS}\n\nAdditional task instructions:\n{skill}"


def main() -> None:
    load_dotenv()
    instructions = build_instructions()
    print(f"Explicitly loading local skill: {SKILL_PATH}")

    with DefaultAzureCredential() as credential:
        with AIProjectClient(
            endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            credential=credential,
        ) as project:
            agent = project.agents.create_version(
                agent_name="study-buddy-stage09b",
                definition=PromptAgentDefinition(
                    model=os.environ["CHAT_DEPLOYMENT"],
                    instructions=instructions,
                ),
            )
            print(f"Agent: {agent.name}, version: {agent.version}")

            with project.get_openai_client(agent_name=agent.name) as client:
                conversation = client.conversations.create()
                print(f"Conversation: {conversation.id}")

                for question in [
                    "My study topic is vector embeddings. Explain it briefly.",
                    "Quiz me on my study topic. Do not reveal the answer yet.",
                ]:
                    response = client.responses.create(
                        conversation=conversation.id,
                        input=question,
                    )
                    if response.status != "completed":
                        raise RuntimeError(
                            f"Response {response.id} did not complete: "
                            f"status={response.status}, error={response.error}, "
                            f"details={response.incomplete_details}"
                        )
                    if not response.output_text.strip():
                        raise RuntimeError(
                            f"Response {response.id} returned no text: "
                            f"{response.output!r}"
                        )
                    print(f"\nQ: {question}\nAnswer:\n{response.output_text}")
                    if response.usage is not None:
                        print(
                            f"[input tokens: {response.usage.input_tokens}, "
                            f"output tokens: {response.usage.output_tokens}]"
                        )

    print("\nAgent version and conversation retained in Foundry.")
    print("Rerunning reads the skill again and creates a new agent version and conversation.")


if __name__ == "__main__":
    main()
