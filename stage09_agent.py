"""Stage 9: a managed prompt agent and a two-turn conversation in Foundry."""

import os

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


AGENT_NAME = "study-buddy-stage09"
AGENT_INSTRUCTIONS = """\
You are Study Buddy, a patient tutor for people new to Azure AI.
Explain concepts in plain language and define unfamiliar terms.
Keep each answer under 120 words.
End each answer with one line starting with "Try this:".
If you don't know an answer, say so instead of inventing facts.
You have no tools or access to the user's documents in this stage.
"""


def main() -> None:
    load_dotenv()
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    deployment = os.environ["CHAT_DEPLOYMENT"]

    with DefaultAzureCredential() as credential:
        with AIProjectClient(endpoint=endpoint, credential=credential) as project:
            agent = project.agents.create_version(
                agent_name=AGENT_NAME,
                definition=PromptAgentDefinition(
                    model=deployment,
                    instructions=AGENT_INSTRUCTIONS,
                ),
            )
            print(f"Agent: {agent.name}, version: {agent.version}")

            # This client routes requests to the named agent rather than a raw model.
            with project.get_openai_client(agent_name=agent.name) as client:
                conversation = client.conversations.create()
                print(f"Conversation: {conversation.id}")

                questions = [
                    "My study topic today is vector embeddings. "
                    "Explain them using a simple analogy.",
                    "Give me one quiz question about my study topic, "
                    "without revealing the answer.",
                ]

                for question in questions:
                    response = client.responses.create(
                        conversation=conversation.id,
                        input=question,
                    )
                    if response.status != "completed":
                        raise RuntimeError(
                            f"Agent response {response.id} did not complete: "
                            f"status={response.status}, error={response.error}, "
                            f"details={response.incomplete_details}"
                        )
                    if not response.output_text.strip():
                        raise RuntimeError(
                            f"Agent response {response.id} returned no text; "
                            f"inspect its output items: {response.output!r}"
                        )

                    print(f"\nQ: {question}")
                    print(f"Answer:\n{response.output_text}")
                    if response.usage is not None:
                        print(
                            f"[input tokens: {response.usage.input_tokens}, "
                            f"output tokens: {response.usage.output_tokens}]"
                        )

                print("\nAgent and conversation retained in Foundry.")
                print("Rerunning creates another agent version and a new conversation.")


if __name__ == "__main__":
    main()
