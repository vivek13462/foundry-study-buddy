# Foundry Study Buddy — Learning Roadmap

One incremental Python project that teaches modern AI/LLM app concepts on
Microsoft Azure AI Foundry. Each stage adds exactly one concept.

## Stages

| #   | Stage | Concept(s) | Status |
|-----|-------|-----------|--------|
| 1   | Basic LLM chat | LLM interaction | ✅ done |
| 2   | Tokenization | Tokens, context windows/management | ✅ done |
| 3   | Prompt engineering | System prompts, few-shot | ✅ done |
| 4   | Structured outputs | JSON schema, Pydantic | ✅ done |
| 5   | Embeddings | Embeddings | ✅ done |
| 6   | Vector search | Vector DB / vector + hybrid search (Azure AI Search) | ✅ done |
| 7   | RAG | Retrieval-Augmented Generation | ✅ done |
| 8   | Tool / function calling | Function/tool calling | ✅ done |
| 9   | AI agents | Managed prompt agent and conversation (Foundry Agent Service) | ✅ done |
| 9b  | Skills | Reusable SKILL.md instructions, explicit local injection first | ✅ done |
| 10  | Agentic workflows | Multi-step agent loop (DIY harness) | ✅ done |
| 10b | Agent harness & orchestration | Replace the DIY loop with Agent Framework; distinguish tool loops from workflow orchestration | ✅ done |
| 10c | Multi-agent systems | Sequential Researcher -> Tutor; compare application-owned and service-managed agents | ⬜ next |
| 10d | A2A | Agent-to-agent protocol across systems (Foundry A2A tool) | ⬜ NEW |
| 11  | Memory | Memory systems (threads/state) | ⬜ |
| 12  | MCP | Model Context Protocol (tools/data) | ⬜ |
| 13  | Streaming | Streaming responses | ⬜ |
| 14  | Caching & latency | Caching, latency optimization | ⬜ |
| 15  | Guardrails | Guardrails, hallucination mitigation | ⬜ |
| 16  | Observability | Tracing (App Insights + OpenTelemetry) | ⬜ |
| 17  | Evaluation | Evaluation frameworks (Foundry Evaluation SDK) | ⬜ |
| 18  | Feedback loops | User feedback loops | ⬜ |
| 19  | Fine-tuning vs RAG | When to fine-tune vs retrieve | ⬜ |

## Decisions
- Multi-agent: cover BOTH Microsoft Agent Framework and Foundry Connected Agents.
- "Harness" = the agent runtime loop (build by hand, then use Agent Framework).
- Agent arc runs in order, after RAG (7) and tool calling (8).

## Key concept mapping
- MCP  = connects an agent to TOOLS/DATA.
- A2A  = connects an agent to OTHER AGENTS (cross-system).
- Skill = reusable task instructions, optionally with reference files and scripts; not just a tool.
- Stage 9b explicitly injects one local skill into a prompt agent. It does not register a Foundry Skills API resource or implement automatic discovery/on-demand loading.
- Stage 10b uses a basic Agent with two local tools, not the larger Harness preset. Multi-agent orchestration patterns are introduced in Stage 10c.
