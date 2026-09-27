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
| 5   | Embeddings | Embeddings | ⬜ next |
| 6   | Vector search | Vector DB / vector + hybrid search (Azure AI Search) | ⬜ |
| 7   | RAG | Retrieval-Augmented Generation | ⬜ |
| 8   | Tool / function calling | Function/tool calling | ⬜ |
| 9   | AI agents | Single hosted agent (Foundry Agent Service) | ⬜ |
| 9b  | Skills | Attaching tools/knowledge to an agent ("skills") | ⬜ NEW |
| 10  | Agentic workflows | Multi-step agent loop (DIY harness) | ⬜ |
| 10b | Agent harness & orchestration | Runtime loop + Microsoft Agent Framework (sequential/concurrent/group-chat/handoff/Magentic) | ⬜ NEW |
| 10c | Multi-agent systems | Connected / orchestrated agents | ⬜ NEW |
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
- Skill = an agent-facing capability = a tool/knowledge source attached to an agent.
