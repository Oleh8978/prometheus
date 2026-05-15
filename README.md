# Gemini hackathon starter

End-to-end template for the **Arize @ Google Cloud Partnerships Hackathon** track: a small **Google ADK** agent (pattern from [google/adk-samples personalized-shopping](https://github.com/google/adk-samples/tree/main/python/agents/personalized-shopping)), **OpenInference** instrumentation for ADK, **[phoenix.otel.register](https://arize.com/docs/phoenix/get-started/get-started-tracing)** for Phoenix Cloud tracing, and **Gemini CLI** MCP config for `@arizeai/phoenix-mcp`.

This repo uses a **tiny in-memory catalog** so you can run locally in minutes (no PyTorch, Pyserini, or multi-gigabyte product downloads). The agent still exposes the same **search** / **click** tools and a shopping-focused system prompt derived from the upstream sample.

## Prerequisites

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/)
- Google auth for Gemini: either `GOOGLE_API_KEY` **or** Vertex (`gcloud auth application-default login` + project/location)
- Phoenix Cloud API key ([Phoenix](https://app.phoenix.arize.com))

## 10-minute quickstart

1. **Clone and install**
  ```bash
   cd gemini-hackathon
   cp .env.example .env
   # Edit .env: PHOENIX_API_KEY, PHOENIX_COLLECTOR_ENDPOINT (Hostname with /s/...), and either GOOGLE_API_KEY or Vertex settings.
   uv sync
  ```
2. **Run a traced shopping turn**
  ```bash
   make run MESSAGE='Find a floral dress in size M'
  ```
3. **Open Phoenix** — project name defaults to `PHOENIX_PROJECT_NAME` (`gemini-hackathon`). Confirm LLM and tool spans appear.
4. **(Optional) ADK CLI**
  ```bash
   make run-adk
   # Find a floral dress in size M
  ```
   This path also loads `.env` and initializes Phoenix tracing.

### Phoenix MCP (Gemini CLI)

Phoenix MCP runs **inside Gemini CLI**, not inside the Python ADK process. After traces are flowing from `make run`, you can inspect the same Phoenix space from the CLI. Setup patterns and clients are covered in [Phoenix MCP server](https://arize.com/docs/phoenix/integrations/phoenix-mcp-server).

1. **Configure MCP** — Ensure `[.gemini/settings.json](.gemini/settings.json)` in this repo (or `~/.gemini/settings.json`) includes the `phoenix` server with `@arizeai/phoenix-mcp@latest`. Set `--baseUrl` to your Phoenix space hostname (same idea as `PHOENIX_COLLECTOR_ENDPOINT`: `https://app.phoenix.arize.com/s/your-space`) and set `--apiKey` to your Phoenix API key (`px_live_...`), or keep keys only in env if your CLI supports that pattern.
2. **Export your API key** in the shell that launches Gemini CLI (if the MCP server reads it from the environment):
  ```bash
   export PHOENIX_API_KEY=...
  ```
3. **Start Gemini CLI** from the repo root (or merge the `mcpServers` block into your global Gemini config). Restart the CLI if you just changed MCP settings.
4. **Agent queries Phoenix via MCP (runtime superpower)** — With `@arizeai/phoenix-mcp` configured, the assistant gets **tools** over your Phoenix workspace (traces, sessions, experiments, prompts, datasets, and more). Try prompts such as:
  - *“In Phoenix, show me the last 3 traces in my **gemini-hackathon** project.”*
  - *“In Phoenix, summarize my latest experiment results.”*
  - *“In Phoenix, create a prompt that classifies user intent.”*
   Additional ideas (sessions, annotation configs, datasets): [Using the Phoenix MCP server](https://arize.com/docs/phoenix/integrations/phoenix-mcp-server#using-the-phoenix-mcp-server).
5. **(Optional)** The same file defines **Phoenix Docs MCP** (`phoenix-docs`) for in-IDE Phoenix documentation.

More context: [Phoenix docs](https://arize.com/docs/phoenix).

## Layout


| Path                       | Purpose                                                                                                                                |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`                | This quickstart                                                                                                                        |
| `.env.example`             | `PHOENIX_`*, `GOOGLE_`*, optional `GEMINI_MODEL`                                                                                       |
| `.gemini/settings.json`    | Phoenix MCP + Phoenix Docs MCP                                                                                                         |
| `agent/main.py`            | One-shot CLI run with tracing                                                                                                          |
| `agent/instrumentation.py` | `[phoenix.otel.register(..., auto_instrument=True)](https://arize.com/docs/phoenix/integrations/python/google-adk/google-adk-tracing)` |
| `agent/shopping_demo/`     | ADK `root_agent`, prompt, tools, mini webshop                                                                                          |
| `Makefile`                 | `make setup`, `make run`, `make run-adk`                                                                                               |


## Upstream credit

Agent structure and prompts are adapted from **Google ADK Samples** — [personalized-shopping](https://github.com/google/adk-samples/tree/main/python/agents/personalized-shopping) (Apache-2.0). Replace `shopping_demo/mini_webshop.py` with the full WebShop stack when you need the original fidelity.

## License

Apache-2.0 — see [LICENSE](LICENSE).