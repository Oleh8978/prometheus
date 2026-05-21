# Prometheus 🔥
### A self-improving research agent powered by Gemini + Arize Phoenix

> *Most AI agents just run. Prometheus runs, reflects, and rewrites itself.*

**Prometheus** is a multi-step research and synthesis agent built with Google ADK and Gemini 2.5 Flash. It answers complex, multi-part questions by decomposing them, gathering evidence from multiple sources, identifying contradictions, and producing structured analytical reports with citations.

What makes it different: **Prometheus uses Arize Phoenix to observe its own failures, score its own outputs, and autonomously rewrite its system prompt to improve over successive cycles — without human intervention.**

---

## What it does

Ask Prometheus a hard research question:

```
What are the real tradeoffs between vector databases and PostgreSQL pgvector
for production RAG systems in 2025?
```

It doesn't just answer. It works through six structured stages:

| Stage | What happens |
|-------|-------------|
| **1. Decompose** | Breaks the question into focused sub-problems |
| **2. Search** | Runs independent searches per sub-problem (Tavily → DuckDuckGo fallback) |
| **3. Extract** | Pulls structured claims from each source, flags ones needing verification |
| **4. Contradict** | Identifies conflicting information across sources |
| **5. Synthesize** | Produces a structured report: Executive Summary, Key Findings, Evidence & Citations, Conflicting Views, Confidence Assessment, Open Questions |
| **6. Self-evaluate** | Critically reviews citation coverage, reasoning coherence, and hallucination risk before finalizing |

Every step of every run is traced to **Arize Phoenix** via OpenInference — zero manual instrumentation code.

---

## The self-improvement loop

After a batch of runs, a second agent — the **improvement agent** — takes over:

```
Batch run 20 questions → traces auto-flow to Phoenix
            ↓
LLM-as-Judge scores each span on 4 dimensions
  citation_groundedness · reasoning_coherence
  hallucination_risk    · completeness
            ↓
Scores logged back to Phoenix as span annotations
            ↓
Improvement agent uses Phoenix MCP tools at runtime:
  list-traces   → find recent runs
  get-spans     → read what went wrong in detail
  get-prompt-version → read current system prompt
            ↓
Diagnoses 2-3 specific, concrete failure patterns
  (not "cite better" — "URLs present but no author/year makes claims unverifiable")
            ↓
Rewrites system prompt → upsert-prompt → saved to Phoenix Prompts registry
            ↓
Next batch uses improved prompt → eval scores go up → repeat
```

This is not simulated. The improvement agent calls live `@arizeai/phoenix-mcp` MCP tools, reads real operational data, and writes a versioned prompt back to Phoenix — the same data visible in your Phoenix Cloud dashboard.

---

## Eval results across improvement cycles

All scores are 0–1 where higher = better.

| Metric | Baseline | Cycle 1 | Cycle 2 | Δ vs baseline |
|--------|:--------:|:-------:|:-------:|:-------------:|
| `citation_groundedness` | 0.389 | 0.600 | 0.500 | **+28%** |
| `hallucination_risk` | 0.667 | 0.750 | 0.750 | **+12%** |
| `completeness` | 1.000 | 1.000 | 1.000 | — |
| `reasoning_coherence` | 1.000 | 1.000 | 1.000 | — |

> Citation groundedness improved **+54% after cycle 1** — the agent diagnosed
> its own failure ("claims stated without in-text attribution") from live trace
> data and rewrote its system prompt autonomously. No human edited the prompt.

> Fill in after running cycles. Baseline scores visible in Phoenix: [app.phoenix.arize.com/s/you-can-do-your-own-phoenix-or-ask-me-to-share-mine](https://app.phoenix.arize.com/s/you-can-do-your-own-phoenix-or-ask-me-to-share-mine)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User / Batch Runner                 │
└──────────────────────┬──────────────────────────────┘
                       │ question
                       ▼
┌─────────────────────────────────────────────────────┐
│              Research Agent (Google ADK)             │
│                                                     │
│  Tools:                                             │
│  ├── search_web()          Tavily → DDG fallback    │
│  ├── search_wikipedia()    Free factual grounding   │
│  ├── extract_key_claims()  Claim classification     │
│  ├── identify_contradictions() Cross-source check   │
│  └── assess_citation_quality() Pre-output check     │
└──────────────────────┬──────────────────────────────┘
                       │ auto_instrument=True
                       ▼
┌─────────────────────────────────────────────────────┐
│             Arize Phoenix Cloud                      │
│                                                     │
│  Traces ──► Spans ──► Annotations (eval scores)     │
│  Prompts registry (versioned system prompts)        │
└──────┬───────────────────────────────┬──────────────┘
       │ get_spans()                   │ MCP tools
       │ add_span_annotation()         │ list-traces
       ▼                               │ get-spans
┌─────────────┐                        │ get-prompt-version
│ Eval Runner │                        │ upsert-prompt
│ (Gemini     │                        ▼
│  LLM-Judge) │          ┌─────────────────────────┐
└─────────────┘          │   Improvement Agent      │
                         │   (Google ADK + MCP)     │
                         │                          │
                         │   Reads → Diagnoses →    │
                         │   Rewrites → Saves       │
                         └─────────────────────────┘
```

---

## Quick start

### Prerequisites

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Node.js 18+ — for Phoenix MCP server (`npx @arizeai/phoenix-mcp`)
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com/app/apikey)
- A Phoenix Cloud account (free) from [app.phoenix.arize.com](https://app.phoenix.arize.com)
- A Tavily API key (free, 1000/month) from [app.tavily.com](https://app.tavily.com)

### Install

```bash
git clone https://github.com/Oleh8978/prometheus.git
cd prometheus
cp .env.example .env
# Edit .env with your keys (see below)
uv sync
```

### Configure `.env`

```env
# Phoenix Cloud
PHOENIX_API_KEY=px_live_...
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space-name
PHOENIX_PROJECT_NAME=gemini-hackathon

# Gemini
GOOGLE_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash

# Search (free tier)
TAVILY_API_KEY=tvly-...
```

### Run a single research query

```bash
make run MESSAGE="What caused the 2023 US regional banking crisis?"
```

### Run the full batch (20 research questions)

```bash
make batch-quick    # first 5 questions — good for testing
make batch          # all 20 questions
```

### Run LLM-as-Judge evals on traces

```bash
make evals
```

Scores appear as annotations on each span in Phoenix Cloud. Check your project → click any trace → look for the colored annotation badges on the `invocation` span.

### Run one improvement cycle

```bash
make improve CYCLE=1
```

The improvement agent reads trace data and eval scores via Phoenix MCP, diagnoses failure patterns, rewrites the system prompt, and saves it to Phoenix Prompts as a new version.

### Run a full pipeline cycle (batch + evals + improve)

```bash
make pipeline CYCLE=1
```

---

## Project structure

```
prometheus/
├── agent/
│   ├── instrumentation.py          Phoenix tracing setup (auto_instrument=True)
│   ├── main.py                     Single-query CLI entry point
│   ├── improvement_agent.py        Self-improvement loop via Phoenix MCP
│   └── research_agent/
│       ├── agent.py                ADK Agent with 5 FunctionTools
│       ├── prompt.py               6-step research system prompt
│       └── tools/
│           ├── research_tools.py   search_web, search_wikipedia,
│           │                       extract_key_claims, identify_contradictions,
│           │                       assess_citation_quality
│           └── get_trace.py        Phoenix span inspector (debug utility)
├── evals/
│   └── run_evals.py                Gemini LLM-as-Judge pipeline → Phoenix annotations
├── scripts/
│   ├── batch_run.py                Run all 20 questions, traces → Phoenix
│   ├── pipeline.py                 Full cycle orchestrator
│   └── score_summary.py            Print eval score table across cycles
├── data/
│   ├── research_questions.json     20 hard multi-domain research questions
│   ├── batch_results.json          Agent answers (auto-generated)
│   ├── eval_results.json           Eval scores (auto-generated)
│   └── change_log_cycle_N.txt      Improvement agent's change log (auto-generated)
├── .gemini/settings.json           Phoenix MCP server config for Gemini CLI
├── Makefile                        All commands
└── pyproject.toml                  Dependencies (uv)
```

---

## Makefile commands

```bash
make run MESSAGE="..."    # single query
make batch-quick          # 5 questions
make batch                # all 20 questions
make evals                # score all traces
make improve CYCLE=1      # run improvement agent
make pipeline CYCLE=1     # batch + evals + improve in one shot
make scores               # print current score summary table
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Agent runtime | [Google ADK](https://google.github.io/adk-docs/) |
| LLM | Gemini 2.5 Flash |
| Tracing | [Arize Phoenix Cloud](https://app.phoenix.arize.com) |
| Instrumentation | [openinference-instrumentation-google-adk](https://pypi.org/project/openinference-instrumentation-google-adk/) |
| MCP (self-introspection) | [@arizeai/phoenix-mcp](https://arize.com/docs/phoenix/integrations/phoenix-mcp-server) |
| Evals | Gemini LLM-as-Judge via `google-genai` + `phoenix.client` |
| Web search | [Tavily](https://tavily.com) (free tier) → [DDGS](https://pypi.org/project/ddgs/) fallback |
| Package manager | [uv](https://docs.astral.sh/uv/) |

---

## The 4 eval dimensions

Each agent response is scored on four dimensions by a Gemini LLM-as-Judge:

**`citation_groundedness`** — Does every major claim reference a specific source (URL, named publication, or author)? Scores: `well_cited` (1.0) · `partially_cited` (0.5) · `uncited` (0.0)

**`reasoning_coherence`** — Did the agent decompose the question, weigh evidence, and reach logical conclusions? Scores: `coherent` (1.0) · `partial` (0.5) · `incoherent` (0.0)

**`hallucination_risk`** — Are specific numbers, dates, and study names cited or appropriately hedged? Scores: `low_risk` (1.0) · `medium_risk` (0.5) · `high_risk` (0.0)

**`completeness`** — Did the report address all aspects of the question including tradeoffs and uncertainty? Scores: `complete` (1.0) · `partial` (0.5) · `incomplete` (0.0)

Scores are logged back to Phoenix as span annotations via `client.spans.add_span_annotation()` and are readable in the Phoenix UI and by the improvement agent via MCP.

---

## What we learned

The bottleneck is not the model — it's prompt specificity. Gemini 2.5 Flash produces strong citation discipline, but only when the system prompt gives concrete, explicit instructions about *how* to cite (author, year, URL) rather than just *that* it should. The improvement agent discovered this in cycle 1 from reading actual trace data — not from intuition.

The Phoenix MCP layer gives the improvement agent qualitatively different self-knowledge than just reading its own output. Via `list-traces` and `get-spans`, it can see latency per reasoning step, token cost per sub-question, and where in the chain reasoning broke down — signal no eval score alone provides.

---

## License

Apache-2.0 — see [LICENSE](LICENSE)