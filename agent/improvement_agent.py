"""
Self-improvement orchestrator.
Uses Phoenix MCP tools (list-traces, get-spans, get-prompt-version, upsert-prompt)
combined with the local eval results to rewrite the system prompt.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp.client.stdio import StdioServerParameters
from google.genai import types


# ── Helper: load eval summary for the agent's context ─────────────────────────

def load_eval_summary() -> str:
    """Load the latest eval results and format as a readable summary."""
    results_path = Path(__file__).resolve().parents[1] / "data" / "eval_results.json"
    if not results_path.exists():
        return "No eval results found. Run evals/run_evals.py first."

    results = json.loads(results_path.read_text())
    if not results:
        return "Eval results file is empty."

    import pandas as pd
    df = pd.DataFrame(results)

    summary_lines = ["=== Eval Score Summary ==="]
    for name, group in df.groupby("name"):
        avg = group["score"].mean()
        low = group[group["score"] < 0.5]
        summary_lines.append(f"\n{name}: avg={avg:.2f} ({len(low)} spans below 0.5)")
        # Show 2 worst cases with explanations
        for _, row in low.head(2).iterrows():
            expl = str(row.get("explanation", ""))[:200]
            summary_lines.append(f"  - span {row['span_id'][:8]}... label={row.get('label')} | {expl}")

    summary = "\n".join(summary_lines)
    if len(summary) > 2000:
        summary = summary[:2000].rstrip() + "\n...[summary truncated to avoid token limits]"
    return summary


# ── Phoenix MCP toolset ────────────────────────────────────────────────────────

def make_phoenix_mcp() -> McpToolset:
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=[
                    "@arizeai/phoenix-mcp@latest",
                    "--baseUrl", os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
                    "--apiKey", os.environ["PHOENIX_API_KEY"],
                ],
            ),
            timeout=30.0,
        ),
        tool_filter=[
            "list-traces",
            "get-spans",
            "get-prompt-version",
            "upsert-prompt",
        ],
    )


# ── Improvement agent system prompt ───────────────────────────────────────────

IMPROVEMENT_INSTRUCTION = """
You are a prompt engineer specializing in improving AI research agents.
You have access to Phoenix MCP tools that let you inspect the agent's operational data.

Your job in this session:

1. Call list-traces to get the 10 most recent traces from the research agent project.
2. Call get-spans on 3-5 traces that had the most complex questions.
3. Call get-prompt-version to read the current system prompt (name: "research-system-prompt").
4. Read the eval summary provided in your context — it shows where the agent scores lowest.
5. Identify 2-3 SPECIFIC, CONCRETE failure patterns. Be precise:
   - Bad: "the agent needs to cite better"
   - Good: "the agent cites URLs but not author/year, making claims unverifiable"
6. Rewrite the system prompt to fix those specific patterns. Keep what works.
7. Call upsert-prompt to save the new version with name "research-system-prompt".
8. Write a short "change log" explaining: what you changed, why, and what you expect to improve.

Rules:
- Only change what is genuinely broken. Don't rewrite for the sake of it.
- Keep the 6-step research workflow structure — it's the core of the agent.
- Make the new prompt more specific, not longer. Vague additions don't help.
- The change log must be honest — judges will read it.
"""


# ── Run the improvement agent ──────────────────────────────────────────────────

async def run_improvement_cycle(cycle_number: int = 1) -> str:
    """Run one improvement cycle. Returns the agent's change log."""
    from instrumentation import setup_tracing
    setup_tracing()

    eval_summary = load_eval_summary()

    # Inject eval summary into the first user message
    user_message = f"""
Run an improvement cycle for the research agent.

Here is the current eval performance summary:
{eval_summary}

Follow your instructions: inspect traces, read the current prompt,
identify failure patterns, rewrite the prompt, save it, and write a change log.
Label the new prompt version as: cycle-{cycle_number}
"""

    phoenix_tools = make_phoenix_mcp()

    improvement_agent = Agent(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        name="improvement_agent",
        instruction=IMPROVEMENT_INSTRUCTION,
        tools=[phoenix_tools],
    )

    app_name = "improvement_agent"
    user_id = "orchestrator"
    session_id = f"cycle_{cycle_number}_{secrets.token_hex(4)}"

    runner = InMemoryRunner(agent=improvement_agent, app_name=app_name)
    await runner.session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    change_log = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=user_message)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        change_log += part.text
    finally:
        try:
            await phoenix_tools.close()
        except Exception as e:
            print(f"Warning: error closing Phoenix MCP toolset: {e}")

    print(f"\n── Improvement Cycle {cycle_number} Change Log ──")
    print(change_log)

    # Save the change log
    log_path = Path(__file__).resolve().parents[1] / "data" / f"change_log_cycle_{cycle_number}.txt"
    log_path.write_text(change_log)
    print(f"\nChange log saved to {log_path}")

    return change_log


if __name__ == "__main__":
    import sys
    cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    asyncio.run(run_improvement_cycle(cycle))