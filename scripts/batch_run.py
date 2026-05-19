"""Run all 20 research questions and send traces to Phoenix."""
from __future__ import annotations

import asyncio
import json
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from google.adk.runners import InMemoryRunner
from google.genai import types

from agent.instrumentation import setup_tracing
from agent.research_agent.agent import root_agent


async def run_question(runner: InMemoryRunner, question: str, qid: str) -> str:
    """Run one research question, return the agent's response text."""
    app_name = "research_agent_batch"
    user_id = "batch_runner"
    session_id = f"{qid}_{secrets.token_hex(4)}"

    await runner.session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=question)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_text += part.text

    return response_text


async def main():
    setup_tracing()

    questions_path = Path(__file__).resolve().parents[1] / "data" / "research_questions.json"
    questions = json.loads(questions_path.read_text())

    # Optionally run only a subset for testing: questions[:5]
    subset = questions[:5] if "--quick" in sys.argv else questions

    runner = InMemoryRunner(agent=root_agent, app_name="research_agent_batch")

    results = []
    for i, q in enumerate(subset):
        print(f"\n[{i+1}/{len(subset)}] {q['id']}: {q['question'][:80]}...")
        try:
            answer = await run_question(runner, q["question"], q["id"])
            results.append({"id": q["id"], "question": q["question"], "answer": answer, "status": "ok"})
            print(f"  ✓ {len(answer)} chars")
        except Exception as e:
            results.append({"id": q["id"], "question": q["question"], "answer": "", "status": f"error: {e}"})
            print(f"  ✗ {e}")

    out_path = Path(__file__).resolve().parents[1] / "data" / "batch_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDone. Results saved to {out_path}")
    print("Open Phoenix Cloud to see all traces.")


if __name__ == "__main__":
    asyncio.run(main())