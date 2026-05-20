"""Full pipeline: batch run → evals → improve → repeat."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from scripts.batch_run import main as batch_run
from evals.run_evals import run_evals
from agent.improvement_agent import run_improvement_cycle


async def run_cycle(cycle: int, quick: bool = False):
    print(f"\n{'='*60}")
    print(f" CYCLE {cycle}")
    print(f"{'='*60}")

    print(f"\n[1/3] Running batch queries...")
    if quick:
        sys.argv.append("--quick")
    await batch_run()

    print(f"\n[2/3] Running evals...")
    scores = await run_evals()
    if scores is not None:
        summary = scores.groupby("name")["score"].mean().round(3)
        print(f"\nCycle {cycle} scores:\n{summary}")

    print(f"\n[3/3] Running improvement agent...")
    await run_improvement_cycle(cycle_number=cycle)

    print(f"\n✓ Cycle {cycle} complete.")


if __name__ == "__main__":
    cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    quick = "--quick" in sys.argv
    asyncio.run(run_cycle(cycle, quick))
