import asyncio
from phoenix.client import Client

def should_run_improvement(project: str, threshold: int = 10) -> bool:
    """Trigger improvement after every 10 new unscored traces."""
    client = Client()
    spans = client.spans.get_spans(project_identifier=project)
    unscored = [
        s for s in spans
        if s.get("name", "").startswith("invocation")
        and not s.get("annotations")  # no eval scores yet
    ]
    return len(unscored) >= threshold

async def main():
    if should_run_improvement("gemini-hackathon"):
        print("Running eval + improvement cycle...")
        # run evals
        from evals.run_evals import run_evals
        await run_evals()
        # run improvement
        from agent.improvement_agent import run_improvement_cycle
        await run_improvement_cycle()
    else:
        print("Not enough new traces yet.")

asyncio.run(main())