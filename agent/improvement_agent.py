"""
Improvement agent coordinator - reads traces and triggers self-improvement loop.
"""

import asyncio
from agent.research_agent.improvement import analyze_traces_and_improve


async def run_improvement_cycle(cycle_number: int) -> str:
    """
    Main entry point for the improvement cycle.
    
    Args:
        cycle_number: The run number that triggered the improvement (should be a multiple of IMPROVEMENT_THRESHOLD)
    
    Returns:
        Change log describing what was improved
    """
    try:
        # Run the improvement analysis and prompt update
        change_log = await analyze_traces_and_improve(cycle_number)
        return change_log
    except Exception as e:
        return f"Improvement cycle error: {type(e).__name__}: {str(e)}"
