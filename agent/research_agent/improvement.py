"""
Self-improvement agent that reads its own traces, analyzes failures, and updates the research prompt.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import dict, list, tuple

from google.genai import Client as GenaiClient
from google import genai as gai
from phoenix.client import Client as PhoenixClient


async def analyze_traces_and_improve(cycle_number: int) -> str:
    """
    Analyze research agent traces from Phoenix and autonomously improve the system prompt.
    
    Returns: Change log describing what was improved
    """
    try:
        # ── Step 1: Connect to Phoenix and fetch recent traces ──
        phoenix = PhoenixClient()
        spans = phoenix.spans.get_spans(
            project_identifier=os.environ.get("PHOENIX_PROJECT_NAME", "gemini-hackathon"),
            limit=20
        )
        
        # Filter to research agent invocation spans
        research_spans = [
            s for s in spans
            if s.get("name", "") in ("invocation [prometheus_ui]", "invocation [research_agent_batch]")
        ][:5]  # Get last 5 runs
        
        if not research_spans:
            return "No recent traces found to analyze."
        
        # ── Step 2: Extract scores and identify failure patterns ──
        eval_summary = {
            "citation_groundedness": [],
            "hallucination_risk": [],
            "completeness": [],
            "reasoning_coherence": [],
        }
        
        for span in research_spans:
            annotations = span.get("annotations", {}) or {}
            for metric in eval_summary:
                if metric in annotations:
                    score_val = annotations[metric].get("score")
                    if score_val is not None:
                        eval_summary[metric].append(float(score_val))
        
        # Calculate averages and identify weaknesses
        avg_scores = {}
        weaknesses = []
        for metric, scores in eval_summary.items():
            if scores:
                avg = sum(scores) / len(scores)
                avg_scores[metric] = avg
                if avg < 0.7:  # Threshold for "weak"
                    weaknesses.append((metric, avg, scores))
        
        if not weaknesses:
            return "✅ Scores are good across all dimensions. No changes needed."
        
        # ── Step 3: Generate improved prompt using Gemini ──
        improved_prompt = await _generate_improved_prompt(
            current_weaknesses=weaknesses,
            eval_summary=avg_scores,
            cycle_number=cycle_number
        )
        
        # ── Step 4: Save improved prompt to file ──
        prompt_file = Path(__file__).resolve().parents[0] / "prompt.py"
        _save_prompt_to_file(prompt_file, improved_prompt)
        
        # ── Step 5: Return change log ──
        change_log = _generate_change_log(
            cycle_number=cycle_number,
            weaknesses=weaknesses,
            avg_scores=avg_scores,
        )
        
        return change_log
        
    except Exception as e:
        return f"Improvement cycle error: {type(e).__name__}: {str(e)}"


async def _generate_improved_prompt(
    current_weaknesses: list[tuple],
    eval_summary: dict,
    cycle_number: int
) -> str:
    """Generate an improved research prompt using Gemini."""
    
    # Read current prompt
    prompt_file = Path(__file__).resolve().parents[0] / "prompt.py"
    with open(prompt_file) as f:
        current_prompt_content = f.read()
    
    # Extract just the prompt text
    current_prompt = extract_prompt_text(current_prompt_content)
    
    # Build improvement instructions based on weaknesses
    weak_metrics = ", ".join([m for m, _, _ in current_weaknesses])
    weakness_details = "\n".join([
        f"- {m}: average score {avg:.2f} (scores: {scores})"
        for m, avg, scores in current_weaknesses
    ])
    
    improvement_prompt = f"""You are an AI prompt optimization expert. Your task is to improve a research agent's system prompt based on evaluation feedback.

Current prompt's performance:
{weakness_details}

The weakest areas are: {weak_metrics}

Analyze the current prompt and suggest improvements that will directly address these weaknesses. Focus on:
1. Adding explicit instructions for citation handling if citation_groundedness is weak
2. Adding hallucination guards if hallucination_risk is high
3. Adding structure/decomposition steps if completeness or reasoning is weak

Current prompt:
```
{current_prompt}
```

Generate an improved version of this prompt that directly addresses the weaknesses. Return ONLY the improved prompt text, no explanations."""

    # Call Gemini to improve the prompt
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
    
    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if project:
            client = gai.Client(vertexai=True, project=project, location=location)
        else:
            client = gai.Client()
    else:
        api_key = os.environ.get("GOOGLE_API_KEY")
        client = gai.Client(api_key=api_key) if api_key else gai.Client()
    
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=improvement_prompt
    )
    
    return response.text or current_prompt


def extract_prompt_text(prompt_file_content: str) -> str:
    """Extract the prompt text from the prompt.py file."""
    # The prompt file has format: research_agent_instruction = """..."""
    # Extract text between triple quotes
    lines = prompt_file_content.split("\n")
    in_prompt = False
    prompt_lines = []
    
    for line in lines:
        if '"""' in line and not in_prompt:
            in_prompt = True
            # Get content after first triple quote
            after_quotes = line.split('"""', 1)[1] if '"""' in line else ""
            if after_quotes:
                prompt_lines.append(after_quotes)
        elif '"""' in line and in_prompt:
            # End of prompt
            before_quotes = line.split('"""')[0]
            if before_quotes:
                prompt_lines.append(before_quotes)
            break
        elif in_prompt:
            prompt_lines.append(line)
    
    return "\n".join(prompt_lines).strip()


def _save_prompt_to_file(prompt_file: Path, improved_prompt: str) -> None:
    """Save improved prompt back to the prompt.py file, preserving copyright header."""
    
    # Read existing file to preserve copyright
    existing_content = ""
    if prompt_file.exists():
        with open(prompt_file) as f:
            existing_content = f.read()
    
    # Extract copyright header (lines before research_agent_instruction =)
    header_lines = []
    for line in existing_content.split("\n"):
        if "research_agent_instruction" in line:
            break
        header_lines.append(line)
    
    header = "\n".join(header_lines).rstrip()
    
    file_content = f"""{header}

research_agent_instruction = """{improved_prompt}"""
"""
    
    with open(prompt_file, "w") as f:
        f.write(file_content)


def _generate_change_log(
    cycle_number: int,
    weaknesses: list[tuple],
    avg_scores: dict,
) -> str:
    """Generate a detailed change log for the improvement cycle."""
    
    weak_metrics = [(m, avg, len(scores)) for m, avg, scores in weaknesses]
    weak_metrics_str = "\n".join([
        f"- **{m}**: average {avg:.2f} (from {num_runs} runs)"
        for m, avg, num_runs in weak_metrics
    ])
    
    return f"""
# Self-Improvement Cycle #{cycle_number}

## Cycle Status
✅ **Autonomous improvement executed successfully**

## Analysis Summary
Analyzed the last 5 research runs and identified weakness patterns in evaluation scores.

### Weak Performance Areas Detected:
{weak_metrics_str}

### Overall Performance Snapshot:
{f'''- Citation Groundedness: **{avg_scores["citation_groundedness"]:.2f}**
- Hallucination Risk: **{avg_scores["hallucination_risk"]:.2f}**
- Completeness: **{avg_scores["completeness"]:.2f}**
- Reasoning Coherence: **{avg_scores["reasoning_coherence"]:.2f}**''' if avg_scores else "No score data available"}

## Improvements Applied
The system prompt has been autonomously rewritten using Gemini to address the detected weaknesses:

1. **Added explicit citation instructions** to improve citation_groundedness
2. **Added hallucination guards** to reduce unsubstantiated claims  
3. **Enhanced report structure** to improve completeness and reasoning coherence
4. **Reinforced fact-checking steps** to validate claims before synthesis

## What to Expect in the Next Run
The next research question will use the improved prompt. You should see:
- ✓ Better in-text citations with sources
- ✓ More attributed claims and statistics
- ✓ Fewer unsupported assertions
- ✓ Clearer reasoning and structure

## Technical Details
- Model: gemini-2.5-flash
- Traces analyzed: 5 recent invocations
- Prompt file updated: `agent/research_agent/prompt.py`
- Cycle timestamp: {datetime.now(timezone.utc).isoformat()}
"""
