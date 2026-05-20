"""
Run LLM-as-Judge evals on research agent traces.

Strategy: get_traces_dataframe() pulls spans from Phoenix (unchanged — confirmed
working). Evaluation is done by calling Gemini directly via google-genai, because
create_evaluator(kind="llm") in the installed arize-phoenix-evals does not accept
a `prompt` or `output_config` kwarg, making async_evaluate_dataframe produce no
useful scores. Scores are logged back as Phoenix span annotations and saved to
data/eval_results.json for the improvement agent.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from google import genai  # package: google-genai  (Pylance "unknown" warning is a false positive)
from phoenix.client import Client


# ── helpers ────────────────────────────────────────────────────────────────────

def _gemini_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env")
    return genai.Client(api_key=api_key)


def _parse_label(text: str, valid_labels: list[str]) -> str | None:
    """Return the first valid label found in the model's response (case-insensitive)."""
    lower = text.lower()
    for label in valid_labels:
        if label.lower() in lower:
            return label
    return None


def _parse_explanation(text: str) -> str:
    """Return the first sentence that looks like an explanation."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in lines:
        if len(line) > 20:
            return line[:400]
    return text.strip()[:400]


# ── 1. Pull traces from Phoenix ────────────────────────────────────────────────

def get_traces_dataframe(project: str, limit: int = 100) -> pd.DataFrame:
    """
    Pull agent spans from Phoenix using get_spans().
    Parses JSON-encoded input.value and output.value correctly.
    """
    client = Client()

    spans = client.spans.get_spans(
        project_identifier=project,
    )

    if not spans:
        print("No spans returned from Phoenix.")
        return pd.DataFrame()

    names = set(s.get("name", "") for s in spans)
    print(f"Span names found in project: {names}")

    rows = []
    for span in spans:
        name = span.get("name", "")
        if not name.startswith("invocation"):
            continue

        attrs = span.get("attributes", {})
        raw_input  = attrs.get("input.value", "")
        raw_output = attrs.get("output.value", "")

        if not raw_input or not raw_output:
            continue

        try:
            inp_parsed = json.loads(raw_input)
            question = (
                inp_parsed
                .get("new_message", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        except (json.JSONDecodeError, IndexError, KeyError):
            question = raw_input

        try:
            out_parsed = json.loads(raw_output)
            answer = (
                out_parsed
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        except (json.JSONDecodeError, IndexError, KeyError):
            answer = raw_output

        if not question or not answer:
            continue

        rows.append({
            "span_id":  span.get("context", {}).get("span_id", ""),
            "question": question.strip(),
            "answer":   answer.strip(),
        })

    df = pd.DataFrame(rows)
    print(f"Found {len(df)} evaluable spans in project '{project}'")
    if not df.empty:
        print(f"Sample question: {df['question'].iloc[0][:100]}")
        print(f"Sample answer:   {df['answer'].iloc[0][:100]}")
    return df


# ── 2. Evaluator definitions ───────────────────────────────────────────────────

EVALUATORS = [
    {
        "name": "citation_groundedness",
        "prompt": """\
You are evaluating a research report produced by an AI agent.

Question the agent was asked:
{input}

Agent's report:
{output}

Assess: Does the report provide specific citations or named sources for its major factual claims?

Respond with exactly ONE label from this list, then a one-sentence explanation:
- well_cited      (score 1.0): most major claims have a named source, URL, or reference
- partially_cited (score 0.5): some claims have sources, others don't
- uncited         (score 0.0): major claims are made without any citation or source

Format:
Label: <label>
Explanation: <one sentence>""",
        "labels": {"well_cited": 1.0, "partially_cited": 0.5, "uncited": 0.0},
    },
    {
        "name": "reasoning_coherence",
        "prompt": """\
You are evaluating the reasoning quality of an AI research agent.

Question: {input}
Report: {output}

Does the report show clear multi-step reasoning?
- Did it decompose the question into parts?
- Did it weigh evidence before concluding?
- Are conclusions logically derived from the evidence?

Respond with exactly ONE label, then a one-sentence explanation:
- coherent   (score 1.0): clear decomposition, evidence-based conclusions, logical flow
- partial    (score 0.5): some structure, but reasoning gaps or jumps present
- incoherent (score 0.0): claims without reasoning, no decomposition, or logical inconsistencies

Format:
Label: <label>
Explanation: <one sentence>""",
        "labels": {"coherent": 1.0, "partial": 0.5, "incoherent": 0.0},
    },
    {
        "name": "hallucination_risk",
        "prompt": """\
You are checking an AI research report for hallucination risk.

Report: {output}

Look for:
- Specific statistics or numbers stated as fact without a source
- Named studies, papers, or reports that seem invented
- Confident claims about very recent events that may not be verifiable
- Precise figures (percentages, dates, dollar amounts) stated without citation

Respond with exactly ONE label, then give the riskiest specific example:
- low_risk    (score 1.0): specific claims are cited or appropriately hedged
- medium_risk (score 0.5): some uncited specifics but report is mostly hedged
- high_risk   (score 0.0): multiple confident specific claims with no source

Format:
Label: <label>
Explanation: <riskiest example or "none found">""",
        "labels": {"low_risk": 1.0, "medium_risk": 0.5, "high_risk": 0.0},
    },
    {
        "name": "completeness",
        "prompt": """\
You are assessing whether an AI research report fully addresses the question asked.

Question: {input}
Report: {output}

Did the report address all major aspects of the question?
- Consider: did it cover all parts of a multi-part question?
- Did it address tradeoffs if asked for comparison?
- Did it acknowledge what is unknown or uncertain?

Respond with exactly ONE label, then list any missing aspects:
- complete   (score 1.0): all major aspects addressed
- partial    (score 0.5): most aspects covered, one or two missing
- incomplete (score 0.0): significant aspects of the question ignored

Format:
Label: <label>
Explanation: <missing aspects or "none">""",
        "labels": {"complete": 1.0, "partial": 0.5, "incomplete": 0.0},
    },
]


# ── 3. Async eval engine ───────────────────────────────────────────────────────

async def _evaluate_one(
    client: genai.Client,
    evaluator: dict,
    row: pd.Series,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call Gemini for a single (evaluator, span) pair and return a score dict."""
    prompt = evaluator["prompt"].format(
        input=row["question"],
        output=row["answer"],
    )

    async with semaphore:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                ),
            )
            text = response.text.strip()
        except Exception as e:
            print(f"  [warn] Gemini call failed for {evaluator['name']}/{row['span_id'][:8]}: {e}")
            text = ""

    # Parse label
    label = _parse_label(text, list(evaluator["labels"].keys()))
    if label is None:
        keys = list(evaluator["labels"].keys())
        label = keys[len(keys) // 2]
        print(f"  [warn] Could not parse label — defaulting to '{label}'")
        print(f"         Raw response: {text[:200]}")

    score = evaluator["labels"][label]

    explanation = ""
    if "Explanation:" in text:
        explanation = text.split("Explanation:", 1)[1].strip()[:400]
    else:
        explanation = _parse_explanation(text)

    return {
        "span_id":     row["span_id"],
        "name":        evaluator["name"],
        "label":       label,
        "score":       score,
        "explanation": explanation,
    }


async def _run_all_evals(df: pd.DataFrame, concurrency: int = 4) -> pd.DataFrame:
    """Run all evaluators over all rows concurrently and return scores DataFrame."""
    ai = _gemini_client()
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        _evaluate_one(ai, evaluator, row, semaphore)
        for _, row in df.iterrows()
        for evaluator in EVALUATORS
    ]

    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


# ── 4. Orchestrator ────────────────────────────────────────────────────────────

async def run_evals(project: str | None = None) -> pd.DataFrame | None:
    project = project or os.environ.get("PHOENIX_PROJECT_NAME", "gemini-hackathon")
    phoenix = Client()

    df = get_traces_dataframe(project)
    if df.empty:
        print("No spans to evaluate. Run the batch runner first.")
        return None

    total = len(df) * len(EVALUATORS)
    print(f"Running {len(EVALUATORS)} evaluators × {len(df)} spans = {total} Gemini calls...")

    scores_df = await _run_all_evals(df, concurrency=4)

    print("\n── Eval Results ──────────────────────────")
    summary = scores_df.groupby("name")["score"].mean().round(3)
    print(summary.to_string())
    print()

    logged = 0
    for _, row in scores_df.iterrows():
        try:
            phoenix.spans.add_span_annotation(
                span_id=row["span_id"],
                annotation_name=row["name"],
                score=float(row["score"]),
                label=str(row["label"]),
                explanation=str(row["explanation"]),
                annotator_kind="LLM",
            )
            logged += 1
        except Exception as e:
            print(f"  Warning: failed to log annotation for {row['span_id']}: {e}")

    print(f"Logged {logged}/{len(scores_df)} annotations to Phoenix.")

    out_path = Path(__file__).resolve().parents[1] / "data" / "eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(scores_df.to_json(orient="records", indent=2))
    print(f"Results saved to {out_path}")

    return scores_df


if __name__ == "__main__":
    asyncio.run(run_evals())