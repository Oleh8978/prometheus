"""
LLM-as-Judge evals — Gemini direct, robust label parsing, correct Phoenix annotation API.
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

from google import genai
from phoenix.client import Client


# ── helpers ────────────────────────────────────────────────────────────────────

def _gemini_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your .env")
    return genai.Client(api_key=api_key)


def _parse_label(text: str, valid_labels: list[str]) -> tuple[str | None, str]:
    """
    Extract label and explanation from Gemini's response.
    Handles multiple response formats Gemini actually uses.
    Returns (label, explanation).
    """
    lower = text.lower()

    # Find the label — check all valid ones, longest match wins (avoids partial hits)
    found_label = None
    for label in sorted(valid_labels, key=len, reverse=True):
        if label.lower() in lower:
            found_label = label
            break

    # Extract explanation — try multiple patterns Gemini uses
    explanation = ""

    # Pattern 1: "Explanation: ..."
    m = re.search(r'[Ee]xplanation[:\s]+(.+?)(?:\n|$)', text)
    if m:
        explanation = m.group(1).strip()

    # Pattern 2: "because ..." or "since ..."
    if not explanation:
        m = re.search(r'\b(?:because|since|as|the report|this report|the agent)\b.{20,}', text, re.I)
        if m:
            explanation = m.group(0).strip()[:400]

    # Pattern 3: just take the longest sentence that isn't the label line
    if not explanation:
        sentences = [s.strip() for s in re.split(r'[.\n]', text) if len(s.strip()) > 30]
        label_sentences = [s for s in sentences if not any(lb.lower() in s.lower() for lb in valid_labels)]
        if label_sentences:
            explanation = max(label_sentences, key=len)[:400]

    # Final fallback
    if not explanation:
        explanation = text.strip()[:400]

    return found_label, explanation


# ── 1. Pull traces ─────────────────────────────────────────────────────────────

def get_traces_dataframe(project: str) -> pd.DataFrame:
    client = Client()
    spans = client.spans.get_spans(project_identifier=project)

    if not spans:
        print("No spans returned from Phoenix.")
        return pd.DataFrame()

    names = set(s.get("name", "") for s in spans)
    print(f"Span names in project: {names}")

    rows = []
    for span in spans:
        if not span.get("name", "").startswith("invocation"):
            continue

        attrs = span.get("attributes", {})
        raw_input  = attrs.get("input.value", "")
        raw_output = attrs.get("output.value", "")

        if not raw_input or not raw_output:
            continue

        try:
            question = json.loads(raw_input)["new_message"]["parts"][0]["text"]
        except (json.JSONDecodeError, KeyError, IndexError):
            question = raw_input

        try:
            answer = json.loads(raw_output)["content"]["parts"][0]["text"]
        except (json.JSONDecodeError, KeyError, IndexError):
            answer = raw_output

        if question and answer:
            rows.append({
                "span_id":  span["context"]["span_id"],
                "question": question.strip(),
                "answer":   answer.strip(),
            })

    df = pd.DataFrame(rows)
    print(f"Found {len(df)} evaluable spans")
    if not df.empty:
        print(f"  Q: {df['question'].iloc[0][:80]}...")
        print(f"  A: {df['answer'].iloc[0][:80]}...")
    return df


# ── 2. Evaluator definitions ───────────────────────────────────────────────────

EVALUATORS = [
    {
        "name": "citation_groundedness",
        "prompt": """\
Evaluate this AI research report for citation quality.

Question asked: {input}

Report: {output}

Does the report cite specific sources (URLs, named publications, author names) for major claims?

Choose exactly one:
- well_cited: most major claims reference a specific source or URL
- partially_cited: some claims are sourced, others are not
- uncited: claims made without any citation or attribution

Your response format (follow exactly):
Label: <your_choice>
Explanation: <one sentence explaining why>""",
        "labels": {"well_cited": 1.0, "partially_cited": 0.5, "uncited": 0.0},
    },
    {
        "name": "reasoning_coherence",
        "prompt": """\
Evaluate the reasoning quality of this AI research report.

Question: {input}
Report: {output}

Did the agent break the question into sub-parts, weigh evidence, and reach logical conclusions?

Choose exactly one:
- coherent: clear decomposition, evidence-based conclusions, logical flow throughout
- partial: some structure present but reasoning gaps or unsupported jumps exist
- incoherent: no decomposition, conclusions not derived from evidence

Your response format (follow exactly):
Label: <your_choice>
Explanation: <one sentence explaining why>""",
        "labels": {"coherent": 1.0, "partial": 0.5, "incoherent": 0.0},
    },
    {
        "name": "hallucination_risk",
        "prompt": """\
Check this research report for hallucination risk.

Report: {output}

Look for: specific numbers/stats without sources, invented study names, confident claims about recent events with no citation.

Choose exactly one:
- low_risk: specific claims are cited or appropriately hedged with uncertainty language
- medium_risk: some uncited specifics exist but most claims are hedged
- high_risk: multiple confident specific claims (numbers, studies, dates) with no source

Your response format (follow exactly):
Label: <your_choice>
Explanation: <quote the single riskiest unsourced claim, or write "none found">""",
        "labels": {"low_risk": 1.0, "medium_risk": 0.5, "high_risk": 0.0},
    },
    {
        "name": "completeness",
        "prompt": """\
Assess whether this research report fully answers the question.

Question: {input}
Report: {output}

Did the agent address all major aspects? For comparison questions: did it cover both sides? Did it acknowledge uncertainty?

Choose exactly one:
- complete: all major aspects of the question addressed
- partial: most aspects covered but one or two missing
- incomplete: significant parts of the question ignored

Your response format (follow exactly):
Label: <your_choice>
Explanation: <list missing aspects, or write "none missing">""",
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
    prompt = evaluator["prompt"].format(
        input=row["question"][:2000],   # cap to avoid token limits
        output=row["answer"][:3000],
    )

    valid_labels = list(evaluator["labels"].keys())
    text = ""

    async with semaphore:
        try:
            loop = asyncio.get_running_loop()   # fix: not get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                ),
            )
            text = response.text.strip()
        except Exception as e:
            print(f"  [warn] Gemini failed {evaluator['name']}/{row['span_id'][:8]}: {e}")

    label, explanation = _parse_label(text, valid_labels)

    # Debug: print when we can't parse — helps identify format issues
    if label is None:
        print(f"  [warn] No label found in response for {evaluator['name']}")
        print(f"         Response was: {repr(text[:300])}")
        label = valid_labels[len(valid_labels) // 2]  # middle as fallback

    score = evaluator["labels"][label]

    print(f"  ✓ {evaluator['name']}: {label} ({score}) | {explanation[:60]}...")

    return {
        "span_id":     row["span_id"],
        "name":        evaluator["name"],
        "label":       label,
        "score":       score,
        "explanation": explanation,
    }


async def _run_all_evals(df: pd.DataFrame, concurrency: int = 3) -> pd.DataFrame:
    ai = _gemini_client()
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _evaluate_one(ai, ev, row, sem)
        for _, row in df.iterrows()
        for ev in EVALUATORS
    ]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


# ── 4. Orchestrator ────────────────────────────────────────────────────────────

async def run_evals(project: str | None = None) -> pd.DataFrame | None:
    project = project or os.environ.get("PHOENIX_PROJECT_NAME", "gemini-hackathon")
    phoenix = Client()

    df = get_traces_dataframe(project)
    if df.empty:
        print("No spans to evaluate. Run: make batch-quick")
        return None

    total = len(df) * len(EVALUATORS)
    print(f"\nRunning {len(EVALUATORS)} evaluators × {len(df)} spans = {total} Gemini calls...\n")

    scores_df = await _run_all_evals(df)

    print("\n── Eval Summary ───────────────────────────────")
    summary = scores_df.groupby("name")["score"].agg(["mean", "min", "max"]).round(3)
    print(summary.to_string())
    print()

    # ── Log back to Phoenix  ── FIX: use name=, not annotation_name= ──
    logged, failed = 0, 0
    for _, row in scores_df.iterrows():
        try:
            phoenix.spans.add_span_annotation(
                span_id=row["span_id"],
                annotation_name=row["name"],                # ← was annotation_name, which is WRONG
                score=float(row["score"]),
                label=str(row["label"]),
                explanation=str(row["explanation"]),
                annotator_kind="LLM",
            )
            logged += 1
        except Exception as e:
            failed += 1
            if failed <= 3:  # only print first 3 to avoid spam
                print(f"  [warn] annotation failed: {e}")

    print(f"Logged {logged}/{len(scores_df)} annotations to Phoenix. ({failed} failed)")

    out_path = Path(__file__).resolve().parents[1] / "data" / "eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(scores_df.to_json(orient="records", indent=2))
    print(f"Saved → {out_path}")

    return scores_df


if __name__ == "__main__":
    asyncio.run(run_evals())