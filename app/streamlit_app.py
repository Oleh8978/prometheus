import streamlit as st
import asyncio
import sys
import os
import json
import time
import secrets
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_env = ROOT / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)

from google.adk.runners import InMemoryRunner
from google.genai import types
from google import genai as gai
from phoenix.client import Client

from agent.research_agent.agent import root_agent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prometheus — Self-Improving Research Agent",
    page_icon="🔥",
    layout="wide",
)

# ── Sample questions judges can click ─────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What are the real tradeoffs between vector databases and PostgreSQL pgvector for production RAG in 2025?",
    "How do Anthropic and OpenAI differ in their approaches to AI safety?",
    "What caused the 2023 US regional banking crisis and what regulatory changes resulted?",
    "Compare GLP-1 drugs (Ozempic, Wegovy) vs traditional bariatric surgery on 5-year outcomes.",
    "What are the main technical bottlenecks preventing autonomous vehicles from reaching SAE Level 4?",
    "How does the EU AI Act classify AI systems by risk, and what does it mean for LLM startups?",
]

IMPROVEMENT_THRESHOLD = 5  # trigger improvement after this many questions

# ── Session state ──────────────────────────────────────────────────────────────
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "score_history" not in st.session_state:
    st.session_state.score_history = []
if "improvement_ran" not in st.session_state:
    st.session_state.improvement_ran = False
if "last_improvement_run" not in st.session_state:
    st.session_state.last_improvement_run = 0
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""


# ── Helper: run the research agent ────────────────────────────────────────────
def run_agent(question: str) -> str:
    async def _run():
        runner = InMemoryRunner(agent=root_agent, app_name="prometheus_ui")
        session_id = f"ui_{secrets.token_hex(4)}"
        await runner.session_service.create_session(
            app_name="prometheus_ui", user_id="web_user", session_id=session_id
        )
        answer = ""
        async for event in runner.run_async(
            user_id="web_user",
            session_id=session_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=question)]
            ),
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        answer += part.text
        return answer
    return asyncio.run(_run())


# ── Helper: score a response with LLM-as-Judge ────────────────────────────────
def score_response(question: str, answer: str) -> dict:
    """Score directly from question+answer — no Phoenix re-fetch needed."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    ai = gai.Client(api_key=api_key) if api_key else gai.Client()

    EVALS = [
        ("citation_groundedness",
         f"Does this research report cite specific sources (URLs, author names, publications) for its major claims?\n"
         f"Q: {question[:600]}\nA: {answer[:2000]}\n"
         f"Label exactly one: well_cited / partially_cited / uncited\nFormat — Label: <choice>",
         {"well_cited": 1.0, "partially_cited": 0.5, "uncited": 0.0}),

        ("hallucination_risk",
         f"Does this report make confident specific claims (numbers, study names, dates) WITHOUT citing a source?\n"
         f"A: {answer[:2000]}\n"
         f"Label exactly one: low_risk / medium_risk / high_risk\nFormat — Label: <choice>",
         {"low_risk": 1.0, "medium_risk": 0.5, "high_risk": 0.0}),

        ("completeness",
         f"Does this report fully address all parts of the question, including tradeoffs and uncertainty?\n"
         f"Q: {question[:600]}\nA: {answer[:2000]}\n"
         f"Label exactly one: complete / partial / incomplete\nFormat — Label: <choice>",
         {"complete": 1.0, "partial": 0.5, "incomplete": 0.0}),

        ("reasoning_coherence",
         f"Does this report break the question into parts, weigh evidence, and reach logical conclusions?\n"
         f"A: {answer[:2000]}\n"
         f"Label exactly one: coherent / partial / incoherent\nFormat — Label: <choice>",
         {"coherent": 1.0, "partial": 0.5, "incoherent": 0.0}),
    ]

    scores = {}
    labels_out = {}
    for name, prompt, label_map in EVALS:
        try:
            resp = ai.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            text = resp.text.strip().lower()
            label = next((l for l in label_map if l in text), list(label_map.keys())[1])
            scores[name] = label_map[label]
            labels_out[name] = label
        except Exception:
            scores[name] = None
            labels_out[name] = "error"

    return scores, labels_out


# ── Helper: log scores to Phoenix ─────────────────────────────────────────────
def log_to_phoenix(scores: dict, labels: dict, question: str):
    """Find the most recent prometheus_ui span and log annotations to it."""
    try:
        time.sleep(2)  # wait for Phoenix to commit the trace
        phoenix = Client()
        spans = phoenix.spans.get_spans(
            project_identifier=os.environ.get("PHOENIX_PROJECT_NAME", "gemini-hackathon")
        )
        # Get the most recent invocation span
        ui_spans = [
            s for s in spans
            if s.get("name", "") in ("invocation [prometheus_ui]", "invocation [research_agent_batch]")
        ]
        if not ui_spans:
            return

        # Sort by start_time descending — pick newest
        ui_spans.sort(
            key=lambda s: s.get("start_time", ""),
            reverse=True
        )
        span_id = ui_spans[0]["context"]["span_id"]

        for name, score in scores.items():
            if score is not None:
                phoenix.spans.add_span_annotation(
                    span_id=span_id,
                    annotation_name=name,
                    score=float(score),
                    label=str(labels.get(name, "")),
                    explanation="Auto-scored by Prometheus UI",
                    annotator_kind="LLM",
                )
    except Exception:
        pass  # annotation logging is best-effort


# ── Helper: run improvement cycle ─────────────────────────────────────────────
def run_improvement():
    try:
        from agent.improvement_agent import run_improvement_cycle

        async def _improve():
            return await run_improvement_cycle(
                cycle_number=st.session_state.run_count
            )
        return asyncio.run(_improve())
    except Exception as e:
        return f"Improvement cycle error: {e}"


# ── UI Layout ──────────────────────────────────────────────────────────────────
col_main, col_side = st.columns([2, 1])

with col_main:
    st.title("🔥 Prometheus")
    st.markdown(
        "**A self-improving research agent.** Ask a complex question — "
        "Prometheus decomposes it, searches multiple sources, identifies contradictions, "
        "and synthesizes a cited report. Every response is scored by an LLM-as-Judge. "
        "After **5 questions**, the improvement agent reads its own traces via "
        "**Arize Phoenix MCP** and autonomously rewrites its system prompt."
    )

    # Progress bar toward improvement trigger
    progress = min(st.session_state.run_count / IMPROVEMENT_THRESHOLD, 1.0)
    st.progress(progress,
        text=f"Questions answered: {st.session_state.run_count} / {IMPROVEMENT_THRESHOLD} "
             f"{'— improvement cycle will trigger after next run ✓' if st.session_state.run_count >= IMPROVEMENT_THRESHOLD else f'— improvement triggers at {IMPROVEMENT_THRESHOLD}'}"
    )

    st.divider()

    # Question input
    question = st.text_area(
        "Research question",
        value=st.session_state.selected_question,
        placeholder="What are the real tradeoffs between vector databases and pgvector for production RAG in 2025?",
        height=100,
        key="question_input",
    )

    run_btn = st.button("🔍 Research", type="primary", use_container_width=True)

    if run_btn:
        if not question.strip():
            st.warning("Enter a question first — or click one of the examples on the right.")
        else:
            # ── Step 1: Run the agent ──────────────────────────────────────
            with st.spinner("Researching... (30–90 seconds — agent is searching, extracting claims, and synthesizing)"):
                start = time.time()
                result = run_agent(question)
                elapsed = round(time.time() - start, 1)

            st.success(f"Done in {elapsed}s")
            st.markdown(result)
            st.divider()

            # ── Step 2: Score the response ────────────────────────────────
            with st.spinner("Scoring with LLM-as-Judge (4 dimensions)..."):
                scores, labels = score_response(question, result)
                log_to_phoenix(scores, labels, question)

            # Show score metrics
            st.caption("**Quality scores (LLM-as-Judge) — higher is always better**")
            c1, c2, c3, c4 = st.columns(4)
            score_display = {
                "Citations":     ("citation_groundedness",  "well_cited=1.0 · partially=0.5 · uncited=0.0"),
                "Hallucination": ("hallucination_risk",      "low_risk=1.0 · medium=0.5 · high_risk=0.0"),
                "Completeness":  ("completeness",            "complete=1.0 · partial=0.5 · incomplete=0.0"),
                "Reasoning":     ("reasoning_coherence",     "coherent=1.0 · partial=0.5 · incoherent=0.0"),
            }
            for col, (display_name, (key, help_text)) in zip(
                [c1, c2, c3, c4], score_display.items()
            ):
                v = scores.get(key)
                lbl = labels.get(key, "")
                col.metric(
                    display_name,
                    f"{v:.2f}" if v is not None else "—",
                    delta=lbl,
                    help=help_text,
                )

            # Store in history
            st.session_state.run_count += 1
            st.session_state.score_history.append({
                "run": st.session_state.run_count,
                "question": question[:60] + "...",
                "scores": scores,
            })

            st.caption(
                f"Run #{st.session_state.run_count} · "
                f"Traced to [Arize Phoenix]"
                f"Scores logged as span annotations"
            )

            # ── Step 3: Auto-trigger improvement after threshold ──────────
            should_improve = (
                st.session_state.run_count >= IMPROVEMENT_THRESHOLD
                and st.session_state.run_count % IMPROVEMENT_THRESHOLD == 0
                and st.session_state.last_improvement_run != st.session_state.run_count
            )
            if should_improve:
                st.divider()
                st.info(
                    f"🔄 **{IMPROVEMENT_THRESHOLD} questions answered.** "
                    "Triggering the self-improvement loop — "
                    "the improvement agent is reading its own traces via Phoenix MCP..."
                )
                with st.spinner(
                    "Improvement agent: reading traces → diagnosing failures → rewriting prompt → saving to Phoenix Prompts..."
                ):
                    change_log = run_improvement()

                if isinstance(change_log, str) and change_log.startswith("Improvement cycle error:"):
                    st.error(change_log)
                else:
                    st.session_state.improvement_ran = True
                    st.session_state.last_improvement_run = st.session_state.run_count
                    st.success("✅ Improvement cycle complete — system prompt updated in Phoenix Prompts registry.")
                    with st.expander("📋 See what the improvement agent changed and why"):
                        st.markdown(change_log)

                    st.info(
                        "Ask another question to see the improved responses. "
                        "The agent now uses the updated prompt."
                    )

with col_side:
    st.markdown("### Try these questions")
    st.caption("Click any to auto-fill the research box")

    for i, q in enumerate(SAMPLE_QUESTIONS):
        if st.button(q[:65] + "…", key=f"sample_{i}", use_container_width=True):
            st.session_state.selected_question = q
            st.rerun()

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
**1. Ask** a complex, multi-part question

**2. Research** — agent decomposes → searches (Tavily + Wikipedia) → extracts claims → identifies contradictions → synthesizes report

**3. Score** — Gemini LLM-as-Judge scores the response on 4 dimensions automatically

**4. Trace** — every span logged to Arize Phoenix (latency, tokens, tool calls)

**5. Improve** — after **5 questions**, a second agent reads its own traces via Phoenix MCP, diagnoses failure patterns, and rewrites the system prompt autonomously
    """)

    st.divider()
    st.markdown("### Score history")
    if st.session_state.score_history:
        for entry in reversed(st.session_state.score_history[-5:]):
            with st.expander(f"Run #{entry['run']} — {entry['question']}", expanded=False):
                s = entry["scores"]
                cols = st.columns(2)
                cols[0].metric("Citations", f"{s.get('citation_groundedness', 0):.2f}" if s.get('citation_groundedness') is not None else "—")
                cols[0].metric("Hallucination", f"{s.get('hallucination_risk', 0):.2f}" if s.get('hallucination_risk') is not None else "—")
                cols[1].metric("Completeness", f"{s.get('completeness', 0):.2f}" if s.get('completeness') is not None else "—")
                cols[1].metric("Reasoning", f"{s.get('reasoning_coherence', 0):.2f}" if s.get('reasoning_coherence') is not None else "—")
    else:
        st.caption("No runs yet — ask your first question.")

    st.divider()
    st.markdown("### Links")
    st.markdown("""
- [💻 GitHub](https://github.com/Oleh8978/prometheus)
    """)