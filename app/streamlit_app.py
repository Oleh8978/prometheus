import streamlit as st
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from google.adk.runners import InMemoryRunner
from google.genai import types
from agent.research_agent.agent import root_agent

st.set_page_config(page_title="Prometheus", page_icon="🔥")
st.title("Prometheus")
st.caption("A self-improving research agent powered by Gemini + Arize Phoenix")

question = st.text_area(
    "Research question",
    placeholder="What are the real tradeoffs between vector databases and pgvector for production RAG in 2025?",
    height=100
)

if st.button("Research", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Researching... (this takes 30-60 seconds)"):
            async def run(q):
                import secrets
                runner = InMemoryRunner(
                    agent=root_agent,
                    app_name="prometheus_ui"
                )
                session_id = f"ui_{secrets.token_hex(4)}"
                await runner.session_service.create_session(
                    app_name="prometheus_ui",
                    user_id="web_user",
                    session_id=session_id
                )
                answer = ""
                async for event in runner.run_async(
                    user_id="web_user",
                    session_id=session_id,
                    new_message=types.Content(
                        role="user",
                        parts=[types.Part(text=q)]
                    ),
                ):
                    if event.is_final_response() and event.content:
                        for part in event.content.parts:
                            if part.text:
                                answer += part.text
                return answer

            result = asyncio.run(run(question))
            st.markdown(result)
            st.divider()
            st.caption("Every run is traced to Arize Phoenix · Eval scores logged automatically")