# Copyright 2026 Oleh8978
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool, google_search
from dotenv import load_dotenv

from instrumentation import setup_tracing
from research_agent.prompt import research_agent_instruction

from research_agent.tools.research_tools import (
    search_web,
    search_wikipedia,
    extract_key_claims,
    identify_contradictions,
    assess_citation_quality,
)

# Ensure ADK CLI runs (`adk run shopping_demo`) load local env and tracing.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
setup_tracing()

_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

root_agent = Agent(
    model=_model,
    name="research_agent",
    instruction=research_agent_instruction,
        tools=[
        FunctionTool(func=search_web),              # Tavily → DDG fallback
        FunctionTool(func=search_wikipedia),        # factual grounding
        FunctionTool(func=extract_key_claims),      # pull structured claims from text
        FunctionTool(func=identify_contradictions), # spot conflicting sources
        FunctionTool(func=assess_citation_quality), # self-check citations before output
    ],
)
