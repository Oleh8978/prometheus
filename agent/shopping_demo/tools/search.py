# Copyright 2025 Google LLC
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

from google.adk.tools import ToolContext

from shopping_demo.mini_webshop import get_webshop_env


async def search(keywords: str, tool_context: ToolContext) -> str:
    """Search for keywords in the webshop.

    Args:
      keywords: The keywords to search for.
      tool_context: ADK tool context (artifacts optional).

    Returns:
      Text observation of the search results page.
    """
    webshop_env = get_webshop_env()
    action_string = f"search[{keywords}]"
    webshop_env.server.assigned_instruction_text = f"Find me {keywords}."
    webshop_env.step(action_string)

    ob = webshop_env.observation
    idx = ob.find("Back to Search")
    if idx >= 0:
        ob = ob[idx:]

    # Optional artifact (skipped if unsupported in minimal runs)
    try:
        from google.genai import types

        if webshop_env.state.get("html"):
            await tool_context.save_artifact(
                "html",
                types.Part.from_uri(
                    file_uri=webshop_env.state["html"], mime_type="text/html"
                ),
            )
    except (ValueError, ImportError, TypeError):
        pass

    return ob
