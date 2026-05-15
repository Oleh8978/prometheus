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

"""Phoenix tracing: ``register(..., auto_instrument=True)`` per ADK doc.

https://arize.com/docs/phoenix/integrations/python/google-adk/google-adk-tracing

Requires ``google-adk>=1.32`` and ``openinference-instrumentation-google-adk>=0.1.11``.

Environment: ``PHOENIX_API_KEY``, ``PHOENIX_COLLECTOR_ENDPOINT``, optional ``PHOENIX_PROJECT_NAME``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from phoenix.otel import register

_provider: Optional[Any] = None


def setup_tracing() -> Optional[Any]:
    """Returns the tracer provider when Phoenix auth is configured, else ``None``."""
    global _provider
    if _provider is not None:
        return _provider
    if not (os.environ.get("PHOENIX_API_KEY") or "").strip():
        return None
    _provider = register(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "gemini-hackathon"),
        batch=False,
        auto_instrument=True,
        verbose=False,
    )
    return _provider
