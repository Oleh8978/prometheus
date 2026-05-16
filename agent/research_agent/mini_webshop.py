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

"""Tiny in-memory webshop for the hackathon starter (no PyTorch / Pyserini / downloads).

The real google/adk-samples personalized-shopping agent uses a large indexed catalog.
This module preserves the same *tool contract* (search[...] / click[...] steps) so the
ADK agent prompt and tool wiring stay familiar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_BUTTON_LINE = "Buttons you can click: {names}"


@dataclass
class Product:
    asin: str
    title: str
    price: str
    description: str
    features: str
    reviews: str
    colors: List[str]
    sizes: List[str]
    color: str = "default"
    size: str = "M"


_CATALOG: List[Product] = [
    Product(
        asin="B09P5CRVQ6",
        title="Floral Summer Dress flowy midi",
        price="$48",
        description="Lightweight floral print dress with adjustable straps.",
        features="Breathable fabric; machine washable; lined bodice.",
        reviews="4.6/5 — customers love the fit for warm weather.",
        colors=["Blue Floral", "Sage"],
        sizes=["S", "M", "L"],
    ),
    Product(
        asin="B0DEMO0001",
        title="Classic Denim Jacket",
        price="$59",
        description="Medium wash denim jacket with chest pockets.",
        features="Stretch denim; metal buttons; unisex cut.",
        reviews="4.4/5 — runs true to size.",
        colors=["Indigo", "Light Wash"],
        sizes=["S", "M", "L", "XL"],
    ),
    Product(
        asin="B0DEMO0002",
        title="Linen Blend Trousers",
        price="$42",
        description="Tapered linen-blend pants for office or weekend.",
        features="Wrinkle resistant blend; elastic waist.",
        reviews="4.3/5 — great for travel.",
        colors=["Khaki", "Black"],
        sizes=["30", "32", "34", "36"],
    ),
]


def _visible_buttons(names: List[str]) -> str:
    return _BUTTON_LINE.format(names=", ".join(names))


class MiniWebshopEnv:
    """Minimal text observation environment compatible with gym-style step()."""

    def __init__(self) -> None:
        self._mode: str = "search_page"
        self._query: str = ""
        self._results: List[Product] = []
        self._selected: Optional[Product] = None
        self._subpage: Optional[str] = None
        self.observation: str = ""
        self.state: dict = {"html": ""}
        self.server = _ServerStub()
        self.reset()

    def reset(self) -> str:
        self._mode = "search_page"
        self._query = ""
        self._results = []
        self._selected = None
        self._subpage = None
        self._render()
        return self.observation

    def _render(self) -> None:
        if self._mode == "search_page":
            self.observation = (
                "WebShop [Search]\n"
                "Instruction: "
                + getattr(self.server, "assigned_instruction_text", "Find a product.")
                + "\n"
                + _visible_buttons(["Search"])
            )
        elif self._mode == "results":
            lines = [
                f"WebShop [Results for '{self._query}']",
                "Instruction: Pick a product to inspect.",
            ]
            for p in self._results:
                lines.append(f"- [{p.asin}] {p.title} — {p.price}")
            buttons = ["Back to Search"] + [p.asin for p in self._results]
            lines.append(_visible_buttons(buttons))
            self.observation = "\n".join(lines)
        elif self._mode == "product" and self._selected:
            p = self._selected
            if self._subpage == "description":
                self.observation = (
                    f"WebShop [Description — {p.title}]\n{p.description}\n"
                    + _visible_buttons(["< Prev"])
                )
            elif self._subpage == "features":
                self.observation = (
                    f"WebShop [Features — {p.title}]\n{p.features}\n"
                    + _visible_buttons(["< Prev"])
                )
            elif self._subpage == "reviews":
                self.observation = (
                    f"WebShop [Reviews — {p.title}]\n{p.reviews}\n"
                    + _visible_buttons(["< Prev"])
                )
            else:
                opts: List[str] = [
                    "Description",
                    "Features",
                    "Reviews",
                    "Back to Search",
                    "< Prev",
                ]
                for c in p.colors:
                    opts.append(f"color[{c}]")
                for s in p.sizes:
                    opts.append(f"size[{s}]")
                opts.append("Buy Now")
                self.observation = (
                    f"WebShop [Product {p.asin}]\n{p.title} — {p.price}\n"
                    f"Selected color: {p.color} | size: {p.size}\n"
                    + _visible_buttons(opts)
                )
        elif self._mode == "done":
            self.observation = (
                "WebShop [Ordered]\nThanks — your order is being processed.\n"
                + _visible_buttons(["Back to Search"])
            )
        else:
            self.observation = "WebShop\n" + _visible_buttons(["Back to Search"])

    def step(self, action_string: str) -> tuple[None, Optional[float], bool, dict]:
        done = False
        reward: Optional[float] = None
        action_string = action_string.strip()

        m = re.match(r"^search\[(.*)\]$", action_string)
        if m:
            q = m.group(1).strip().lower()
            self._query = q
            self._results = [
                p
                for p in _CATALOG
                if q in p.title.lower()
                or any(q in w for w in p.title.lower().split())
                or q in p.asin.lower()
            ]
            if not self._results:
                self._results = list(_CATALOG)
            self._mode = "results"
            self._selected = None
            self._subpage = None
            self._render()
            return None, reward, done, {}

        m = re.match(r"^click\[(.*)\]$", action_string)
        if not m:
            self.observation = f"Unknown action: {action_string}\n" + _visible_buttons(
                ["Back to Search"]
            )
            return None, reward, done, {}

        btn = m.group(1).strip()

        if btn == "Back to Search":
            self.reset()
            return None, reward, done, {}

        if self._mode == "search_page":
            self._render()
            return None, reward, done, {}

        if self._mode == "results":
            for p in self._results:
                if btn == p.asin:
                    self._selected = Product(
                        asin=p.asin,
                        title=p.title,
                        price=p.price,
                        description=p.description,
                        features=p.features,
                        reviews=p.reviews,
                        colors=list(p.colors),
                        sizes=list(p.sizes),
                        color=p.color,
                        size=p.size,
                    )
                    self._mode = "product"
                    self._subpage = None
                    self._render()
                    return None, reward, done, {}
            self._render()
            return None, reward, done, {}

        if self._mode == "product" and self._selected:
            p = self._selected
            if btn == "< Prev":
                self._subpage = None
                self._render()
                return None, reward, done, {}
            if btn == "Description":
                self._subpage = "description"
                self._render()
                return None, reward, done, {}
            if btn == "Features":
                self._subpage = "features"
                self._render()
                return None, reward, done, {}
            if btn == "Reviews":
                self._subpage = "reviews"
                self._render()
                return None, reward, done, {}
            if btn.startswith("color[") and btn.endswith("]"):
                color = btn[6:-1]
                if color in p.colors:
                    p.color = color
                self._subpage = None
                self._render()
                return None, reward, done, {}
            if btn.startswith("size[") and btn.endswith("]"):
                size = btn[5:-1]
                if size in p.sizes:
                    p.size = size
                self._subpage = None
                self._render()
                return None, reward, done, {}
            if btn == "Buy Now":
                self._mode = "done"
                reward = 1.0
                done = True
                self._render()
                return None, reward, done, {}

        self._render()
        return None, reward, done, {}


@dataclass
class _ServerStub:
    assigned_instruction_text: str = "Find a product."


class _Registry:
    _env: Optional[MiniWebshopEnv] = None


def get_webshop_env() -> MiniWebshopEnv:
    if _Registry._env is None:
        _Registry._env = MiniWebshopEnv()
        _Registry._env.reset()
    return _Registry._env
