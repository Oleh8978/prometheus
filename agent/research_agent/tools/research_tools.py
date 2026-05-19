# Copyright 2026 Oleh8978 — Apache-2.0
"""
Research tools — all free, no paid APIs required.
Priority: Tavily (best quality) → DuckDuckGo (fallback) → Wikipedia (factual grounding).
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Optional


# ── 1. Web Search — Tavily primary, DuckDuckGo fallback ───────────────────────

def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web for a query and return structured results with full content.
    Use this for each research sub-question — one specific query per call.
    Returns titles, URLs, and content snippets from the most relevant pages.

    Args:
        query: A specific, focused search query. One sub-question at a time.
        num_results: Number of results to return (default 5, max 8).

    Returns:
        JSON string with {query, source, results: [{title, url, content, score}]}.
    """
    num_results = min(int(num_results), 8)

    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        result = _search_tavily(query, num_results, tavily_key)
        if result:
            return result

    # Fallback to DuckDuckGo if Tavily fails or key missing
    return _search_duckduckgo(query, num_results)


def _search_tavily(query: str, num: int, api_key: str) -> Optional[str]:
    """Tavily — returns full content, not just snippets. Best for AI agents."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            max_results=num,
            search_depth="advanced",  # deeper content extraction, still free
            include_answer=True,       # Tavily also gives an AI summary
            include_raw_content=False, # raw HTML would blow token budget
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", "")[:600],  # cap to save tokens
                "score":   round(r.get("score", 0.0), 3),
            })

        # Tavily's own AI answer is useful context for the agent
        tavily_answer = response.get("answer", "")

        return json.dumps({
            "query":          query,
            "source":         "tavily",
            "tavily_summary": tavily_answer[:400] if tavily_answer else None,
            "results":        results,
        })

    except Exception as e:
        # Don't crash — return None so fallback triggers
        print(f"[search_web] Tavily failed: {e}")
        return None


def _search_duckduckgo(query: str, num: int) -> str:
    """DuckDuckGo — no API key, always free, good fallback."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(
                query,
                max_results=num,
                safesearch="moderate",
            )):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "content": r.get("body", "")[:500],
                    "score":   None,
                })
        return json.dumps({
            "query":   query,
            "source":  "duckduckgo",
            "results": results,
        })
    except Exception as e:
        return json.dumps({
            "query":   query,
            "source":  "duckduckgo",
            "error":   str(e),
            "results": [],
        })


# ── 2. Wikipedia Lookup — free factual grounding ───────────────────────────────

def search_wikipedia(topic: str, sentences: int = 8) -> str:
    """
    Look up a topic on Wikipedia and return a factual summary.
    Use this to ground definitions, historical facts, or well-established concepts
    where you need a reliable, citable, non-hallucinated source.
    Wikipedia is NOT suitable for recent events (past 6 months).

    Args:
        topic: The specific concept, person, event, or technology to look up.
        sentences: Number of summary sentences to return (default 8).

    Returns:
        JSON string with {topic, summary, url, found, note}.
    """
    try:
        # Wikipedia REST API — no key, completely free
        encoded = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Prometheus-ResearchAgent/1.0 (hackathon project)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        if data.get("type") == "disambiguation":
            return json.dumps({
                "topic":  topic,
                "found":  False,
                "note":   "Disambiguation page — try a more specific topic name.",
                "url":    data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            })

        extract = data.get("extract", "")
        # Trim to requested sentence count
        all_sentences = re.split(r'(?<=[.!?])\s+', extract)
        trimmed = " ".join(all_sentences[:sentences])

        return json.dumps({
            "topic":       data.get("title", topic),
            "found":       True,
            "summary":     trimmed,
            "url":         data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "description": data.get("description", ""),
            "note":        f"Wikipedia content. Cite as: Wikipedia, '{data.get('title', topic)}', accessed 2025.",
        })

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return json.dumps({
                "topic": topic,
                "found": False,
                "note":  f"No Wikipedia article found for '{topic}'. Try alternate spelling.",
            })
        return json.dumps({"topic": topic, "found": False, "error": str(e)})
    except Exception as e:
        return json.dumps({"topic": topic, "found": False, "error": str(e)})


# ── 3. Extract Key Claims ──────────────────────────────────────────────────────

def extract_key_claims(source_text: str, topic: str) -> str:
    """
    Extract and classify factual claims from source text on a given topic.
    Identifies statistics, dated facts, hedged assertions, and strong claims.
    Flags which claims most need verification before including in a report.

    Args:
        source_text: Raw text from a search result (pass the 'content' field).
        topic: The sub-question or research angle this text addresses.

    Returns:
        JSON string with {topic, claims: [{text, type, needs_verification}],
        claim_count}.
    """
    if not source_text or not source_text.strip():
        return json.dumps({"topic": topic, "claims": [], "claim_count": 0,
                           "note": "Empty input — pass a non-empty content string."})

    sentences = re.split(r'(?<=[.!?])\s+', source_text.strip())

    stat_pat   = re.compile(r'\d+\.?\d*\s*(%|percent|million|billion|trillion|x\b|ms\b|GB|TB|seconds)', re.I)
    date_pat   = re.compile(r'\b(202[0-9]|January|February|March|April|May|June|July|August|September|October|November|December)\b', re.I)
    hedged_pat = re.compile(r'\b(may|might|could|suggests|indicates|reportedly|appears to|estimated|projected|expected)\b', re.I)
    strong_pat = re.compile(r'\b(is|are|was|were|has|have|shows|proves|demonstrated|confirmed|established)\b', re.I)
    name_pat   = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')  # proper nouns

    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) < 25:
            continue

        if stat_pat.search(s):
            claims.append({"text": s[:280], "type": "statistic",   "needs_verification": True})
        elif date_pat.search(s) and strong_pat.search(s):
            claims.append({"text": s[:280], "type": "dated_fact",  "needs_verification": True})
        elif hedged_pat.search(s):
            claims.append({"text": s[:280], "type": "hedged",      "needs_verification": False})
        elif strong_pat.search(s) and (name_pat.search(s) or len(s) > 80):
            claims.append({"text": s[:280], "type": "assertion",   "needs_verification": True})

    return json.dumps({
        "topic":       topic,
        "claim_count": len(claims),
        "claims":      claims[:12],
    })


# ── 4. Identify Contradictions ─────────────────────────────────────────────────

def identify_contradictions(claims_a: str, claims_b: str) -> str:
    """
    Compare two sets of claims from different sources and identify contradictions.
    Pass JSON strings from extract_key_claims for two different sources.
    Use after gathering evidence from multiple sources on the same sub-question.

    Args:
        claims_a: JSON string from extract_key_claims for source A.
        claims_b: JSON string from extract_key_claims for source B.

    Returns:
        JSON string with {contradictions: [{topic, claim_a, claim_b, severity}],
        contradiction_count, summary}.
    """
    try:
        data_a = json.loads(claims_a) if isinstance(claims_a, str) else claims_a
        data_b = json.loads(claims_b) if isinstance(claims_b, str) else claims_b
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input", "contradictions": []})

    list_a = [c["text"] for c in data_a.get("claims", [])]
    list_b = [c["text"] for c in data_b.get("claims", [])]

    if not list_a or not list_b:
        return json.dumps({
            "contradiction_count": 0,
            "contradictions": [],
            "summary": "No claims to compare — ensure both inputs are from extract_key_claims.",
        })

    contradictions = []

    # Check numeric disagreements (>25% difference = flag it)
    def extract_numbers(texts):
        nums = {}
        for text in texts:
            for m in re.finditer(r'(\d+\.?\d*)\s*(%|percent|ms|GB|million|billion|x\b)', text, re.I):
                unit = m.group(2).lower().strip()
                nums.setdefault(unit, []).append((float(m.group(1)), text[:150]))
        return nums

    nums_a = extract_numbers(list_a)
    nums_b = extract_numbers(list_b)

    for unit in set(nums_a) & set(nums_b):
        vals_a = [v for v, _ in nums_a[unit]]
        vals_b = [v for v, _ in nums_b[unit]]
        if vals_a and vals_b:
            a, b = max(vals_a), max(vals_b)
            if a > 0 and abs(a - b) / a > 0.25:
                severity = "high" if abs(a - b) / a > 0.5 else "medium"
                contradictions.append({
                    "topic":    f"Numeric disagreement ({unit}): {a} vs {b}",
                    "claim_a":  nums_a[unit][0][1],
                    "claim_b":  nums_b[unit][0][1],
                    "severity": severity,
                })

    # Check direct negations on shared topics
    negations = re.compile(r'\b(not|never|no |cannot|impossible|incorrect|false|wrong)\b', re.I)
    for ca in list_a[:6]:
        for cb in list_b[:6]:
            words_a = set(ca.lower().split())
            words_b = set(cb.lower().split())
            stopwords = {"the","a","an","is","in","of","to","and","or","for","on","at","by","with"}
            shared = (words_a & words_b) - stopwords
            if len(shared) >= 4:
                a_neg = bool(negations.search(ca))
                b_neg = bool(negations.search(cb))
                if a_neg != b_neg:  # one negates, one doesn't
                    contradictions.append({
                        "topic":    f"Possible contradiction on: {', '.join(list(shared)[:4])}",
                        "claim_a":  ca[:150],
                        "claim_b":  cb[:150],
                        "severity": "medium",
                    })
                    break

    topic_a = data_a.get("topic", "source A")
    topic_b = data_b.get("topic", "source B")

    return json.dumps({
        "topic_a":             topic_a,
        "topic_b":             topic_b,
        "contradiction_count": len(contradictions),
        "contradictions":      contradictions[:5],
        "claims_compared":     f"{len(list_a)} vs {len(list_b)}",
        "summary": (
            f"Found {len(contradictions)} potential contradiction(s) between "
            f"'{topic_a}' and '{topic_b}'. "
            "Resolve high-severity ones before including claims in your report."
            if contradictions else
            f"No numeric or direct contradictions detected between {len(list_a)} "
            f"and {len(list_b)} claims. Sources appear consistent on this topic."
        ),
    })


# ── 5. Assess Citation Quality ─────────────────────────────────────────────────

def assess_citation_quality(report_text: str, sources_used: str) -> str:
    """
    Review a draft report section and check citation coverage before finalizing.
    Call this on your draft before writing the final response to the user.
    Identifies which claims lack a source and gives a concrete recommendation.

    Args:
        report_text: The draft text to review.
        sources_used: Comma-separated list of source URLs or names actually used.

    Returns:
        JSON string with {coverage_score, cited_count, uncited_count,
        unsupported_claims, recommendation}.
    """
    if not report_text.strip():
        return json.dumps({"error": "Empty report text.", "coverage_score": 0.0})

    sources = [s.strip() for s in sources_used.split(",") if s.strip()]
    sentences = re.split(r'(?<=[.!?])\s+', report_text.strip())

    factual = [
        s for s in sentences
        if len(s) > 40 and re.search(
            r'\b(is|are|was|were|shows|found|reported|according|has|have|'
            r'demonstrates|indicates|suggests|estimated|percent|million)\b', s, re.I
        )
    ]

    url_pat     = re.compile(r'https?://\S+')
    bracket_pat = re.compile(r'\[[\w\s,\.]+\]|\(\d{4}\)|\([\w]+,\s*\d{4}\)')
    source_pat  = re.compile(r'\b(according to|source:|via |from |per |cited in)\b', re.I)

    cited, uncited = [], []
    for s in factual:
        has_cite = (
            url_pat.search(s)
            or bracket_pat.search(s)
            or source_pat.search(s)
            or any(src.lower()[:20] in s.lower() for src in sources if len(src) > 5)
        )
        (cited if has_cite else uncited).append(s)

    total = len(factual)
    score = round(len(cited) / total, 2) if total > 0 else 0.0

    if score >= 0.8:
        rec = "Good citation coverage. Verify all URLs are valid before finalizing."
    elif score >= 0.5:
        rec = (
            f"{len(uncited)} claim(s) lack citations. Add 'according to [source]' "
            "or a URL for each before finalizing."
        )
    else:
        rec = (
            "Weak citation coverage — most claims are unsourced. "
            "Revise by attributing each major claim to a specific source or URL."
        )

    return json.dumps({
        "coverage_score":    score,
        "total_claims":      total,
        "cited_count":       len(cited),
        "uncited_count":     len(uncited),
        "unsupported_claims": uncited[:3],
        "has_url_citations": bool(url_pat.search(report_text)),
        "recommendation":    rec,
    })