"""Research tools for the multi-step synthesis agent."""
from __future__ import annotations
import json
from typing import Optional


def extract_key_claims(source_text: str, topic: str) -> str:
    """
    Extract factual claims from a block of source text related to a topic.
    Returns a structured list of claims with confidence indicators.

    Args:
        source_text: Raw text from a search result or document.
        topic: The sub-question or topic these claims should relate to.

    Returns:
        JSON string with list of {claim, confidence, needs_verification}.
    """
    # This is a structured prompt tool — ADK will call the LLM with this docstring
    # The LLM fills in the logic. Keep the docstring detailed so ADK understands the contract.
    return json.dumps({
        "error": "Call this via ADK — the LLM executes the extraction logic."
    })

def identify_contradictions(claims_a: str, claims_b: str) -> str:
    """
    Compare two sets of claims (as JSON strings from extract_key_claims)
    and identify factual contradictions or disagreements between them.

    Args:
        claims_a: JSON string of claims from source A.
        claims_b: JSON string of claims from source B.

    Returns:
        JSON string with list of {topic, claim_a, claim_b, severity}.
    """
    return json.dumps({"contradictions": []})


def assess_citation_quality(report_text: str, sources_used: str) -> str:
    """
    Review a draft report and assess whether each major claim is backed
    by a specific cited source. Flag unsupported claims.

    Args:
        report_text: The draft report text.
        sources_used: Comma-separated list of source URLs or names used.

    Returns:
        JSON string with {coverage_score, unsupported_claims, recommendation}.
    """
    return json.dumps({"coverage_score": 0.0, "unsupported_claims": [], "recommendation": ""})