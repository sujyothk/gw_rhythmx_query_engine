from __future__ import annotations

from typing import Any, Dict, List


SYSTEM_INSTRUCTIONS = """You are a careful clinical assistant.
Follow these rules strictly:
1) Use ONLY the provided FACTS and CONTEXT. Do not invent details.
2) If the information is missing, respond exactly: Not found in provided records.
3) Output MUST follow this format:
   - Short clinical statements that include relevant medical codes when available (ICD-10, SNOMED, LOINC, RxNorm, etc.)
   - Final line must be: Source: <comma-separated FHIR ResourceType/id list>
4) Do not include any sources that are not present in FACTS/CONTEXT.
5) Be concise. Prefer 1–8 lines of facts.
"""


def build_prompt(question: str, facts: List[Dict[str, Any]], retrieved: List[Dict[str, Any]]) -> str:
    # Facts are already formatted as clinical statements with codes.
    facts_block = "\n".join([f"- {f['text']} (Source: {f['source']})" for f in facts]) if facts else "(none)"

    # Context remains as evidence; sources are embedded in the chunk header.
    ctx_block = "\n\n".join([f"[Source: {r['source']}]\n{r['text']}" for r in retrieved]) if retrieved else "(none)"

    return f"""{SYSTEM_INSTRUCTIONS}

QUESTION:
{question}

FACTS (preferred; use these first):
{facts_block}

CONTEXT (supporting evidence):
{ctx_block}

Write the final answer using only the facts/context. Ensure the last line is:
Source: <comma-separated list of FHIR ResourceType/id>
"""
