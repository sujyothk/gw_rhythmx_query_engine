from __future__ import annotations

import re
from typing import Any, Dict, List

from .pipeline import EngineIndex
from .query_router import route
from .handlers import (
    handle_condition,
    handle_medications,
    handle_allergies,
    handle_avoid_due_to_allergies,
    handle_labs,
    handle_diabetes_complications,
    handle_encounters,
)
from .prompts import build_prompt
from .providers.hf_transformers import HFGenerator


def answer_query(
    idx: EngineIndex,
    query: str,
    top_k: int = 8,
    use_llm: bool = False,
    llm_provider: str = "hf",
    llm_model_path: str = "models/tinyllama",
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> Dict[str, Any]:
    rq = route(query)

    hits = idx.retrieval.search(query, top_k=top_k)
    retrieved = [{"source": c.source, "score": float(score), "text": c.text} for c, score in hits]

    facts: List[Dict[str, Any]] = []
    deterministic_answer = ""

    if rq.intent == "condition":
        out = handle_condition(idx, rq.entity)
        facts, deterministic_answer = out["facts"], out["answer"]
    elif rq.intent == "medication":
        out = handle_medications(idx, rq.entity)
        facts, deterministic_answer = out["facts"], out["answer"]
    elif rq.intent == "allergy":
        out = handle_allergies(idx)
        facts, deterministic_answer = out["facts"], out["answer"]
    elif rq.intent == "avoid_allergies":
        out = handle_avoid_due_to_allergies(idx)
        facts, deterministic_answer = out["facts"], out["answer"]
    elif rq.intent == "diabetes_complications":
        out = handle_diabetes_complications(idx)
        facts, deterministic_answer = out["facts"], out["answer"]
    elif rq.intent == "labs":
        out = handle_labs(idx, query)
        facts, deterministic_answer = out["facts"], out["answer"]
    elif rq.intent == "encounter":
        out = handle_encounters(idx)
        facts, deterministic_answer = out["facts"], out["answer"]
    else:
        facts, deterministic_answer = [], "Not found in provided records."

    det_citations = sorted({f["source"] for f in facts}) if facts else []

    used_llm = False
    final_answer = deterministic_answer

    if use_llm:
        if llm_provider != "hf":
            raise ValueError("Only llm_provider='hf' is supported in this build.")

        prompt = build_prompt(query, facts=facts, retrieved=retrieved)

        try:
            gen = HFGenerator(model_path=llm_model_path, temperature=temperature, max_tokens=max_tokens)
            llm_text = gen.generate(prompt)

            # Guardrails:
            # 1) Must include a Source: line with at least one FHIR resource id
            # 2) Must not introduce new/unknown FHIR ids (reduce hallucinated citations)
            if llm_text and "Source:" in llm_text:
                cited_set = set(re.findall(r"[A-Za-z]+/[A-Za-z0-9\-\.]+", llm_text))
                if cited_set:
                    if det_citations:
                        if cited_set.issubset(set(det_citations)):
                            final_answer = llm_text
                            used_llm = True
                    else:
                        # If we had no deterministic citations, accept the LLM output
                        final_answer = llm_text
                        used_llm = True
        except Exception:
            # Fall back to deterministic
            used_llm = False
            final_answer = deterministic_answer

    return {
        "question": query,
        "intent": rq.intent,
        "entity": rq.entity,
        "used_llm": used_llm,
        "answer": final_answer,
        "citations": det_citations,
        "retrieved": retrieved,
    }
