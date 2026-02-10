from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dateutil import parser as dtparser

from .pipeline import EngineIndex


# --- Coding helpers -----------------------------------------------------------

def _coding_label(system: str) -> Optional[str]:
    if not system:
        return None
    s = system.lower().strip()
    if "loinc" in s or s == "http://loinc.org":
        return "LOINC"
    if "snomed" in s or "snomed.info/sct" in s:
        return "SNOMED"
    if "icd-10" in s or "icd10" in s:
        return "ICD-10"
    if "icd-9" in s or "icd9" in s:
        return "ICD-9"
    if "rxnorm" in s or "nlm.nih.gov/research/umls/rxnorm" in s:
        return "RxNorm"
    if "cpt" in s:
        return "CPT"
    return None


def _parse_coding_triplet(coding: str) -> Tuple[str, str, str]:
    # Expected format from normalize.py: system|code|display (some parts may be missing)
    parts = (coding or "").split("|")
    system = parts[0] if len(parts) > 0 else ""
    code = parts[1] if len(parts) > 1 else ""
    display = parts[2] if len(parts) > 2 else ""
    return system.strip(), code.strip(), display.strip()


def _extract_code_refs(codings: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c in codings or []:
        system, code, _display = _parse_coding_triplet(c)
        if not code:
            continue
        label = _coding_label(system)
        if not label:
            continue
        out.setdefault(label, [])
        if code not in out[label]:
            out[label].append(code)
    return out


def _format_code_refs(codings: List[str], prefer: Optional[List[str]] = None) -> str:
    refs = _extract_code_refs(codings)
    if not refs:
        return ""
    order = prefer or ["ICD-10", "SNOMED", "LOINC", "RxNorm", "CPT", "ICD-9"]
    parts: List[str] = []
    for k in order:
        if k in refs and refs[k]:
            codes = refs[k][:3]  # keep compact
            parts.append(f"{k}: {', '.join(codes)}")
    for k, codes in refs.items():
        if k in order or not codes:
            continue
        parts.append(f"{k}: {', '.join(codes[:3])}")
    return "; ".join(parts)


# --- Answer formatting --------------------------------------------------------

def format_answer(lines: List[str], sources: List[str]) -> str:
    if not lines:
        return "Not found in provided records."
    uniq = []
    for s in sources:
        if s and s not in uniq:
            uniq.append(s)
    return "\n".join(lines + ["", "Source: " + ", ".join(uniq)])


def _sort_by_date(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    def parse_dt(v: Any) -> float:
        if not v:
            return -1.0
        try:
            return dtparser.parse(v).timestamp()
        except Exception:
            return -1.0

    return sorted(rows, key=lambda r: parse_dt(r.get(key)), reverse=True)


# --- Handlers ----------------------------------------------------------------

def handle_condition(idx: EngineIndex, entity: str) -> Dict[str, Any]:
    conditions = idx.normalized.conditions
    if not conditions:
        return {"facts": [], "answer": "Not found in provided records."}

    ent = (entity or "").strip().lower()
    matches = []

    if ent:
        for c in conditions:
            name = (c.get("text") or "").lower()
            codes_blob = " ".join(c.get("codings") or []).lower()
            if ent in name or ent in codes_blob:
                matches.append(c)
    else:
        matches = conditions

    if not matches:
        return {"facts": [], "answer": "Not found in provided records."}

    matches = _sort_by_date(matches, key="recorded")

    lines: List[str] = []
    facts: List[Dict[str, Any]] = []
    sources: List[str] = []

    for c in matches:
        name = c.get("text") or "Unknown condition"
        refs = _format_code_refs(c.get("codings") or [], prefer=["ICD-10", "SNOMED", "ICD-9"])
        recorded = c.get("recorded") or c.get("onset") or ""
        line = f"The patient has {name}" if ent else f"Condition: {name}"
        if refs:
            line += f" ({refs})"
        if recorded:
            line += f" recorded on {recorded}"
        lines.append(line)
        facts.append({"source": c["source"], "text": line})
        sources.append(c["source"])

    return {"facts": facts, "answer": format_answer(lines, sources)}


def handle_medications(idx: EngineIndex, entity: str) -> Dict[str, Any]:
    meds = idx.normalized.medications
    if not meds:
        return {"facts": [], "answer": "Not found in provided records."}

    ent = (entity or "").strip().lower()
    filtered: List[Dict[str, Any]] = []

    # Prefer reason-based match
    if ent:
        for m in meds:
            if ent in (m.get("reason") or "").lower():
                filtered.append(m)

        # fallback: match medication text
        if not filtered:
            for m in meds:
                if ent in (m.get("text") or "").lower():
                    filtered.append(m)

    use_list = filtered if filtered else meds
    # Prefer active/current first
    use_list = [m for m in use_list if (m.get("status") or "").lower() not in {"stopped", "entered-in-error"}] or use_list
    use_list = _sort_by_date(use_list, key="effective")

    lines: List[str] = []
    facts: List[Dict[str, Any]] = []
    sources: List[str] = []

    for m in use_list:
        name = m.get("text") or "Unknown medication"
        refs = _format_code_refs(m.get("codings") or [], prefer=["RxNorm", "SNOMED"])
        dt = m.get("effective") or ""
        dose = m.get("dosageText") or ""
        reason = m.get("reason") or ""
        line = f"Medication: {name}"
        if refs:
            line += f" ({refs})"
        if dose:
            line += f"; dosage: {dose}"
        if reason:
            line += f"; indication: {reason}"
        if dt:
            line += f"; date: {dt}"
        lines.append(line)
        facts.append({"source": m["source"], "text": line})
        sources.append(m["source"])

    return {"facts": facts, "answer": format_answer(lines, sources)}


def handle_allergies(idx: EngineIndex) -> Dict[str, Any]:
    alls = idx.normalized.allergies
    if not alls:
        return {"facts": [], "answer": "Not found in provided records."}

    lines: List[str] = []
    facts: List[Dict[str, Any]] = []
    sources: List[str] = []

    for a in alls:
        name = a.get("text") or "Unknown allergen"
        refs = _format_code_refs(a.get("codings") or [], prefer=["SNOMED", "RxNorm"])
        reactions = a.get("reactions") or ""
        line = f"Allergy: {name}"
        if refs:
            line += f" ({refs})"
        if reactions:
            line += f"; reaction: {reactions}"
        lines.append(line)
        facts.append({"source": a["source"], "text": line})
        sources.append(a["source"])

    return {"facts": facts, "answer": format_answer(lines, sources)}


def handle_avoid_due_to_allergies(idx: EngineIndex) -> Dict[str, Any]:
    meds = idx.normalized.medications or []
    alls = idx.normalized.allergies or []

    current_meds = [m for m in meds if (m.get("status") or "").lower() not in {"stopped", "entered-in-error"}]
    current_med_sources = [m.get("source") for m in current_meds if m.get("source")]

    if not alls:
        return {"facts": [], "answer": "Not found in provided records."}

    # Basic allergen -> keyword mapping (conservative, demo-friendly)
    allergen_keywords = {
        "penicillin": ["penicillin", "amoxicillin", "ampicillin", "piperacillin", "ticarcillin", "nafcillin", "oxacillin", "dicloxacillin"],
        "sulfonamide": ["sulfonamide", "sulfa", "sulfameth", "sulfadiazine", "sulfisoxazole", "bactrim", "septra", "trimethoprim-sulfamethoxazole"],
        "iodinated contrast": ["iodinated", "contrast", "iohexol", "iopamidol", "ioversol", "iothalamate"],
    }

    allergy_facts: List[Dict[str, Any]] = []
    avoid_recs: List[str] = []
    sources: List[str] = []

    # Build avoid recommendations from recorded allergies
    for a in alls:
        a_name = (a.get("text") or "").strip()
        a_lower = a_name.lower()
        refs = _format_code_refs(a.get("codings") or [], prefer=["SNOMED", "RxNorm"])
        reactions = a.get("reactions") or ""
        if not a_name:
            continue

        # Decide if this is medication-related
        is_med_related = any(k in a_lower for k in ["penicillin", "sulfon", "drug", "antibiotic", "contrast", "iodin"])
        if not is_med_related:
            # Still include as allergy fact, but not a medication avoidance recommendation
            line = f"Allergy: {a_name}"
            if refs:
                line += f" ({refs})"
            if reactions:
                line += f"; reaction: {reactions}"
            allergy_facts.append({"source": a["source"], "text": line})
            sources.append(a["source"])
            continue

        # Build a conservative avoidance statement
        avoid_stmt = f"Avoid medications in the {a_name} class due to recorded allergy"
        if refs:
            avoid_stmt += f" ({refs})"
        if reactions:
            avoid_stmt += f"; reaction: {reactions}"
        avoid_recs.append(avoid_stmt)
        allergy_facts.append({"source": a["source"], "text": avoid_stmt})
        sources.append(a["source"])

    # Match current meds against allergy keywords (if any)
    matched_lines: List[str] = []
    matched_sources: List[str] = []
    if meds:
        for m in meds:
            med_name = (m.get("text") or "").lower()
            if not med_name:
                continue
            for a in alls:
                a_name = (a.get("text") or "").lower()
                if not a_name:
                    continue
                keys = []
                for k, kws in allergen_keywords.items():
                    if k in a_name:
                        keys.extend(kws)
                # Also include direct allergy name token
                keys.append(a_name)

                if any(kw and kw in med_name for kw in keys):
                    med_refs = _format_code_refs(m.get("codings") or [], prefer=["RxNorm", "SNOMED"])
                    alg_refs = _format_code_refs(a.get("codings") or [], prefer=["SNOMED", "RxNorm"])
                    line = f"Avoid {m.get('text') or 'this medication'}"
                    if med_refs:
                        line += f" ({med_refs})"
                    line += f" due to allergy to {a.get('text') or 'recorded allergen'}"
                    if alg_refs:
                        line += f" ({alg_refs})"
                    matched_lines.append(line)
                    matched_sources.extend([m["source"], a["source"]])

    lines: List[str] = []
    facts: List[Dict[str, Any]] = []

    if matched_lines:
        # Best case: current meds conflict with allergies
        for line in matched_lines:
            lines.append(line)
        # Add allergy-derived avoid statements as supporting facts
        for f in allergy_facts:
            if f["text"] not in lines:
                lines.append(f["text"])
        facts = [{"source": s, "text": t} for s, t in zip(matched_sources, matched_lines)] + allergy_facts
        sources = matched_sources + sources
        return {"facts": facts, "answer": format_answer(lines, sources)}

    # No direct conflicts found
    if not avoid_recs:
        # Only non-med allergies exist
        out = handle_allergies(idx)
        return out

    # Include current medication resources for traceability
    med_src_preview = ", ".join(current_med_sources[:6]) + ("..." if len(current_med_sources) > 6 else "")
    if med_src_preview:
        lines.append(f"No current medications ({med_src_preview}) directly match recorded medication allergies.")
    else:
        lines.append("No current medications directly match recorded medication allergies.")
    lines.extend(avoid_recs)
    # Add current medications to citations for the "no conflicts" conclusion
    sources.extend([s for s in current_med_sources if s])
    facts = [{"source": f["source"], "text": f["text"]} for f in allergy_facts]
    return {"facts": facts, "answer": format_answer(lines, sources)}


def _is_abnormal(interpretation_text: str, interpretation_codings: List[str]) -> Optional[str]:
    # Returns a short label like "High" / "Low" / "Abnormal" / "Normal"
    codes = []
    for c in interpretation_codings or []:
        _sys, code, display = _parse_coding_triplet(c)
        if code:
            codes.append(code.upper())
        if display:
            # Sometimes display is already "High"/"Normal"
            pass

    if any(c in {"H", "HH", "L", "LL", "A", "AA"} for c in codes):
        if "H" in codes or "HH" in codes:
            return "High"
        if "L" in codes or "LL" in codes:
            return "Low"
        return "Abnormal"

    t = (interpretation_text or "").lower()
    if any(w in t for w in ["high", "elevated", "above"]):
        return "High"
    if any(w in t for w in ["low", "below"]):
        return "Low"
    if "normal" in t:
        return "Normal"
    return None


def handle_labs(idx: EngineIndex, query_text: str) -> Dict[str, Any]:
    obs = idx.normalized.observations
    if not obs:
        return {"facts": [], "answer": "Not found in provided records."}

    q = (query_text or "").lower()
    # Common lab terms to help narrow results
    target_terms = ["hba1c", "a1c", "hemoglobin a1c", "creatinine", "egfr", "bun", "urea", "renal", "kidney", "cholesterol", "glucose", "blood pressure", "bmi"]
    want_terms = [t for t in target_terms if t in q]

    obs_use = obs
    if want_terms:
        cand = []
        for o in obs:
            name = (o.get("text") or "").lower()
            codes = " ".join(o.get("codings") or []).lower()
            if any(t in name for t in want_terms) or any(t in codes for t in want_terms):
                cand.append(o)
        obs_use = cand

    if not obs_use:
        return {"facts": [], "answer": "Not found in provided records."}

    # Group by LOINC code if available (preferred), else by test name.
    def obs_ts(o: Dict[str, Any]) -> float:
        dt = o.get("effective") or o.get("issued")
        if not dt:
            return -1.0
        try:
            return dtparser.parse(dt).timestamp()
        except Exception:
            return -1.0

    best_by_key: Dict[str, Dict[str, Any]] = {}
    for o in obs_use:
        codings = o.get("codings") or []
        refs = _extract_code_refs(codings)
        loinc_code = (refs.get("LOINC") or [None])[0]
        key = f"LOINC:{loinc_code}" if loinc_code else f"NAME:{(o.get('text') or '').lower()}"
        ts = obs_ts(o)
        prev = best_by_key.get(key)
        if prev is None or ts > obs_ts(prev):
            best_by_key[key] = o

    selected = list(best_by_key.values())
    selected.sort(key=obs_ts, reverse=True)
    selected = selected[: min(10, len(selected))]

    lines: List[str] = []
    facts: List[Dict[str, Any]] = []
    sources: List[str] = []

    for o in selected:
        test = o.get("text") or "Unknown test"
        value = o.get("value") or ""
        dt = o.get("effective") or o.get("issued") or ""
        loinc = _format_code_refs(o.get("codings") or [], prefer=["LOINC"])
        interp_label = _is_abnormal(o.get("interpretation") or "", o.get("interpretationCodings") or [])
        line = f"The most recent {test}"
        if loinc:
            line += f" ({loinc})"
        if value:
            line += f" value is {value}"
        if interp_label:
            if interp_label in {"High", "Low", "Abnormal"}:
                line += f" — abnormal ({interp_label})"
            elif interp_label == "Normal":
                line += " — normal"
        if dt:
            line += f"; date: {dt}"

        lines.append(line)
        facts.append({"source": o["source"], "text": line})
        sources.append(o["source"])

    return {"facts": facts, "answer": format_answer(lines, sources)}


def handle_diabetes_complications(idx: EngineIndex) -> Dict[str, Any]:
    conditions = idx.normalized.conditions or []
    if not conditions:
        return {"facts": [], "answer": "Not found in provided records."}

    diabetes_markers = ["diabetes", "diabetic"]
    icd_diabetes_prefixes = ("E10", "E11", "E13")  # common diabetes codes

    def is_diabetes(c: Dict[str, Any]) -> bool:
        name = (c.get("text") or "").lower()
        if any(m in name for m in diabetes_markers):
            return True
        refs = _extract_code_refs(c.get("codings") or [])
        icd = refs.get("ICD-10") or []
        return any(code.upper().startswith(icd_diabetes_prefixes) for code in icd)

    has_diabetes = any(is_diabetes(c) for c in conditions)

    # Complication keywords (small, explainable list)
    complication_keywords = [
        "neuropathy", "retinopathy", "nephropathy", "kidney", "ckd", "chronic kidney",
        "ulcer", "foot", "amac", "macular", "microvascular", "macrovascular", "gastroparesis",
    ]

    complications: List[Dict[str, Any]] = []
    for c in conditions:
        name = (c.get("text") or "").lower()
        if any(m in name for m in diabetes_markers):
            # "diabetic nephropathy" etc counts as complication
            complications.append(c)
            continue
        if has_diabetes and any(k in name for k in complication_keywords):
            complications.append(c)

    if not complications:
        return {"facts": [], "answer": "Not found in provided records."}

    complications = _sort_by_date(complications, key="recorded")

    lines: List[str] = []
    facts: List[Dict[str, Any]] = []
    sources: List[str] = []

    for c in complications:
        name = c.get("text") or "Unknown condition"
        refs = _format_code_refs(c.get("codings") or [], prefer=["ICD-10", "SNOMED"])
        recorded = c.get("recorded") or c.get("onset") or ""
        line = f"Diabetes-related complication: {name}"
        if refs:
            line += f" ({refs})"
        if recorded:
            line += f" recorded on {recorded}"
        lines.append(line)
        facts.append({"source": c["source"], "text": line})
        sources.append(c["source"])

    return {"facts": facts, "answer": format_answer(lines, sources)}


def handle_encounters(idx: EngineIndex) -> Dict[str, Any]:
    enc = idx.normalized.encounters
    if not enc:
        return {"facts": [], "answer": "Not found in provided records."}

    enc_sorted = _sort_by_date(enc, key="start")
    last_two = enc_sorted[:2] if enc_sorted else enc[:2]

    lines: List[str] = []
    facts: List[Dict[str, Any]] = []
    sources: List[str] = []

    for e in last_two:
        start = e.get("start") or ""
        typ = e.get("type") or "Unknown encounter"
        reason = e.get("reason") or ""
        line = f"Encounter on {start}: {typ}" if start else f"Encounter: {typ}"
        if reason:
            line += f"; reason: {reason}"
        lines.append(line)
        facts.append({"source": e["source"], "text": line})
        sources.append(e["source"])

    return {"facts": facts, "answer": format_answer(lines, sources)}
