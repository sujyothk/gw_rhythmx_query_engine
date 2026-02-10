from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dateutil import parser as dtparser

from .types import Chunk, Normalized
from .fhir_loader import FhirResource


def normalize(resources: List[FhirResource]) -> Normalized:
    conditions: List[Dict[str, Any]] = []
    medications: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    allergies: List[Dict[str, Any]] = []
    encounters: List[Dict[str, Any]] = []
    chunks: List[Chunk] = []

    for r in resources:
        source = f"{r.resource_type}/{r.resource_id}"
        raw = r.raw

        if r.resource_type == "Condition":
            row = {
                "source": source,
                "text": _best_text(raw.get("code")),
                "codings": _all_codings(raw.get("code")),
                "onset": _best_date(raw.get("onsetDateTime")),
                "recorded": _best_date(raw.get("recordedDate")),
                "clinicalStatus": _best_text(raw.get("clinicalStatus")),
                "verificationStatus": _best_text(raw.get("verificationStatus")),
            }
            conditions.append(row)
            chunks.append(Chunk(source=source, text=_chunk_text_condition(row), meta=row))

        elif r.resource_type in ("MedicationStatement", "MedicationRequest"):
            row = _normalize_medication(r.resource_type, source, raw)
            medications.append(row)
            chunks.append(Chunk(source=source, text=_chunk_text_med(row), meta=row))

        elif r.resource_type == "Observation":
            interp_txt, interp_codings = _interpretation(raw.get("interpretation"))
            row = {
                "source": source,
                "text": _best_text(raw.get("code")),
                "codings": _all_codings(raw.get("code")),
                "value": _obs_value(raw),
                "effective": _best_date(raw.get("effectiveDateTime") or (raw.get("period") or {}).get("start")),
                "issued": _best_date(raw.get("issued")),
                "status": raw.get("status"),
                "interpretation": interp_txt,
                "interpretationCodings": interp_codings,
            }
            observations.append(row)
            chunks.append(Chunk(source=source, text=_chunk_text_obs(row), meta=row))

        elif r.resource_type == "AllergyIntolerance":
            row = {
                "source": source,
                "text": _best_text(raw.get("code")),
                "codings": _all_codings(raw.get("code")),
                "criticality": raw.get("criticality"),
                "clinicalStatus": _best_text(raw.get("clinicalStatus")),
                "verificationStatus": _best_text(raw.get("verificationStatus")),
                "category": raw.get("category") or [],
                "reactions": _reactions(raw.get("reaction")),
            }
            allergies.append(row)
            chunks.append(Chunk(source=source, text=_chunk_text_allergy(row), meta=row))

        elif r.resource_type == "Encounter":
            period = raw.get("period") or {}
            row = {
                "source": source,
                "type": _best_text((raw.get("type") or [{}])[0] if isinstance(raw.get("type"), list) else raw.get("type")),
                "reason": _best_text((raw.get("reasonCode") or [{}])[0] if isinstance(raw.get("reasonCode"), list) else raw.get("reasonCode")),
                "start": _best_date(period.get("start")),
                "end": _best_date(period.get("end")),
                "status": raw.get("status"),
            }
            encounters.append(row)
            chunks.append(Chunk(source=source, text=_chunk_text_encounter(row), meta=row))

        else:
            # Fallback: still index as generic chunk for recall.
            txt = _best_text(raw.get("text")) or raw.get("resourceType", "")
            chunks.append(Chunk(source=source, text=f"Resource: {source}\n{txt}", meta={"source": source}))

    return Normalized(
        conditions=conditions,
        medications=medications,
        observations=observations,
        allergies=allergies,
        encounters=encounters,
        chunks=chunks,
    )


# --- Normalizers --------------------------------------------------------------

def _normalize_medication(resource_type: str, source: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    # MedicationStatement fields
    if resource_type == "MedicationStatement":
        eff = raw.get("effectiveDateTime")
        if not eff:
            effp = raw.get("effectivePeriod") or {}
            eff = effp.get("start") or effp.get("end")
        if not eff:
            eff = raw.get("dateAsserted")

        return {
            "source": source,
            "resourceType": resource_type,
            "text": _best_text(raw.get("medicationCodeableConcept")),
            "codings": _all_codings(raw.get("medicationCodeableConcept")),
            "status": raw.get("status"),
            "effective": _best_date(eff),
            "dosageText": _dosage_text(raw.get("dosage")),
            "reason": _best_text((raw.get("reasonCode") or [{}])[0] if isinstance(raw.get("reasonCode"), list) else raw.get("reasonCode")),
        }

    # MedicationRequest fields
    med_cc = raw.get("medicationCodeableConcept")
    authored = raw.get("authoredOn") or raw.get("dateWritten")
    return {
        "source": source,
        "resourceType": resource_type,
        "text": _best_text(med_cc),
        "codings": _all_codings(med_cc),
        "status": raw.get("status") or raw.get("intent"),
        "effective": _best_date(authored),
        "dosageText": _dosage_instruction_text(raw.get("dosageInstruction")),
        "reason": _best_text((raw.get("reasonCode") or [{}])[0] if isinstance(raw.get("reasonCode"), list) else raw.get("reasonCode")),
    }


def _dosage_instruction_text(dosage_instruction: Any) -> str:
    if not isinstance(dosage_instruction, list):
        return ""
    parts: List[str] = []
    for d in dosage_instruction:
        if not isinstance(d, dict):
            continue
        txt = d.get("text") or ""
        if txt:
            parts.append(txt)
    return "; ".join([p for p in parts if p])


def _interpretation(interpretation: Any) -> Tuple[str, List[str]]:
    # Returns (interpretation_text, interpretation_codings)
    if not isinstance(interpretation, list):
        return "", []
    texts: List[str] = []
    codings: List[str] = []
    for item in interpretation:
        if not isinstance(item, dict):
            continue
        t = _best_text(item) or item.get("text") or ""
        if t:
            texts.append(t)
        codings.extend(_all_codings(item))
    # Prefer unique
    uniq = []
    for c in codings:
        if c and c not in uniq:
            uniq.append(c)
    return "; ".join(texts), uniq


def _obs_value(raw: Dict[str, Any]) -> str:
    # Prefer component-based formatting when available (e.g., Blood Pressure 145/92)
    comps = raw.get("component")
    if isinstance(comps, list) and comps:
        # Try blood pressure special case: systolic + diastolic
        sys_v = None
        dia_v = None
        unit = None
        for c in comps:
            if not isinstance(c, dict):
                continue
            code_txt = _best_text(c.get("code")) or ""
            vq = c.get("valueQuantity") or {}
            val = vq.get("value")
            if val is None:
                continue
            if unit is None:
                unit = vq.get("unit")
            if "systolic" in code_txt.lower():
                sys_v = val
            if "diastolic" in code_txt.lower():
                dia_v = val
        if sys_v is not None and dia_v is not None:
            return f"{sys_v}/{dia_v} {unit or ''}".strip()

        # Generic component string
        parts: List[str] = []
        for c in comps:
            if not isinstance(c, dict):
                continue
            name = _best_text(c.get("code")) or "component"
            vq = c.get("valueQuantity")
            if isinstance(vq, dict) and vq.get("value") is not None:
                parts.append(f"{name}: {vq.get('value')} {vq.get('unit') or ''}".strip())
        if parts:
            return "; ".join(parts)

    # Fallback to standard scalar values
    vq = raw.get("valueQuantity")
    if isinstance(vq, dict):
        val = vq.get("value")
        unit = vq.get("unit")
        if val is not None:
            return f"{val} {unit or ''}".strip()
    vs = raw.get("valueString")
    if isinstance(vs, str):
        return vs
    vb = raw.get("valueBoolean")
    if vb is not None:
        return str(vb)
    return ""


def _reactions(reaction: Any) -> str:
    if not isinstance(reaction, list):
        return ""
    parts: List[str] = []
    for r in reaction:
        if not isinstance(r, dict):
            continue
        mani = r.get("manifestation") or []
        mani_txt = []
        for m in mani:
            if isinstance(m, dict):
                mani_txt.append(_best_text(m) or (m.get("text") or ""))
        sev = _best_text(r.get("severity")) or r.get("severity") or ""
        exp = _best_text(r.get("exposureRoute")) or ""
        piece = ", ".join([x for x in mani_txt if x])
        if sev:
            piece = (piece + f" severity={sev}").strip()
        if exp:
            piece = (piece + f" route={exp}").strip()
        if piece:
            parts.append(piece)
    return "; ".join(parts)


def _dosage_text(dosage: Any) -> str:
    if not isinstance(dosage, list):
        return ""
    parts: List[str] = []
    for d in dosage:
        if not isinstance(d, dict):
            continue
        txt = d.get("text") or ""
        if txt:
            parts.append(txt)
    return "; ".join([p for p in parts if p])


# --- Chunk builders -----------------------------------------------------------

def _chunk_text_condition(row: Dict[str, Any]) -> str:
    return "\n".join([
        f"Resource: {row['source']}",
        f"Condition: {row.get('text','')}",
        f"Codings: {', '.join(row.get('codings', []))}",
        f"Onset: {row.get('onset')}",
        f"Recorded: {row.get('recorded')}",
        f"ClinicalStatus: {row.get('clinicalStatus')}",
        f"VerificationStatus: {row.get('verificationStatus')}",
    ])


def _chunk_text_med(row: Dict[str, Any]) -> str:
    return "\n".join([
        f"Resource: {row['source']}",
        f"Medication: {row.get('text','')}",
        f"Codings: {', '.join(row.get('codings', []))}",
        f"Status: {row.get('status')}",
        f"Effective: {row.get('effective')}",
        f"Dosage: {row.get('dosageText')}",
        f"Reason: {row.get('reason')}",
    ])


def _chunk_text_obs(row: Dict[str, Any]) -> str:
    return "\n".join([
        f"Resource: {row['source']}",
        f"Observation: {row.get('text','')}",
        f"Codings: {', '.join(row.get('codings', []))}",
        f"Value: {row.get('value')}",
        f"Interpretation: {row.get('interpretation')}",
        f"InterpretationCodings: {', '.join(row.get('interpretationCodings', []))}",
        f"Effective: {row.get('effective')}",
        f"Issued: {row.get('issued')}",
        f"Status: {row.get('status')}",
    ])


def _chunk_text_allergy(row: Dict[str, Any]) -> str:
    return "\n".join([
        f"Resource: {row['source']}",
        f"Allergy: {row.get('text','')}",
        f"Codings: {', '.join(row.get('codings', []))}",
        f"Criticality: {row.get('criticality')}",
        f"ClinicalStatus: {row.get('clinicalStatus')}",
        f"VerificationStatus: {row.get('verificationStatus')}",
        f"Category: {', '.join(row.get('category', [])) if isinstance(row.get('category'), list) else row.get('category')}",
        f"Reactions: {row.get('reactions')}",
    ])


def _chunk_text_encounter(row: Dict[str, Any]) -> str:
    return "\n".join([
        f"Resource: {row['source']}",
        f"EncounterType: {row.get('type','')}",
        f"Reason: {row.get('reason','')}",
        f"Start: {row.get('start')}",
        f"End: {row.get('end')}",
        f"Status: {row.get('status')}",
    ])


# --- Generic helpers ----------------------------------------------------------

def _all_codings(cc: Any) -> List[str]:
    # Extract codings from CodeableConcept-like objects: {coding:[{system,code,display}], text:...}
    if not isinstance(cc, dict):
        return []
    codings = cc.get("coding") or []
    out: List[str] = []
    if isinstance(codings, list):
        for c in codings:
            if not isinstance(c, dict):
                continue
            system = str(c.get("system") or "").strip()
            code = str(c.get("code") or "").strip()
            display = str(c.get("display") or "").strip()
            if system or code or display:
                out.append(f"{system}|{code}|{display}")
    return out


def _best_text(cc: Any) -> str:
    if isinstance(cc, dict):
        if isinstance(cc.get("text"), str) and cc.get("text").strip():
            return cc.get("text").strip()
        codings = cc.get("coding")
        if isinstance(codings, list) and codings:
            display = codings[0].get("display")
            if isinstance(display, str) and display.strip():
                return display.strip()
    if isinstance(cc, str):
        return cc.strip()
    return ""


def _best_date(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        try:
            return dtparser.parse(value).isoformat()
        except Exception:
            return value
    return str(value)
