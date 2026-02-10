from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RoutedQuery:
    intent: str
    entity: str
    raw: str


# NOTE: Use word-boundary regex to avoid substring false positives like:
# - "hypertension" triggering "er"
# - "available" triggering "lab"
_PAT_ALLERGY = re.compile(r"\b(allergy|allergies|allergic|anaphylaxis|rash|hives)\b", re.IGNORECASE)

_PAT_ENCOUNTER = re.compile(
    r"\b(encounter|visit|appointment)\b|\burgent\s+care\b|\bemergency\s+room\b|\ber\b",
    re.IGNORECASE,
)

_PAT_LABS = re.compile(
    r"\b(lab|labs|result|results|observation|loinc|creatinine|egfr|bun|renal|kidney|a1c|hba1c)\b",
    re.IGNORECASE,
)

_PAT_MED = re.compile(r"\b(medication|medications|drug|drugs|prescription|taking)\b", re.IGNORECASE)

_PAT_CONDITION = re.compile(
    r"\b(diagnosis|condition|history\s+of|diagnosed\s+with|does\s+the\s+patient\s+have|do\s+they\s+have)\b",
    re.IGNORECASE,
)


def route(query: str) -> RoutedQuery:
    q = query.strip()
    ql = q.lower()
    entity = _extract_entity(ql)

    # Medication safety / reconciliation
    if re.search(r"\bavoid\w*\b|\bcontraindicat\w*\b", ql) and re.search(r"\ballerg", ql):
        return RoutedQuery(intent="avoid_allergies", entity=entity, raw=query)

    # Diabetes complications
    if re.search(r"\bcomplication(s)?\b", ql) and re.search(r"\b(diabetes|diabetic)\b", ql):
        return RoutedQuery(intent="diabetes_complications", entity=entity, raw=query)

    # Allergies
    if _PAT_ALLERGY.search(ql):
        return RoutedQuery(intent="allergy", entity=entity, raw=query)

    # Encounters
    if _PAT_ENCOUNTER.search(ql):
        return RoutedQuery(intent="encounter", entity=entity, raw=query)

    # Labs / Observations
    if _PAT_LABS.search(ql):
        return RoutedQuery(intent="labs", entity=entity, raw=query)

    # Medications
    if _PAT_MED.search(ql):
        return RoutedQuery(intent="medication", entity=entity, raw=query)

    # Conditions
    if _PAT_CONDITION.search(ql):
        return RoutedQuery(intent="condition", entity=entity, raw=query)

    return RoutedQuery(intent="fallback", entity=entity, raw=query)


def _extract_entity(ql: str) -> str:
    # "have asthma", "diagnosed with asthma"
    m = re.search(r"(?:have|diagnosed\s+with)\s+([a-z0-9\-\s]+?)(\?|\.|,|$)", ql)
    if m:
        return m.group(1).strip()

    # "taking for hypertension"
    m = re.search(r"for\s+([a-z0-9\-\s]+?)(\?|\.|,|$)", ql)
    if m:
        return m.group(1).strip()

    return ""
