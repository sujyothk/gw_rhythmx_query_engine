from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class FhirResource:
    resource_type: str
    resource_id: str
    raw: Dict[str, Any]


def load_fhir_dir(data_dir: str) -> List[FhirResource]:
    """
    Load FHIR resources from JSON files in a directory.

    Supports:
    - single-resource JSON
    - list of resources
    - Bundle with entry[].resource

    Failure modes:
    - missing directory -> FileNotFoundError
    - invalid JSON -> json.JSONDecodeError
    """
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    resources: List[FhirResource] = []
    for fp in sorted(p.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict) and obj.get("resourceType") == "Bundle" and isinstance(obj.get("entry"), list):
            for e in obj["entry"]:
                if isinstance(e, dict) and isinstance(e.get("resource"), dict):
                    resources.extend(_parse_item(e["resource"], fallback_file=fp.name))
            continue

        if isinstance(obj, list):
            for item in obj:
                resources.extend(_parse_item(item, fallback_file=fp.name))
        else:
            resources.extend(_parse_item(obj, fallback_file=fp.name))

    return resources


def _parse_item(item: Any, fallback_file: str) -> List[FhirResource]:
    if not isinstance(item, dict):
        return []
    rtype = item.get("resourceType") or "Unknown"
    rid = item.get("id") or fallback_file
    return [FhirResource(resource_type=rtype, resource_id=rid, raw=item)]
