from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Chunk:
    source: str
    text: str
    meta: Dict[str, Any]


@dataclass
class Normalized:
    conditions: List[Dict[str, Any]]
    medications: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    allergies: List[Dict[str, Any]]
    encounters: List[Dict[str, Any]]
    chunks: List[Chunk]
