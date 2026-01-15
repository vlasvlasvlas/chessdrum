"""
Library loaders for ChessDrum (sound kits + pattern presets).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_SOUND_LIBRARIES: List[Dict[str, Any]] = [
    {
        "id": "classic",
        "name": "Classic",
        "description": "Balanced synthetic drum kit.",
        "author": "ChessDrum",
        "date": "2025-02-01",
        "active": True,
        "type": "builtin",
    },
    {
        "id": "real",
        "name": "Real",
        "description": "Punchy acoustic-style kit.",
        "author": "ChessDrum",
        "date": "2025-02-01",
        "active": True,
        "type": "builtin",
    },
    {
        "id": "dnb",
        "name": "DnB",
        "description": "Tight, fast drum-and-bass kit.",
        "author": "ChessDrum",
        "date": "2025-02-01",
        "active": True,
        "type": "builtin",
    },
    {
        "id": "ethnic",
        "name": "Ethnic",
        "description": "Softer, more organic percussion.",
        "author": "ChessDrum",
        "date": "2025-02-01",
        "active": True,
        "type": "builtin",
    },
    {
        "id": "8bit",
        "name": "8bit",
        "description": "Chiptune / bitcrushed kit.",
        "author": "ChessDrum",
        "date": "2025-02-01",
        "active": True,
        "type": "builtin",
    },
    {
        "id": "bizarre",
        "name": "Bizarre",
        "description": "Experimental, noisy, and weird.",
        "author": "ChessDrum",
        "date": "2025-02-01",
        "active": True,
        "type": "builtin",
    },
]


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    # Try relative to cwd
    cwd_path = os.path.join(os.getcwd(), path)
    if os.path.exists(cwd_path):
        return os.path.abspath(cwd_path)
    # Try relative to repo root (src/..)
    root_path = os.path.join(os.path.dirname(__file__), "..", path)
    if os.path.exists(root_path):
        return os.path.abspath(root_path)
    return os.path.abspath(cwd_path)


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    resolved = _resolve_path(path)
    if not resolved or not os.path.exists(resolved):
        return None
    try:
        with open(resolved, "r") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _normalize_sound_library(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    kit_id = str(entry.get("id", "")).strip()
    if not kit_id:
        return None
    name = str(entry.get("name", kit_id)).strip() or kit_id
    normalized = {
        "id": kit_id,
        "name": name,
        "description": str(entry.get("description", "")).strip(),
        "author": str(entry.get("author", "")).strip(),
        "date": str(entry.get("date", "")).strip(),
        "active": bool(entry.get("active", True)),
        "type": str(entry.get("type", "builtin")).strip().lower(),
        "base_kit": entry.get("base_kit"),
        "notes": entry.get("notes", {}) or {},
        "synth": entry.get("synth", {}) or {},
        "samples": entry.get("samples", {}) or {},
        "gain": entry.get("gain", {}) or {},
    }
    return normalized


def load_sound_libraries(path: Optional[str]) -> List[Dict[str, Any]]:
    data = _load_json(path)
    libs = []
    if isinstance(data, dict):
        libs = data.get("libraries", [])
    if not libs:
        libs = DEFAULT_SOUND_LIBRARIES
    normalized = []
    for entry in libs:
        norm = _normalize_sound_library(entry)
        if norm:
            normalized.append(norm)
    return normalized


def _normalize_pattern_library(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    lib_id = str(entry.get("id", "")).strip()
    if not lib_id:
        return None
    normalized = {
        "id": lib_id,
        "name": str(entry.get("name", lib_id)).strip() or lib_id,
        "description": str(entry.get("description", "")).strip(),
        "author": str(entry.get("author", "")).strip(),
        "date": str(entry.get("date", "")).strip(),
        "active": bool(entry.get("active", True)),
        "kit": entry.get("kit"),
        "patterns": entry.get("patterns", []) or [],
    }
    # Limit to 4 patterns max
    normalized["patterns"] = normalized["patterns"][:4]
    return normalized


def load_pattern_libraries(path: Optional[str]) -> List[Dict[str, Any]]:
    data = _load_json(path)
    libs = []
    if isinstance(data, dict):
        libs = data.get("libraries", [])
    normalized = []
    for entry in libs:
        norm = _normalize_pattern_library(entry)
        if norm:
            normalized.append(norm)
    return normalized
