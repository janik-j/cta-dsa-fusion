"""Shared transforms.json utilities for data_generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scene.validation import validate_transforms


def load_transforms(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"transforms.json must be a JSON object: {p}")
    return data


def write_transforms(path: str | Path, transforms: Mapping[str, Any], *, validate: bool = True) -> None:
    if validate:
        validate_transforms(transforms)
    p = Path(path)
    p.write_text(json.dumps(transforms, indent=2) + "\n", encoding="utf-8")


def ensure_geometry(transforms: dict[str, Any]) -> dict[str, Any]:
    geometry = transforms.get("geometry")
    if not isinstance(geometry, dict):
        geometry = {}
        transforms["geometry"] = geometry
    return geometry


def set_mesh_ref(transforms: dict[str, Any], *, rel_path: str | Path, space: str | None = None) -> None:
    geometry = ensure_geometry(transforms)
    mesh_ref = geometry.get("mesh_ref")
    if not isinstance(mesh_ref, dict):
        mesh_ref = {}
    mesh_ref["path"] = str(rel_path)
    if space is not None:
        mesh_ref["space"] = str(space)
    geometry["mesh_ref"] = mesh_ref
