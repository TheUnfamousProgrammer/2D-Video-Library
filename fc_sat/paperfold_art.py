"""Locate supplied art. Ingest (keying, scale, contact sheet) is a later step."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets" / "art" / "MANIFEST.yaml"
ART_ROOT = ROOT / "assets" / "art"


def art_id(filename: str) -> str:
    """Id is the prefix before a `.png_` or `_png_` upload suffix."""
    name = Path(filename).name
    for token in (".png_", "_png_"):
        if token in name:
            return name.split(token, 1)[0]
    return Path(name).stem


def load_manifest(path: Path | None = None) -> dict:
    path = path or MANIFEST_PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} is not a mapping")
    return raw


def find_source(folder: Path, asset_id: str) -> Path | None:
    hits = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and art_id(path.name) == asset_id
    )
    return hits[0] if hits else None


def missing_assets(root: Path | None = None, manifest: dict | None = None) -> list[str]:
    root = root or ART_ROOT
    manifest = manifest or load_manifest()
    missing = []
    for kind in ("plates", "objects"):
        folder = root / kind
        if not folder.is_dir():
            missing.extend(item["id"] for item in manifest.get(kind) or [])
            continue
        for item in manifest.get(kind) or []:
            if find_source(folder, item["id"]) is None:
                missing.append(item["id"])
    return missing
