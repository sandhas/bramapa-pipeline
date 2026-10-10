"""Level manifest (``level.toml``) writer/reader, spec v0 (``chunk.md``)."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import SPEC_VERSION
from .chunk import DEFAULT_CHUNK_SIZE


@dataclass
class LevelManifest:
    level: int  # 1 = país, 2 = estado, 3 = município
    entity: str  # "<tipo>:<codigo>", e.g. "pais:BR"
    name: str
    width: int  # cells
    height: int
    cell_meters: float
    crs: str = "EPSG:5880"
    origin_x: float = 0.0
    origin_y: float = 0.0
    rotation_deg: float = 45.0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    tileset_id: str = "bramapa-base"
    tileset_version: str = "0.0.1"
    period_start: str | None = None  # ISO 8601: "YYYY", "YYYY-MM" or "YYYY-MM-DD"
    period_end: str | None = None
    entities: list[tuple[int, str, str]] = field(default_factory=list)
    """``(index, id, name)`` rows written as ``[[entities]]``; index 0 is reserved (none)."""

    def __post_init__(self) -> None:
        self.entities = [(int(i), str(eid), str(name)) for i, eid, name in self.entities]
        seen: set[int] = set()
        for idx, eid, _name in self.entities:
            if not (1 <= idx <= 0xFFFF):
                raise ValueError(f"entity index {idx} out of range 1..65535 (0 is reserved)")
            if idx in seen:
                raise ValueError(f"duplicate entity index {idx}")
            seen.add(idx)
            if ":" not in eid:
                raise ValueError(f"entity id must look like '<tipo>:<codigo>', got {eid!r}")
        if self.level not in (1, 2, 3):
            raise ValueError(f"level must be 1, 2 or 3, got {self.level}")
        if ":" not in self.entity:
            raise ValueError(f"entity must look like '<tipo>:<codigo>', got {self.entity!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if (self.period_start is None) != (self.period_end is None):
            raise ValueError("period start and end must both be set or both absent")

    @property
    def chunks_x(self) -> int:
        """Number of chunk columns. Partial trailing chunks are padded with 0 (empty)."""
        return math.ceil(self.width / self.chunk_size)

    @property
    def chunks_y(self) -> int:
        return math.ceil(self.height / self.chunk_size)

    def chunk_coords(self) -> list[tuple[int, int]]:
        return [(cx, cy) for cy in range(self.chunks_y) for cx in range(self.chunks_x)]


def parse_period_date(s: str, *, end: bool) -> date:
    """Expand an ISO ``YYYY``/``YYYY-MM``/``YYYY-MM-DD`` to the first (or last) day it covers."""
    parts = s.split("-")
    if len(parts) == 1:
        y = int(parts[0])
        return date(y, 12, 31) if end else date(y, 1, 1)
    if len(parts) == 2:
        y, m = int(parts[0]), int(parts[1])
        if not end:
            return date(y, m, 1)
        nm = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        return date.fromordinal(nm.toordinal() - 1)
    if len(parts) == 3:
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"invalid ISO 8601 period date: {s!r}")


def _toml_str(s: str) -> str:
    # Basic string: escape backslash, quote and control chars. Good enough for names/ids.
    out = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{out}"'


def _toml_float(x: float) -> str:
    # Always render as a TOML float (with a dot) so the engine reads a consistent type.
    return repr(float(x))


def render_level_toml(m: LevelManifest) -> str:
    lines = [
        f"spec_version = {SPEC_VERSION}",
        "",
        "[level]",
        f"level = {m.level}",
        f"entity = {_toml_str(m.entity)}",
        f"name = {_toml_str(m.name)}",
        "",
        "[grid]",
        f"width = {m.width}",
        f"height = {m.height}",
        f"chunk_size = {m.chunk_size}",
        f"cell_meters = {_toml_float(m.cell_meters)}",
        "",
        "[projection]",
        f"crs = {_toml_str(m.crs)}",
        f"origin_x = {_toml_float(m.origin_x)}",
        f"origin_y = {_toml_float(m.origin_y)}",
        f"rotation_deg = {_toml_float(m.rotation_deg)}",
        "",
        "[tileset]",
        f"id = {_toml_str(m.tileset_id)}",
        f"version = {_toml_str(m.tileset_version)}",
    ]
    if m.period_start is not None and m.period_end is not None:
        lines += [
            "",
            "[period]",
            f"start = {_toml_str(m.period_start)}",
            f"end = {_toml_str(m.period_end)}",
        ]
    for idx, eid, name in m.entities:
        lines += [
            "",
            "[[entities]]",
            f"index = {idx}",
            f"id = {_toml_str(eid)}",
            f"name = {_toml_str(name)}",
        ]
    return "\n".join(lines) + "\n"


def write_level_toml(path: str | Path, manifest: LevelManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_level_toml(manifest), encoding="utf-8")
    return path


def read_level_toml(path: str | Path) -> LevelManifest:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    spec = doc.get("spec_version")
    if spec != SPEC_VERSION:
        raise ValueError(f"unsupported spec_version {spec!r} in {path}")
    lv, grid, proj, ts = doc["level"], doc["grid"], doc["projection"], doc["tileset"]
    period = doc.get("period") or {}
    return LevelManifest(
        level=lv["level"],
        entity=lv["entity"],
        name=lv["name"],
        width=grid["width"],
        height=grid["height"],
        chunk_size=grid.get("chunk_size", DEFAULT_CHUNK_SIZE),
        cell_meters=float(grid["cell_meters"]),
        crs=proj["crs"],
        origin_x=float(proj.get("origin_x", 0.0)),
        origin_y=float(proj.get("origin_y", 0.0)),
        rotation_deg=float(proj.get("rotation_deg", 45.0)),
        tileset_id=ts["id"],
        tileset_version=ts["version"],
        period_start=period.get("start"),
        period_end=period.get("end"),
        entities=[(e["index"], e["id"], e["name"]) for e in doc.get("entities", [])],
    )
