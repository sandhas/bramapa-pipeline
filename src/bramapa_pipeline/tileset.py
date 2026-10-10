"""Tileset manifest (``tiles.toml``) loader, spec v0 (``tileset.md``).

A :class:`Tileset` is the only thing that knows tile ids. The generator works with *covers*
(``"agua"``, ``"floresta"``...) and object *names* (``"arvore"``); :func:`~.autotile.apply_autotile`
and :func:`~.synth.chunkify` resolve them to ids through the tileset, either one loaded from a
real ``tiles.toml`` or the :func:`builtin_tileset` built from the placeholder ids in ``tiles.py``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from . import SPEC_VERSION, tiles

TileClass = Literal["terrain", "overlay", "object"]
TILE_CLASSES: tuple[TileClass, ...] = ("terrain", "overlay", "object")

EMPTY_TILE = 0
"""Reserved tile id: empty cell."""


class TilesetError(ValueError):
    """Raised when ``tiles.toml`` violates the spec or a lookup cannot be resolved."""


@dataclass(frozen=True)
class TileDef:
    id: int
    name: str
    tile_class: TileClass
    col: int = 0
    row: int = 0
    frames: int = 1
    fps: int = 0
    cover: str | None = None


@dataclass(frozen=True)
class AutotileRule:
    """``(cover, neighbour mask) -> tile id``; masks not listed fall back to ``default``."""

    cover: str
    default: int
    masks: dict[int, int] = field(default_factory=dict)

    def tile_for_mask(self, mask: int) -> int:
        return self.masks.get(mask, self.default)

    def lut(self) -> list[int]:
        """Full 16-entry lookup table indexed by mask."""
        return [self.tile_for_mask(m) for m in range(16)]


@dataclass
class Tileset:
    id: str
    version: str
    tiles: dict[int, TileDef]
    autotile: dict[str, AutotileRule] = field(default_factory=dict)
    atlas: str = "atlas.png"
    tile_width: int = 64
    tile_height: int = 32
    cell_width: int = 64
    cell_height: int = 64
    license: str | None = None
    source: Path | None = None

    # Derived indexes, built in __post_init__.
    by_name: dict[str, TileDef] = field(init=False, repr=False)
    by_cover: dict[tuple[TileClass, str], tuple[TileDef, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_name = {}
        by_cover: dict[tuple[TileClass, str], list[TileDef]] = {}
        for t in self.tiles.values():
            if t.name in self.by_name:
                raise TilesetError(f"duplicate tile name {t.name!r} (ids {self.by_name[t.name].id} and {t.id})")
            self.by_name[t.name] = t
            if t.cover is not None:
                by_cover.setdefault((t.tile_class, t.cover), []).append(t)
        self.by_cover = {k: tuple(v) for k, v in by_cover.items()}

    # --- lookups -----------------------------------------------------------------------

    def tile_name(self, tile_id: int) -> str:
        """Human-readable name for an id (``"vazio"`` for 0, ``"?<id>"`` if unknown)."""
        if tile_id == EMPTY_TILE:
            return "vazio"
        t = self.tiles.get(tile_id)
        return t.name if t is not None else f"?{tile_id}"

    def ids_for_class(self, tile_class: TileClass) -> set[int]:
        return {t.id for t in self.tiles.values() if t.tile_class == tile_class}

    def tiles_for_cover(self, tile_class: TileClass, cover: str) -> tuple[TileDef, ...]:
        return self.by_cover.get((tile_class, cover), ())

    def tile_for_cover(self, tile_class: TileClass, cover: str) -> int:
        """Id of the single tile drawing ``cover`` when there is no autotile rule for it."""
        cands = self.tiles_for_cover(tile_class, cover)
        if not cands:
            raise TilesetError(f"tileset {self.id!r} has no {tile_class} tile for cover {cover!r}")
        if len(cands) > 1:
            ids = ", ".join(str(t.id) for t in cands)
            raise TilesetError(
                f"cover {cover!r} ({tile_class}) has {len(cands)} tiles ({ids}) but no [[autotile]] rule"
            )
        return cands[0].id

    def object_id(self, name: str) -> int:
        t = self.by_name.get(name)
        if t is None or t.tile_class != "object":
            raise TilesetError(f"tileset {self.id!r} has no object tile named {name!r}")
        return t.id


# --- loading -----------------------------------------------------------------------------


def _req(table: dict, key: str, where: str):
    if key not in table:
        raise TilesetError(f"{where}: missing required key {key!r}")
    return table[key]


def _parse_tile(entry: dict, where: str) -> TileDef:
    tile_id = _req(entry, "id", where)
    if not isinstance(tile_id, int) or not (1 <= tile_id <= 0xFFFF):
        raise TilesetError(f"{where}: id must be an integer in 1..65535, got {tile_id!r}")
    tile_class = _req(entry, "class", where)
    if tile_class not in TILE_CLASSES:
        raise TilesetError(f"{where}: class must be one of {TILE_CLASSES}, got {tile_class!r}")
    frames = int(entry.get("frames", 1))
    fps = int(entry.get("fps", 0))
    if frames < 1:
        raise TilesetError(f"{where}: frames must be >= 1")
    if (frames == 1) != (fps == 0):
        raise TilesetError(f"{where}: frames = 1 requires fps = 0 and frames > 1 requires fps > 0")
    cover = entry.get("cover")
    if tile_class == "object":
        if cover is not None:
            raise TilesetError(f"{where}: 'cover' is not allowed on object tiles")
    elif not isinstance(cover, str) or not cover:
        raise TilesetError(f"{where}: {tile_class} tiles require a non-empty 'cover'")
    return TileDef(
        id=tile_id,
        name=str(_req(entry, "name", where)),
        tile_class=tile_class,
        col=int(entry.get("col", 0)),
        row=int(entry.get("row", 0)),
        frames=frames,
        fps=fps,
        cover=cover,
    )


def _parse_autotile(entry: dict, tiles_by_id: dict[int, TileDef], where: str) -> AutotileRule:
    cover = _req(entry, "cover", where)
    default = _req(entry, "default", where)
    masks: dict[int, int] = {}
    for key, tile_id in (entry.get("masks") or {}).items():
        if not (isinstance(key, str) and key.isdigit() and str(int(key)) == key and 0 <= int(key) <= 15):
            raise TilesetError(f"{where}: mask keys must be decimal strings \"0\"..\"15\", got {key!r}")
        masks[int(key)] = int(tile_id)
    for tile_id in (default, *masks.values()):
        t = tiles_by_id.get(tile_id)
        if t is None:
            raise TilesetError(f"{where}: references unknown tile id {tile_id}")
        if t.cover != cover:
            raise TilesetError(
                f"{where}: tile {tile_id} ({t.name!r}) has cover {t.cover!r}, rule is for {cover!r}"
            )
    return AutotileRule(cover=str(cover), default=int(default), masks=masks)


def parse_tileset(doc: dict, source: Path | None = None) -> Tileset:
    """Build a :class:`Tileset` from an already-parsed ``tiles.toml`` document."""
    spec = doc.get("spec_version")
    if spec != SPEC_VERSION:
        raise TilesetError(f"unsupported spec_version {spec!r} (expected {SPEC_VERSION})")
    meta = doc.get("tileset")
    if not isinstance(meta, dict):
        raise TilesetError("missing [tileset] table")

    tiles_by_id: dict[int, TileDef] = {}
    for i, entry in enumerate(doc.get("tiles") or []):
        t = _parse_tile(entry, f"[[tiles]] #{i}")
        if t.id in tiles_by_id:
            raise TilesetError(f"duplicate tile id {t.id} ({tiles_by_id[t.id].name!r} and {t.name!r})")
        tiles_by_id[t.id] = t
    if not tiles_by_id:
        raise TilesetError("tileset declares no [[tiles]]")

    autotile: dict[str, AutotileRule] = {}
    for i, entry in enumerate(doc.get("autotile") or []):
        rule = _parse_autotile(entry, tiles_by_id, f"[[autotile]] #{i}")
        if rule.cover in autotile:
            raise TilesetError(f"duplicate [[autotile]] rule for cover {rule.cover!r}")
        autotile[rule.cover] = rule

    return Tileset(
        id=str(_req(meta, "id", "[tileset]")),
        version=str(_req(meta, "version", "[tileset]")),
        tiles=tiles_by_id,
        autotile=autotile,
        atlas=str(meta.get("atlas", "atlas.png")),
        tile_width=int(meta.get("tile_width", 64)),
        tile_height=int(meta.get("tile_height", 32)),
        cell_width=int(meta.get("cell_width", 64)),
        cell_height=int(meta.get("cell_height", 64)),
        license=meta.get("license"),
        source=source,
    )


def load_tileset(path: str | Path) -> Tileset:
    """Load ``tiles.toml`` from a tileset directory (or from the file itself)."""
    p = Path(path)
    if p.is_dir():
        p = p / "tiles.toml"
    if not p.is_file():
        raise TilesetError(f"tileset not found: {p}")
    with open(p, "rb") as f:
        doc = tomllib.load(f)
    return parse_tileset(doc, source=p)


def builtin_tileset() -> Tileset:
    """The placeholder convention from ``tiles.py`` as an in-memory tileset.

    One tile per cover and no autotile rules, so every cover resolves to its base id. Used when
    ``chunk-test`` runs without ``--tileset``; the ids match the ``bramapa-base`` placeholder.
    """
    defs: dict[int, TileDef] = {}
    for tile_id, name in tiles.TERRAIN_NAMES.items():
        defs[tile_id] = TileDef(tile_id, name, "terrain", cover=name)
    for tile_id, name in tiles.OVERLAY_NAMES.items():
        defs[tile_id] = TileDef(tile_id, name, "overlay", cover=name)
    for tile_id, name in tiles.OBJECT_NAMES.items():
        defs[tile_id] = TileDef(tile_id, name, "object")
    return Tileset(id="bramapa-base", version="0.0.1", tiles=defs)
