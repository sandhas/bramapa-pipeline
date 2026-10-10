"""Autotiling: turn a grid of covers into a grid of tile ids using ``[[autotile]]`` rules.

Spec (``tileset.md``, "Esclarecimentos v0"): the neighbour mask has 4 bits, ``N = 1`` at
``(x, y-1)``, ``E = 2`` at ``(x+1, y)``, ``S = 4`` at ``(x, y+1)``, ``W = 8`` at ``(x-1, y)``. A bit
is set when that neighbour has the *same cover*; cells outside the grid count as "same cover" so
the map border never grows an outline. The pipeline computes masks; the engine only sees ids.

Cover grids are 2-D arrays indexed ``[y, x]``. Two encodings are accepted:

* string arrays (``dtype=str``/``object``), where ``""`` or ``None`` means empty;
* integer arrays plus a ``codes`` mapping ``code -> cover``, where code ``0`` (or any code
  missing from the mapping) means empty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .tileset import EMPTY_TILE, TileClass, Tileset, TilesetError

COVER_DTYPE = "<U32"
"""dtype used for string cover grids (longest v0 cover name is ``limite_municipio``)."""

EMPTY_COVER = ""

MASK_N, MASK_E, MASK_S, MASK_W = 1, 2, 4, 8


def empty_cover_grid(height: int, width: int) -> np.ndarray:
    return np.full((height, width), EMPTY_COVER, dtype=COVER_DTYPE)


def encode_covers(
    cover_grid: np.ndarray, codes: Mapping[int, str] | Sequence[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Normalise any accepted cover grid to ``(int codes, names)`` with ``names[0] == ""``.

    ``codes[k]`` is the cover name of code ``k``; code 0 is always the empty cover.
    """
    grid = np.asarray(cover_grid)
    if grid.ndim != 2:
        raise ValueError(f"cover grid must be 2-D, got shape {grid.shape}")
    if grid.dtype.kind in "iu":
        if codes is None:
            raise ValueError("integer cover grids need a `codes` mapping code -> cover name")
        if isinstance(codes, Mapping):
            code_names = dict(codes)
        else:
            code_names = dict(enumerate(codes))
        present = np.unique(grid).tolist()
        names = [EMPTY_COVER]
        remap = np.zeros(int(max(present, default=0)) + 1, dtype=np.int64)
        for c in present:
            if c < 0:
                raise ValueError(f"negative cover code {c}")
            name = code_names.get(c, EMPTY_COVER) if c != 0 else EMPTY_COVER
            if name == EMPTY_COVER:
                continue
            if name in names:
                remap[c] = names.index(name)
            else:
                remap[c] = len(names)
                names.append(name)
        return remap[grid], names
    if codes is not None:
        raise ValueError("`codes` only applies to integer cover grids")
    # String/object grid: None -> "" then factorise.
    if grid.dtype.kind == "O":
        grid = np.where(grid == None, EMPTY_COVER, grid).astype(COVER_DTYPE)  # noqa: E711
    else:
        grid = grid.astype(COVER_DTYPE)
    uniq, inverse = np.unique(grid, return_inverse=True)
    names = uniq.tolist()
    inverse = inverse.reshape(grid.shape)
    if names and names[0] == EMPTY_COVER:
        return inverse, names
    # No empty cells: prepend "" so code 0 is still the empty cover.
    return inverse + 1, [EMPTY_COVER, *names]


def neighbour_mask(codes: np.ndarray) -> np.ndarray:
    """4-bit same-cover mask per cell (``uint8``), vectorised; out-of-grid counts as same."""
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"cover grid must be 2-D, got shape {codes.shape}")
    same_n = np.ones(codes.shape, dtype=bool)
    same_e = np.ones(codes.shape, dtype=bool)
    same_s = np.ones(codes.shape, dtype=bool)
    same_w = np.ones(codes.shape, dtype=bool)
    same_n[1:, :] = codes[1:, :] == codes[:-1, :]  # neighbour at (x, y-1)
    same_s[:-1, :] = codes[:-1, :] == codes[1:, :]  # neighbour at (x, y+1)
    same_e[:, :-1] = codes[:, :-1] == codes[:, 1:]  # neighbour at (x+1, y)
    same_w[:, 1:] = codes[:, 1:] == codes[:, :-1]  # neighbour at (x-1, y)
    mask = (
        same_n.astype(np.uint8) * MASK_N
        | same_e.astype(np.uint8) * MASK_E
        | same_s.astype(np.uint8) * MASK_S
        | same_w.astype(np.uint8) * MASK_W
    )
    return mask.astype(np.uint8)


def apply_autotile(
    cover_grid: np.ndarray,
    tileset: Tileset,
    tile_class: TileClass = "terrain",
    codes: Mapping[int, str] | Sequence[str] | None = None,
) -> np.ndarray:
    """Resolve a cover grid to a ``uint16`` grid of tile ids for one layer.

    Covers with an ``[[autotile]]`` rule pick ``masks[mask]`` or the rule's ``default``; covers
    without one must map to exactly one tile of ``tile_class`` with that cover. Empty cells are 0.
    """
    code_grid, names = encode_covers(cover_grid, codes)
    out = np.zeros(code_grid.shape, dtype=np.uint16)
    if len(names) == 1:
        return out  # all empty
    mask = neighbour_mask(code_grid)
    for code, cover in enumerate(names):
        if code == 0:
            continue
        cells = code_grid == code
        rule = tileset.autotile.get(cover)
        if rule is not None:
            lut = np.asarray(rule.lut(), dtype=np.uint16)
            out[cells] = lut[mask[cells]]
        else:
            out[cells] = tileset.tile_for_cover(tile_class, cover)
    return out


def describe_masks(cover_grid: np.ndarray, cover: str) -> list[str]:
    """ASCII rows showing the mask (hex digit) of every cell with ``cover``; others are ``.``."""
    code_grid, names = encode_covers(cover_grid)
    if cover not in names:
        raise TilesetError(f"cover {cover!r} not present in grid")
    mask = neighbour_mask(code_grid)
    sel = code_grid == names.index(cover)
    rows = []
    for y in range(code_grid.shape[0]):
        rows.append("".join(f"{mask[y, x]:x}" if sel[y, x] else "." for x in range(code_grid.shape[1])))
    return rows


__all__ = [
    "COVER_DTYPE",
    "EMPTY_COVER",
    "EMPTY_TILE",
    "MASK_E",
    "MASK_N",
    "MASK_S",
    "MASK_W",
    "apply_autotile",
    "describe_masks",
    "empty_cover_grid",
    "encode_covers",
    "neighbour_mask",
]
