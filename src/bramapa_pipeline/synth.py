"""Synthetic level generator for ``chunk-test`` (numpy-only value noise).

Not a geodata pipeline: this just produces plausible-looking chunks so the engine and the
tileset have something to render before Fase 2. Everything is deterministic for a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tiles
from .chunk import Chunk, ChunkObject, OBJECT_FLAG_FLIP_H
from .level import LevelManifest


@dataclass
class SyntheticLevel:
    """Full-grid arrays before chunking. Indexed ``[y, x]``."""

    width: int
    height: int
    terrain: np.ndarray
    overlay: np.ndarray
    objects: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, tile, flags)
    elevation: np.ndarray | None = None


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def value_noise(rng: np.random.Generator, height: int, width: int, cell: int) -> np.ndarray:
    """One octave of value noise: random lattice every ``cell`` pixels, smooth bilinear blend."""
    gh, gw = height // cell + 2, width // cell + 2
    lattice = rng.random((gh, gw))
    ys = np.arange(height) / cell
    xs = np.arange(width) / cell
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    ty = _smoothstep(ys - y0)[:, None]
    tx = _smoothstep(xs - x0)[None, :]
    a = lattice[y0[:, None], x0[None, :]]
    b = lattice[y0[:, None], x0[None, :] + 1]
    c = lattice[y0[:, None] + 1, x0[None, :]]
    d = lattice[y0[:, None] + 1, x0[None, :] + 1]
    return _lerp(_lerp(a, b, tx), _lerp(c, d, tx), ty)


def fbm(rng: np.random.Generator, height: int, width: int, base_cell: int, octaves: int = 4) -> np.ndarray:
    """Fractal sum of value-noise octaves, normalized to [0, 1]."""
    out = np.zeros((height, width))
    amp, total, cell = 1.0, 0.0, base_cell
    for _ in range(octaves):
        out += amp * value_noise(rng, height, width, max(cell, 2))
        total += amp
        amp *= 0.5
        cell //= 2
    return out / total


def _line_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham line, inclusive of both ends."""
    cells = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


def _river_path(
    potential: np.ndarray, land: np.ndarray, start: tuple[int, int], max_steps: int
) -> list[tuple[int, int]]:
    """Greedy descent of ``potential`` from ``start`` until the sea; never revisits a cell."""
    h, w = potential.shape
    y, x = start
    path = [(x, y)]
    visited = {(x, y)}
    for _ in range(max_steps):
        if not land[y, x]:
            break
        best = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in visited:
                    cand = (potential[ny, nx], nx, ny)
                    if best is None or cand < best:
                        best = cand
        if best is None:
            break
        _, x, y = best
        visited.add((x, y))
        path.append((x, y))
    return path


def generate(width: int, height: int, seed: int = 42) -> SyntheticLevel:
    rng = np.random.default_rng(seed)
    h, w = height, width

    # --- elevation: fbm minus a radial falloff gives one blobby continent ------------------
    yy, xx = np.mgrid[0:h, 0:w]
    ny = (yy + 0.5) / h * 2.0 - 1.0
    nx = (xx + 0.5) / w * 2.0 - 1.0
    radial = np.sqrt(nx**2 + ny**2)
    base = fbm(rng, h, w, base_cell=max(w, h) // 4, octaves=5)
    elev = base * 1.1 - radial * 0.85 + 0.15
    land = elev > 0.0

    moisture = fbm(rng, h, w, base_cell=max(w, h) // 6, octaves=3)
    lakes = fbm(rng, h, w, base_cell=max(w, h) // 10, octaves=2)

    terrain = np.full((h, w), tiles.OCEANO, dtype=np.uint16)
    terrain[land] = tiles.CAMPO
    terrain[land & (moisture > 0.55)] = tiles.FLORESTA
    terrain[land & (moisture < 0.33) & (elev > 0.1)] = tiles.SAVANA
    terrain[land & (elev > 0.5)] = tiles.MONTANHA
    inland = land & (elev > 0.08)
    terrain[inland & (lakes < 0.22) & (elev < 0.5)] = tiles.AGUA

    # Coast: land cells with at least one ocean cell in the 4-neighbourhood become sand.
    ocean = terrain == tiles.OCEANO
    near_ocean = np.zeros_like(ocean)
    near_ocean[1:, :] |= ocean[:-1, :]
    near_ocean[:-1, :] |= ocean[1:, :]
    near_ocean[:, 1:] |= ocean[:, :-1]
    near_ocean[:, :-1] |= ocean[:, 1:]
    terrain[land & near_ocean & (terrain != tiles.AGUA)] = tiles.AREIA

    # --- urban patch: a small blob on a flat inland campo/floresta area ---------------------
    flat = inland & (elev < 0.3) & np.isin(terrain, [tiles.CAMPO, tiles.FLORESTA, tiles.SAVANA])
    flat_idx = np.flatnonzero(flat)
    urban_center: tuple[int, int] | None = None
    if flat_idx.size:
        # pick the flat cell closest to the continent's centroid so the city is well inland
        cy_c, cx_c = np.argwhere(land).mean(axis=0)
        fy, fx = np.unravel_index(flat_idx, (h, w))
        k = int(np.argmin((fy - cy_c) ** 2 + (fx - cx_c) ** 2))
        uy, ux = int(fy[k]), int(fx[k])
        urban_center = (ux, uy)
        r = max(3, min(w, h) // 40)
        blob = ((yy - uy) ** 2 + (xx - ux) ** 2 <= r**2) & flat
        terrain[blob] = tiles.URBANO

    overlay = np.zeros((h, w), dtype=np.uint16)
    objects: list[tuple[int, int, int, int]] = []
    land_idx = np.flatnonzero(land & (terrain != tiles.AGUA))

    # --- river: from the highest point, flow downhill to the sea ---------------------------
    # Greedy descent on raw elevation gets trapped in basins, so the potential subtracts the
    # radial distance (which grows monotonically towards the coast): the river follows the
    # relief where it can but always drains outward.
    if land.any():
        peak = np.unravel_index(int(np.argmax(np.where(land, elev, -np.inf))), (h, w))
        potential = elev - radial
        for x, y in _river_path(potential, land, (int(peak[0]), int(peak[1])), max_steps=4 * (w + h)):
            if land[y, x] and terrain[y, x] not in (tiles.OCEANO,):
                overlay[y, x] = tiles.RIO

    # --- roads: two lines across the land, through the city if there is one ----------------
    if land_idx.size >= 2:
        def pick() -> tuple[int, int]:
            i = int(rng.choice(land_idx))
            y, x = divmod(i, w)
            return x, y

        hub = urban_center or pick()
        for _ in range(2):
            x1, y1 = pick()
            for x, y in _line_cells(hub[0], hub[1], x1, y1):
                if land[y, x] and overlay[y, x] == 0:
                    overlay[y, x] = tiles.RODOVIA

    # --- objects: trees on forest, buildings/houses in the city ----------------------------
    forest_cells = np.argwhere(terrain == tiles.FLORESTA)
    keep = rng.random(len(forest_cells)) < 0.08
    for y, x in forest_cells[keep]:
        flags = OBJECT_FLAG_FLIP_H if rng.random() < 0.5 else 0
        objects.append((int(x), int(y), tiles.ARVORE, flags))
    urban_cells = np.argwhere(terrain == tiles.URBANO)
    roll = rng.random(len(urban_cells))
    for (y, x), rv in zip(urban_cells, roll):
        if rv < 0.25:
            objects.append((int(x), int(y), tiles.PREDIO, 0))
        elif rv < 0.55:
            objects.append((int(x), int(y), tiles.CASA, 0))

    return SyntheticLevel(w, h, terrain, overlay, objects, elevation=elev)


def chunkify(level: SyntheticLevel, manifest: LevelManifest) -> list[Chunk]:
    """Split full-grid arrays into chunks. Trailing partial chunks are zero-padded."""
    cs = manifest.chunk_size
    H = manifest.chunks_y * cs
    W = manifest.chunks_x * cs
    terrain = np.zeros((H, W), dtype=np.uint16)
    overlay = np.zeros((H, W), dtype=np.uint16)
    terrain[: level.height, : level.width] = level.terrain
    overlay[: level.height, : level.width] = level.overlay

    per_chunk: dict[tuple[int, int], list[ChunkObject]] = {}
    for x, y, tile, flags in level.objects:
        key = (x // cs, y // cs)
        per_chunk.setdefault(key, []).append(ChunkObject(x % cs, y % cs, tile, flags))

    ps = pe = None
    if manifest.period_start is not None and manifest.period_end is not None:
        from .level import parse_period_date

        ps = parse_period_date(manifest.period_start, end=False)
        pe = parse_period_date(manifest.period_end, end=True)

    chunks = []
    for cx, cy in manifest.chunk_coords():
        sl = (slice(cy * cs, (cy + 1) * cs), slice(cx * cs, (cx + 1) * cs))
        chunks.append(
            Chunk(
                cx=cx,
                cy=cy,
                size=cs,
                terrain=terrain[sl].copy(),
                overlay=overlay[sl].copy(),
                objects=per_chunk.get((cx, cy), []),
                period_start=ps,
                period_end=pe,
            )
        )
    return chunks
