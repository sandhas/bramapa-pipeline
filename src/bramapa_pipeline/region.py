"""Pieces shared by the regional levels (2 = state, 3 = municipality).

* grid fitting: centre and cell size / grid size so a region fits the rotated grid;
* OSM overlay rasterisation and object placement rules (buildings by distance to a centre);
* reading the IBGE meshes for one UF.

Everything geometric happens in the grid CRS (``EPSG:5880``); see ``level1.py`` for the rotated
affine. A grid of ``size`` cells of ``c`` metres covers, on the ground, the diamond
``|E - ce| + |N - cn| <= size * c / sqrt2`` around its centre ``(ce, cn)``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np

from . import level1 as L
from .level1 import SQRT2, GridSpec

log = logging.getLogger(__name__)

MIN_CELL_METERS = 700.0
"""Floor for the automatic level 2 cell size (a smaller state would otherwise get sub-km cells
that the 30 m MapBiomas overviews cannot honour without a native read)."""
FIT_MARGIN = 1.04
"""Breathing room around the region, as a factor on the required half-diagonal."""
MAX_LEVEL3_SIZE = 2048


def _coords(geom) -> np.ndarray:
    """``(n, 2)`` array of every vertex of a (multi)polygon/line in the grid CRS."""
    from shapely import get_coordinates

    return np.asarray(get_coordinates(geom), dtype=float)


def rotated_extent(geom) -> tuple[float, float, float]:
    """``(ce, cn, r)``: the centre and half-diagonal of the tightest 45°-diamond around ``geom``.

    In the rotated frame ``u = E + N``, ``v = E - N`` the diamond is an axis-aligned square, so the
    tightest one is centred on the midpoint of the ``u`` and ``v`` ranges with half-side
    ``r = max(range_u, range_v) / 2`` (and ``|dE| + |dN| = max(|du|, |dv|)``).
    """
    c = _coords(geom)
    u = c[:, 0] + c[:, 1]
    v = c[:, 0] - c[:, 1]
    cu = (u.min() + u.max()) / 2
    cv = (v.min() + v.max()) / 2
    r = max(u.max() - u.min(), v.max() - v.min()) / 2
    return (cu + cv) / 2, (cu - cv) / 2, float(r)


def fit_cell_meters(geom, size: int, *, margin: float = FIT_MARGIN, min_cell: float = MIN_CELL_METERS) -> float:
    """Smallest cell size (m) so ``geom`` fits a ``size``-cell rotated grid, rounded up to 10 m."""
    _ce, _cn, r = rotated_extent(geom)
    c = r * margin * SQRT2 / size
    return max(min_cell, math.ceil(c / 10.0) * 10.0)


def fit_size(geom, cell_meters: float, *, margin: float = FIT_MARGIN, multiple: int = 64, cap: int = MAX_LEVEL3_SIZE) -> tuple[int, float, bool]:
    """``(size, cell_meters, capped)``: grid side (multiple of 64) so ``geom`` fits at ``cell_meters``.

    When even ``cap`` cells cannot hold the region at that cell size, the cell size grows to make
    it fit in ``cap`` cells and ``capped`` is ``True``.
    """
    _ce, _cn, r = rotated_extent(geom)
    need = r * margin * SQRT2 / cell_meters
    size = max(multiple, int(math.ceil(need / multiple)) * multiple)
    if size > cap:
        c = math.ceil(r * margin * SQRT2 / cap / 10.0) * 10.0
        return cap, float(c), True
    return size, float(cell_meters), False


def grid_for(geom, *, size: int, cell_meters: float, crs: str = L.GRID_CRS) -> GridSpec:
    """Grid of ``size`` cells of ``cell_meters`` centred on the diamond centre of ``geom``."""
    ce, cn, _r = rotated_extent(geom)
    return GridSpec.from_projected_center(ce, cn, size=size, cell_meters=cell_meters, crs=crs)


# ----------------------------------------------------------------------------- object rules

PREDIO_PERMILLE_NEAR = 250  # within `NEAR_M` of the historic centre
PREDIO_PERMILLE_MID = 80  # at `MID_M`
PREDIO_PERMILLE_FAR = 20  # beyond `FAR_M`
NEAR_M, MID_M, FAR_M = 2_000.0, 5_000.0, 10_000.0
CASA_PERMILLE = 200
TREE_PERMILLE_LEVEL3 = 100
LEVEL2_TREE_PERMILLE = 30
LEVEL2_PREDIO_PERMILLE = 60
LEVEL2_CASA_PERMILLE = 120


def predio_permille(distance_m: np.ndarray) -> np.ndarray:
    """Share (per mille) of urban cells that get a ``predio``, non-increasing with distance.

    ~25% within 2 km of the centre, falling linearly to ~8% at 5 km, then to ~2% at 10 km and
    beyond.
    """
    d = np.asarray(distance_m, dtype=float)
    out = np.empty_like(d)
    out[:] = PREDIO_PERMILLE_FAR
    mid = (d > NEAR_M) & (d <= MID_M)
    far = (d > MID_M) & (d < FAR_M)
    out[d <= NEAR_M] = PREDIO_PERMILLE_NEAR
    out[mid] = PREDIO_PERMILLE_NEAR + (PREDIO_PERMILLE_MID - PREDIO_PERMILLE_NEAR) * (d[mid] - NEAR_M) / (MID_M - NEAR_M)
    out[far] = PREDIO_PERMILLE_MID + (PREDIO_PERMILLE_FAR - PREDIO_PERMILLE_MID) * (d[far] - MID_M) / (FAR_M - MID_M)
    return np.rint(out).astype(np.uint32)


def distance_grid(spec: GridSpec, lon: float, lat: float) -> np.ndarray:
    """Ground distance (m) from every cell centre to a lon/lat point."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", spec.crs, always_xy=True)
    pe, pn = t.transform(lon, lat)
    yy, xx = np.mgrid[0 : spec.size, 0 : spec.size].astype(float)
    es, ns = spec.transform @ (xx + 0.5, yy + 0.5)
    return np.hypot(es - pe, ns - pn)


def city_objects(terrain: np.ndarray, distance_m: np.ndarray | None, *, tree_permille: int, casa_permille: int, predio: int | np.ndarray) -> list[tuple[int, int, str, int]]:
    """``predio``/``casa`` on ``urbano`` (predio first, casa on the rest), ``arvore`` on ``floresta``."""
    urban = terrain == L.URBANO
    perm = predio_permille(distance_m) if distance_m is not None else predio
    keep_p, flip_p = L.sparse_mask(urban, perm, salt=1)
    keep_c, flip_c = L.sparse_mask(urban & ~keep_p, casa_permille, salt=2)
    keep_t, flip_t = L.sparse_mask(terrain == L.FLORESTA, tree_permille, salt=0)
    return L.objects_from_mask(keep_t, flip_t, "arvore") + L.objects_from_mask(keep_p, flip_p, "predio") + L.objects_from_mask(keep_c, flip_c, "casa")


# ----------------------------------------------------------------------------- overlays


def paint_lines(overlay: np.ndarray, gdf, spec: GridSpec, where: np.ndarray, codes: dict[str, int]) -> None:
    """Burn a GeoDataFrame with an ``overlay`` column (cover names) in ascending code priority."""
    if gdf is None or len(gdf) == 0:
        return
    for name, code in sorted(codes.items(), key=lambda kv: kv[1]):
        sel = gdf[gdf["overlay"] == name]
        if len(sel) == 0:
            continue
        m = L.rasterize_mask(sel.geometry, spec, all_touched=True) & where
        overlay[m & (overlay < code)] = code


OSM_OVERLAY_CODES = {"rio": L.RIO, "ferrovia": L.FERROVIA, "rodovia": L.RODOVIA}


def osm_lines(bbox, selectors, cache_path: Path, road_classes, spec: GridSpec):
    """Fetch (or load cached) Overpass ways for ``bbox`` and return them as a projected GeoDataFrame.

    Returns ``None`` when Overpass is unreachable so the caller can fall back.
    """
    from . import osm

    try:
        doc = osm.fetch_overpass(osm.build_query(bbox, selectors), cache_path)
    except osm.OverpassError as e:
        log.warning("OSM unavailable: %s", e)
        return None
    gdf = osm.lines_to_gdf(osm.parse_lines(doc), road_classes)
    if len(gdf) == 0:
        return gdf
    return gdf.to_crs(spec.crs)


def ne_roads_fallback(spec: GridSpec, cache_dir: Path, types=("Major Highway", "Beltway", "Secondary Highway")):
    """Natural Earth roads as ``rodovia`` when Overpass fails (no railways, no rivers)."""
    from . import sources

    p = sources.path_for("ne_10m_roads", cache_dir)
    if not p.exists():
        log.warning("no OSM and no cached ne_10m_roads: level without roads")
        return None
    gdf = L._read_vector(p, ("type",), spec.lonlat_bounds(0.2))
    rt = L._col(gdf, "type")
    gdf = gdf[rt.isin(types)] if rt is not None else gdf
    gdf = gdf[["geometry"]].copy()
    gdf["overlay"] = "rodovia"
    return L._project(gdf, spec, None)


# ----------------------------------------------------------------------------- IBGE meshes


def read_uf_geometries(cache_dir: Path, bbox=None):
    """UF polygons (``entity_id``, ``entity_name``, ``cd_uf``) from the cached BR_UF_2022 in EPSG:4326."""
    from . import sources

    p = sources.path_for("ibge_uf_2022", cache_dir)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `bramapa-pipeline fetch` first")
    gdf = L._read_vector(p, ("CD_UF", "NM_UF", "SIGLA_UF"), bbox)
    out = L.prepare_uf(gdf)
    out["cd_uf"] = [int(v) for v in L._col(gdf, "CD_UF")]
    return out.to_crs("EPSG:4326")


def read_municipios(uf_code: int, cache_dir: Path, bbox=None, *, timeout: float = 300.0):
    """Municipalities of one UF (``entity_id``, ``entity_name``, ``cd_mun``, ``pseudo``) in EPSG:4326, fetching the mesh if needed."""
    from . import sources
    from .level import entity_id, is_pseudo_municipio

    src = sources.municipios_source(uf_code)
    t0 = time.perf_counter()
    p = sources.fetch(src, cache_dir, timeout=timeout)
    gdf = L._read_vector(p, src.fields, bbox, src.layer)
    cd = [int(v) for v in L._col(gdf, "CD_MUN")]
    nm = [str(v) for v in L._col(gdf, "NM_MUN")]
    out = gdf[["geometry"]].copy()
    out["cd_mun"] = cd
    out["pseudo"] = [is_pseudo_municipio(c) for c in cd]
    out["entity_id"] = [entity_id("municipio", c) if not ps else f"agua:{c}" for c, ps in zip(cd, out["pseudo"])]
    out["entity_name"] = nm
    log.info("%s: %d features (%d pseudo) in %.1fs", src.key, len(out), int(out["pseudo"].sum()), time.perf_counter() - t0)
    return out.to_crs("EPSG:4326")


__all__ = [
    "FIT_MARGIN",
    "MAX_LEVEL3_SIZE",
    "MIN_CELL_METERS",
    "OSM_OVERLAY_CODES",
    "city_objects",
    "distance_grid",
    "fit_cell_meters",
    "fit_size",
    "grid_for",
    "ne_roads_fallback",
    "osm_lines",
    "paint_lines",
    "predio_permille",
    "read_municipios",
    "read_uf_geometries",
    "rotated_extent",
]
