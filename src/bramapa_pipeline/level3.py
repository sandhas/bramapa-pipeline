"""Level 3: one municipality (``municipio:NNNNNNN``) at 30 m from MapBiomas, IBGE and OSM.

Grid side is the smallest multiple of 64 that holds the municipality's rotated bounding diamond
at ``cell_meters`` (default 30 m, the MapBiomas resolution), capped at 2048 cells: beyond that the
cell size grows and the build warns. Cells outside the municipality are still painted from
MapBiomas (the surroundings exist) but carry entity 0; inside is entity 1.

Overlays from OpenStreetMap (Overpass): ``highway`` motorway/trunk/primary/secondary → ``rodovia``
(plus tertiary/residential when the bbox has fewer than ``MINOR_ROADS_MAX_WAYS`` of them),
``railway=rail`` → ``ferrovia``, ``waterway=river`` → ``rio``; ``limite_municipio`` on the outline.
Objects: ``predio`` on ``urbano`` with density falling with the distance to the historic centre,
``casa`` on the remaining ``urbano``, ``arvore`` on ``floresta``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import level1 as L
from . import region as R
from .level import entity_id, level_dir, municipio_uf
from .level1 import GridSpec, Level1Grid
from .mapbiomas import Coverage, classify, read_coverage, warp_classes

log = logging.getLogger(__name__)

LONLAT_MARGIN_DEG = 0.02

# Historic centres (lon, lat) used for the building-density falloff; municipalities not listed
# use the centroid of their urban cells.
HISTORIC_CENTRES: dict[int, tuple[float, float]] = {
    4314902: (-51.2300, -30.0330),  # Porto Alegre, Praça da Matriz / Centro Histórico
}


@dataclass
class Level3Inputs:
    cd_mun: int
    name: str
    classes: np.ndarray  # (size, size) uint8 MapBiomas classes
    municipio: object | None = None  # one-row GeoDataFrame in the grid CRS
    land: object | None = None  # GeoDataFrame of land polygons (all municipalities of the UF in bbox + NE countries)
    lines: object | None = None  # overlay column: rodovia / ferrovia / rio
    centre: tuple[float, float] | None = None  # lon, lat
    roads_source: str = "none"
    minor_roads: str = "skipped"
    coverage: Coverage | None = None
    notes: list[str] = field(default_factory=list)


def load_municipio(cd_mun: int, cache_dir: Path):
    """``(geometry in grid CRS, NM_MUN, municipalities GeoDataFrame of the UF in EPSG:4326)``."""
    muns = R.read_municipios(municipio_uf(cd_mun), cache_dir)
    row = muns[muns["cd_mun"] == int(cd_mun)]
    if len(row) == 0:
        raise ValueError(f"municipality {cd_mun} not in the IBGE mesh of UF {municipio_uf(cd_mun)}")
    return row.to_crs(L.GRID_CRS).geometry.iloc[0], str(row["entity_name"].iloc[0]), muns


def make_spec(mun_geom, *, cell_meters: float = 30.0, size: int | None = None) -> tuple[GridSpec, bool]:
    """Grid for the municipality: ``(spec, capped)``; ``size`` forces the side (multiple of 64)."""
    if size is None:
        size, cell, capped = R.fit_size(mun_geom, cell_meters)
    else:
        size = max(64, int(np.ceil(size / 64)) * 64)
        _s, cell, capped = R.fit_size(mun_geom, cell_meters, cap=size)
        cell = max(cell, float(cell_meters))
    if capped:
        log.warning("municipality does not fit %d cells at %g m: cell size raised to %g m", size, cell_meters, cell)
    return R.grid_for(mun_geom, size=size, cell_meters=cell), capped


def load_inputs(
    spec: GridSpec, cd_mun: int, name: str, muns, cache_dir: Path, *, use_osm: bool = True, centre: tuple[float, float] | None = None
) -> Level3Inputs:
    from . import osm, sources

    bbox = spec.lonlat_bounds(LONLAT_MARGIN_DEG)
    eid = entity_id("municipio", cd_mun)
    tag = eid.replace(":", "_")
    t0 = time.perf_counter()
    cov = read_coverage(bbox, spec.cell_meters, Path(cache_dir) / f"mapbiomas_{tag}.tif")
    classes = warp_classes(cov, spec)
    inp = Level3Inputs(cd_mun=int(cd_mun), name=name, classes=classes, coverage=cov)
    inp.centre = centre or HISTORIC_CENTRES.get(int(cd_mun))

    muns_p = muns.to_crs(spec.crs)
    inp.municipio = muns_p[muns_p["cd_mun"] == int(cd_mun)]
    land = [muns_p[["geometry"]]]
    p = sources.path_for("ne_10m_admin_0_countries", cache_dir)
    if p.exists():
        countries = L._read_vector(p, (), bbox)
        if len(countries):
            land.append(L._project(countries[["geometry"]], spec, None))
    import pandas as pd

    inp.land = pd.concat(land, ignore_index=True)

    lines = None
    if use_osm:
        road_classes = osm.ROAD_CLASSES_LEVEL3
        selectors = [osm.highway_selector(osm.ROAD_CLASSES_LEVEL3), osm.RAIL_SELECTOR, osm.RIVER_SELECTOR]
        minor_cache = Path(cache_dir) / f"osm_{tag}_minor_roads.json"
        n_minor = -2 if minor_cache.exists() else osm.count_ways(bbox, (osm.highway_selector(osm.ROAD_CLASSES_MINOR),))
        if n_minor == -2 or 0 <= n_minor < osm.MINOR_ROADS_MAX_WAYS:
            minor = R.osm_lines(bbox, (osm.highway_selector(osm.ROAD_CLASSES_MINOR),), minor_cache, osm.ROAD_CLASSES_MINOR, spec)
            if minor is not None:
                inp.minor_roads = f"included ({len(minor)} ways)"
                road_classes = osm.ROAD_CLASSES_LEVEL3 + osm.ROAD_CLASSES_MINOR
        else:
            inp.minor_roads = f"skipped ({n_minor} ways, limit {osm.MINOR_ROADS_MAX_WAYS})" if n_minor >= 0 else "skipped (count unavailable)"
            minor = None
        main = R.osm_lines(bbox, tuple(selectors), Path(cache_dir) / f"osm_{tag}_roads_rail_rivers.json", osm.ROAD_CLASSES_LEVEL3, spec)
        if main is not None:
            lines = main if minor is None or len(minor) == 0 else pd.concat([main, minor], ignore_index=True)
            lines = lines.set_crs(spec.crs, allow_override=True)
    if lines is not None:
        inp.lines, inp.roads_source = lines, "osm"
    else:
        inp.lines = R.ne_roads_fallback(spec, cache_dir)
        inp.roads_source = "natural earth (fallback: Overpass unavailable)" if use_osm else "natural earth (--no-osm)"
        inp.notes.append("rodovias do Natural Earth (Overpass indisponível); sem ferrovia e sem rio do OSM")
    log.info("level 3 inputs loaded in %.1fs (roads: %s, minor roads: %s)", time.perf_counter() - t0, inp.roads_source, inp.minor_roads)
    return inp


def urban_centroid(spec: GridSpec, terrain: np.ndarray, inside: np.ndarray) -> tuple[float, float] | None:
    """Lon/lat of the mean urban cell inside the municipality (fallback centre)."""
    from pyproj import Transformer

    ys, xs = np.nonzero((terrain == L.URBANO) & inside)
    if xs.size == 0:
        return None
    e, n = spec.transform @ (xs.mean() + 0.5, ys.mean() + 0.5)
    lon, lat = Transformer.from_crs(spec.crs, "EPSG:4326", always_xy=True).transform(e, n)
    return float(lon), float(lat)


def build_grid(spec: GridSpec, inp: Level3Inputs) -> Level1Grid:
    size = spec.size
    t0 = time.perf_counter()
    inside = L.rasterize_mask(inp.municipio.geometry, spec) if inp.municipio is not None and len(inp.municipio) else np.zeros((size, size), dtype=bool)
    land = L.rasterize_mask(inp.land.geometry, spec, all_touched=True) if inp.land is not None and len(inp.land) else inside.copy()
    land |= inside

    entity = inside.astype(np.uint16)
    entities = [(1, entity_id("municipio", inp.cd_mun), inp.name)]

    terrain = classify(inp.classes, land)
    L.beach(terrain)

    overlay = np.zeros((size, size), dtype=np.uint8)
    on_land = terrain != L.OCEANO
    R.paint_lines(overlay, inp.lines, spec, on_land, R.OSM_OVERLAY_CODES)
    overlay[L.mask_outline(inside)] = L.LIMITE_MUNICIPIO

    centre = inp.centre or urban_centroid(spec, terrain, inside)
    dist = R.distance_grid(spec, *centre) if centre is not None else None
    objects = R.city_objects(terrain, dist, tree_permille=R.TREE_PERMILLE_LEVEL3, casa_permille=R.CASA_PERMILLE, predio=R.PREDIO_PERMILLE_FAR)
    log.info(
        "level 3 grid %dx%d built in %.1fs: %d cells inside, %d objects, centre %s",
        size, size, time.perf_counter() - t0, int(inside.sum()), len(objects), centre,
    )
    return Level1Grid(terrain, overlay, entity, entities, objects)


def manifest_for(spec: GridSpec, inp: Level3Inputs, grid: Level1Grid, tileset):
    year = inp.coverage.info.year if inp.coverage is not None and inp.coverage.info.year else None
    return spec.manifest(
        tileset, grid.entities, level=3, entity=entity_id("municipio", inp.cd_mun), name=inp.name,
        period=str(year) if year else None,
    )


def output_dir(root: Path, cd_mun: int) -> Path:
    return level_dir(root, 3, entity_id("municipio", cd_mun))


__all__ = ["HISTORIC_CENTRES", "Level3Inputs", "build_grid", "load_inputs", "load_municipio", "make_spec", "manifest_for", "output_dir", "urban_centroid"]
