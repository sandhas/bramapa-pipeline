"""Level 2: one state (``estado:NN``) at ~1 km from MapBiomas, the IBGE municipal mesh and OSM.

Grid: ``size`` cells (default 1024) centred on the state's rotated bounding diamond, cell size
chosen automatically so the whole state fits (``region.fit_cell_meters``, floor 700 m). Sources:

* **MapBiomas** coverage (decimated through the COG overviews, mode-resampled per cell) for the
  terrain, with the legend table in ``mapbiomas.py``; outside Brazil everything is ``campo``;
* **IBGE** ``<UF>_Municipios_2022`` for ``municipio:<CD_MUN>`` entities and ``limite_municipio``,
  ``BR_UF_2022`` for the neighbouring states (``estado:NN``, ``limite_estado``);
* **Natural Earth** countries (``pais:XX`` + land mask), lakes outside Brazil, rivers (scalerank ≤ 8),
  land boundaries (``limite_pais``);
* **OpenStreetMap** via Overpass: ``highway`` motorway/trunk/primary → ``rodovia``, ``railway=rail``
  → ``ferrovia``; Natural Earth roads when Overpass is down.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import level1 as L
from . import region as R
from .level import entity_id, level_dir
from .level1 import GridSpec, Level1Grid
from .mapbiomas import Coverage, classify, read_coverage, warp_classes

log = logging.getLogger(__name__)

RIVER_MAX_SCALERANK = 8
LONLAT_MARGIN_DEG = 0.1


@dataclass
class Level2Inputs:
    """Projected (grid CRS) layers plus the MapBiomas classes already on the grid."""

    uf_code: int
    uf_name: str
    classes: np.ndarray  # (size, size) uint8 MapBiomas classes, 0 = nodata
    municipios: object | None = None  # entity_id, entity_name, pseudo
    ufs: object | None = None  # entity_id, entity_name, cd_uf (every UF in the bbox, target included)
    countries: object | None = None  # entity_id, entity_name
    lakes: object | None = None
    rivers: object | None = None
    boundaries: object | None = None
    lines: object | None = None  # overlay column: rodovia / ferrovia / rio
    roads_source: str = "none"
    coverage: Coverage | None = None
    notes: list[str] = field(default_factory=list)


def load_uf(uf_code: int, cache_dir: Path):
    """``(geometry in grid CRS, NM_UF)`` of one state from the cached BR_UF_2022."""
    ufs = R.read_uf_geometries(cache_dir)
    row = ufs[ufs["cd_uf"] == int(uf_code)]
    if len(row) == 0:
        raise ValueError(f"UF {uf_code} not in BR_UF_2022")
    geom = row.to_crs(L.GRID_CRS).geometry.iloc[0]
    return geom, str(row["entity_name"].iloc[0])


def make_spec(uf_geom, *, size: int = 1024, cell_meters: float | None = None) -> tuple[GridSpec, bool]:
    """Grid for the state: ``(spec, auto)`` where ``auto`` says the cell size was fitted."""
    auto = cell_meters is None
    c = R.fit_cell_meters(uf_geom, size) if auto else float(cell_meters)
    return R.grid_for(uf_geom, size=size, cell_meters=c), auto


def load_inputs(spec: GridSpec, uf_code: int, uf_name: str, cache_dir: Path, *, use_osm: bool = True) -> Level2Inputs:
    from . import osm, sources

    bbox = spec.lonlat_bounds(LONLAT_MARGIN_DEG)
    simp = spec.cell_meters / 10.0 / 111_000.0
    eid = entity_id("estado", uf_code)

    def read(key: str, prep=None, simplify=simp):
        p = sources.path_for(key, cache_dir)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run `bramapa-pipeline fetch` first")
        gdf = L._read_vector(p, sources.SOURCES[key].fields, bbox, sources.SOURCES[key].layer)
        if prep is not None:
            gdf = prep(gdf)
        return L._project(gdf, spec, simplify)

    t0 = time.perf_counter()
    cov = read_coverage(bbox, spec.cell_meters, Path(cache_dir) / f"mapbiomas_{eid.replace(':', '_')}.tif")
    classes = warp_classes(cov, spec)
    inp = Level2Inputs(uf_code=int(uf_code), uf_name=uf_name, classes=classes, coverage=cov)

    inp.municipios = L._project(R.read_municipios(uf_code, cache_dir, bbox), spec, simp)
    inp.ufs = L._project(R.read_uf_geometries(cache_dir, bbox), spec, simp)
    inp.countries = read("ne_10m_admin_0_countries", L.prepare_countries)
    lakes = read("ne_10m_lakes", simplify=None)
    inp.lakes = lakes[lakes.geometry.area >= 2 * spec.cell_meters**2]
    rivers = read("ne_10m_rivers_lake_centerlines", simplify=None)
    sr = L._col(rivers, "scalerank")
    inp.rivers = rivers[sr.astype(float) <= RIVER_MAX_SCALERANK] if sr is not None else rivers
    inp.boundaries = read("ne_10m_admin_0_boundary_lines_land", simplify=None)

    lines = None
    if use_osm:
        selectors = (osm.highway_selector(osm.ROAD_CLASSES_LEVEL2), osm.RAIL_SELECTOR)
        lines = R.osm_lines(bbox, selectors, Path(cache_dir) / f"osm_{eid.replace(':', '_')}_roads_rail.json", osm.ROAD_CLASSES_LEVEL2, spec)
    if lines is not None:
        inp.lines, inp.roads_source = lines, "osm"
    else:
        inp.lines = R.ne_roads_fallback(spec, cache_dir)
        inp.roads_source = "natural earth (fallback: Overpass unavailable)" if use_osm else "natural earth (--no-osm)"
        inp.notes.append("rodovias do Natural Earth (Overpass indisponível); sem ferrovia")
    log.info("level 2 inputs loaded in %.1fs (roads: %s)", time.perf_counter() - t0, inp.roads_source)
    return inp


def build_grid(spec: GridSpec, inp: Level2Inputs) -> Level1Grid:
    """Cover/overlay/entity grids for the state. No I/O."""
    size = spec.size
    t0 = time.perf_counter()
    labels: dict[int, tuple[str, str]] = {}
    target_eid = entity_id("estado", inp.uf_code)

    # --- entities: countries < other states < municipalities of the target state -------------
    raw = np.zeros((size, size), dtype=np.uint16)
    if inp.countries is not None and len(inp.countries):
        c = inp.countries
        for k, (eid, name) in enumerate(zip(c["entity_id"], c["entity_name"]), start=1):
            labels[k] = (str(eid), str(name))
        raw = L.rasterize_values(c.geometry, np.arange(1, len(c) + 1), spec)
    ne_brazil = np.isin(raw, [k for k, (eid, _n) in labels.items() if eid == "pais:BR"])

    uf_index = np.zeros((size, size), dtype=np.uint16)
    target_uf_code = 0
    if inp.ufs is not None and len(inp.ufs):
        u = inp.ufs
        codes = np.arange(1001, 1001 + len(u))
        for k, eid, name, cd in zip(codes, u["entity_id"], u["entity_name"], u["cd_uf"]):
            labels[int(k)] = (str(eid), str(name))
            if int(cd) == inp.uf_code:
                target_uf_code = int(k)
        uf_center = L.rasterize_values(u.geometry, codes, spec)
        uf_touch = L.rasterize_values(u.geometry, codes, spec, all_touched=True)
        uf_index = np.where(uf_center != 0, uf_center, np.where(ne_brazil, uf_touch, 0)).astype(np.uint16)
        uf_index = L.fill_from_neighbours(uf_index, ne_brazil & (uf_index == 0))
    in_uf = uf_index == target_uf_code if target_uf_code else np.zeros((size, size), dtype=bool)

    mun_index = np.zeros((size, size), dtype=np.uint16)
    n_mun = 0
    if inp.municipios is not None and len(inp.municipios):
        m = inp.municipios
        codes = np.arange(2001, 2001 + len(m))
        for k, eid, name, pseudo in zip(codes, m["entity_id"], m["entity_name"], m["pseudo"]):
            # Lagoons the mesh carries as pseudo-municipalities belong to the state itself.
            labels[int(k)] = (target_eid, inp.uf_name) if pseudo else (str(eid), str(name))
        mun_center = L.rasterize_values(m.geometry, codes, spec)
        mun_touch = L.rasterize_values(m.geometry, codes, spec, all_touched=True)
        mun_index = np.where(mun_center != 0, mun_center, np.where(in_uf, mun_touch, 0)).astype(np.uint16)
        mun_index = L.fill_from_neighbours(mun_index, in_uf & (mun_index == 0))
        # Municipalities smaller than a cell can still lose every cell: give each its centre cell.
        present = set(np.unique(mun_index).tolist())
        inv = ~spec.transform
        for k, geom, pseudo in zip(codes, m.geometry, m["pseudo"]):
            if int(k) in present or pseudo:
                continue
            pt = geom.representative_point()
            fx, fy = inv @ (pt.x, pt.y)
            x, y = int(fx), int(fy)
            if 0 <= x < size and 0 <= y < size:
                mun_index[y, x] = k
        n_mun = int((~m["pseudo"]).sum())
    raw = np.where(mun_index != 0, mun_index, np.where(uf_index != 0, uf_index, raw)).astype(np.uint16)
    land = raw != 0
    entity, entities = L.index_entities(raw, labels)
    n_mun_entities = sum(1 for _i, eid, _n in entities if eid.startswith("municipio:"))
    if n_mun_entities < n_mun:
        log.warning("%d of %d municipalities got no cell", n_mun - n_mun_entities, n_mun)

    # --- terrain ------------------------------------------------------------------------------
    terrain = classify(inp.classes, land)
    if inp.lakes is not None and len(inp.lakes):
        # MapBiomas already has Brazilian water; Natural Earth fills the lakes of the neighbours.
        lake = L.rasterize_mask(inp.lakes.geometry, spec) & land & (inp.classes == 0)
        terrain[lake] = L.AGUA
    L.beach(terrain)

    # --- overlay ------------------------------------------------------------------------------
    overlay = np.zeros((size, size), dtype=np.uint8)
    on_land = terrain != L.OCEANO
    if inp.rivers is not None and len(inp.rivers):
        overlay[L.rasterize_mask(inp.rivers.geometry, spec, all_touched=True) & on_land & (terrain != L.AGUA)] = L.RIO
    R.paint_lines(overlay, inp.lines, spec, on_land, R.OSM_OVERLAY_CODES)
    pseudo_codes = [k for k, (eid, _n) in labels.items() if k > 2000 and eid == target_eid]
    borders_index = np.where(np.isin(mun_index, pseudo_codes), 0, mun_index)
    overlay[L.internal_borders(borders_index)] = L.LIMITE_MUNICIPIO
    overlay[L.internal_borders(uf_index)] = L.LIMITE_ESTADO
    if inp.boundaries is not None and len(inp.boundaries):
        overlay[L.rasterize_mask(inp.boundaries.geometry, spec, all_touched=True) & on_land] = L.LIMITE_PAIS

    # --- objects ------------------------------------------------------------------------------
    objects = R.city_objects(
        terrain, None, tree_permille=R.LEVEL2_TREE_PERMILLE, casa_permille=R.LEVEL2_CASA_PERMILLE, predio=R.LEVEL2_PREDIO_PERMILLE
    )
    log.info(
        "level 2 grid %dx%d built in %.1fs: %d land cells, %d entities (%d municipalities), %d objects",
        size, size, time.perf_counter() - t0, int(land.sum()), len(entities), n_mun_entities, len(objects),
    )
    return Level1Grid(terrain, overlay, entity, entities, objects)


def manifest_for(spec: GridSpec, inp: Level2Inputs, grid: Level1Grid, tileset):
    year = inp.coverage.info.year if inp.coverage is not None and inp.coverage.info.year else None
    return spec.manifest(
        tileset, grid.entities, level=2, entity=entity_id("estado", inp.uf_code), name=inp.uf_name,
        period=str(year) if year else None,
    )


def output_dir(root: Path, uf_code: int) -> Path:
    return level_dir(root, 2, entity_id("estado", uf_code))


__all__ = ["Level2Inputs", "build_grid", "load_inputs", "load_uf", "make_spec", "manifest_for", "output_dir"]
