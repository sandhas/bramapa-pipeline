"""Real level 1 (Brazil and neighbours) from public geodata.

The grid is sampled in ``EPSG:5880`` (SIRGAS 2000 / Brazil Polyconic) with the axes rotated 45°
as ``chunk.md`` demands: grid ``+x`` points south-east on the ground, ``+y`` south-west, so north
is up on screen and cell ``(0, 0)`` is the northernmost cell. For cell ``(x, y)``::

    E = oe + (x - y) * c / sqrt2
    N = on - (x + y) * c / sqrt2

with ``c = cell_meters`` and ``(oe, on)`` the projected position of the *north vertex* of cell
``(0, 0)`` (GDAL's pixel-corner convention, ``col = x``, ``row = y``). That is the rotated affine
``Affine(c/sqrt2, -c/sqrt2, oe, -c/sqrt2, -c/sqrt2, on)``; GDAL rasterizes and warps straight into
it, so no un-rotated intermediate raster exists.

Pipeline: :func:`load_inputs` reads the cached sources into GeoDataFrames already in the grid
CRS; :func:`build_grid` turns them into full-grid cover/overlay/entity arrays (pure numpy +
``rasterize``, unit-testable on tiny synthetic frames); :func:`write_level` autotiles through a
:class:`~.tileset.Tileset`, chunks and writes ``level.toml``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from rasterio.transform import Affine

from .autotile import apply_autotile
from .chunk import OBJECT_FLAG_FLIP_H, Chunk, ChunkObject, chunk_filename, write_chunk
from .level import LevelManifest, write_level_toml
from .tileset import Tileset

log = logging.getLogger(__name__)

SQRT2 = math.sqrt(2.0)
GRID_CRS = "EPSG:5880"
ROTATION_DEG = 45.0

# Terrain cover codes used inside the builder (uint8), 1-based because `apply_autotile` reads
# code 0 as "empty cell" and every cell of level 1 has a cover (ocean included).
OCEANO, CAMPO, FLORESTA, SAVANA, CAATINGA, PANTANAL, AGUA, AREIA, MONTANHA = range(1, 10)
TERRAIN_CODES: dict[int, str] = {
    OCEANO: "oceano",
    CAMPO: "campo",
    FLORESTA: "floresta",
    SAVANA: "savana",
    CAATINGA: "caatinga",
    PANTANAL: "pantanal",
    AGUA: "agua",
    AREIA: "areia",
    MONTANHA: "montanha",
}

# Overlay codes, in *increasing* priority: when two overlays hit one cell the higher wins.
RIO, RODOVIA, LIMITE_ESTADO, LIMITE_PAIS = 1, 2, 3, 4
OVERLAY_CODES: dict[int, str] = {
    RIO: "rio",
    RODOVIA: "rodovia",
    LIMITE_ESTADO: "limite_estado",
    LIMITE_PAIS: "limite_pais",
}

# IBGE biome name -> terrain code. Lookup is accent/case-insensitive (see :func:`biome_code`).
BIOME_COVER: dict[str, int] = {
    "amazonia": FLORESTA,
    "mata atlantica": FLORESTA,
    "cerrado": SAVANA,
    "caatinga": CAATINGA,
    "pampa": CAMPO,
    "pantanal": PANTANAL,
}

MOUNTAIN_METERS = 2000.0
RIVER_MAX_SCALERANK = 6
LAKE_MAX_SCALERANK = 3
ROAD_TYPES = ("Major Highway", "Beltway")
TREE_PERMILLE = 30  # ~3% of forest cells get an `arvore` object

PREVIEW_COLOURS: dict[str, tuple[int, int, int]] = {
    "oceano": (28, 60, 110),
    "agua": (70, 130, 200),
    "campo": (170, 200, 110),
    "floresta": (30, 110, 50),
    "savana": (200, 170, 80),
    "caatinga": (210, 190, 140),
    "pantanal": (90, 160, 140),
    "areia": (240, 225, 170),
    "montanha": (130, 120, 115),
}


def _strip_accents(s: str) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def biome_code(name: str) -> int:
    """IBGE ``Bioma`` value -> terrain code; unknown names fall back to ``campo``."""
    key = " ".join(_strip_accents(str(name)).lower().split())
    code = BIOME_COVER.get(key)
    if code is None:
        log.warning("unknown biome %r, painting campo", name)
        return CAMPO
    return code


# ----------------------------------------------------------------------------- grid geometry


@dataclass(frozen=True)
class GridSpec:
    """Size, cell size and placement of the rotated grid in the projected CRS."""

    size: int = 1024
    cell_meters: float = 7000.0
    center_lon: float = -53.0
    center_lat: float = -14.0
    crs: str = GRID_CRS

    @property
    def center_projected(self) -> tuple[float, float]:
        from pyproj import Transformer

        t = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        e, n = t.transform(self.center_lon, self.center_lat)
        return float(e), float(n)

    @property
    def origin(self) -> tuple[float, float]:
        """Projected ``(E, N)`` of the north vertex of cell ``(0, 0)``.

        Chosen so that the *centre* of cell ``(size/2, size/2)`` is the projected centre: that
        centre sits at grid coords ``(size/2 + 0.5, size/2 + 0.5)``.
        """
        ce, cn = self.center_projected
        k = self.size / 2 + 0.5
        oe = ce  # (x - y) == 0 on the diagonal
        on = cn + 2 * k * self.cell_meters / SQRT2
        return oe, on

    @property
    def transform(self) -> Affine:
        oe, on = self.origin
        s = self.cell_meters / SQRT2
        return Affine(s, -s, oe, -s, -s, on)

    def cell_center(self, x: float, y: float) -> tuple[float, float]:
        """Projected ``(E, N)`` of the centre of cell ``(x, y)``."""
        e, n = self.transform @ (x + 0.5, y + 0.5)
        return float(e), float(n)

    def projected_bounds(self) -> tuple[float, float, float, float]:
        """``(min_e, min_n, max_e, max_n)`` of the rotated grid's four corners."""
        t = self.transform
        corners = [t @ (0, 0), t @ (self.size, 0), t @ (0, self.size), t @ (self.size, self.size)]
        es = [c[0] for c in corners]
        ns = [c[1] for c in corners]
        return min(es), min(ns), max(es), max(ns)

    def lonlat_bounds(self, margin_deg: float = 1.0) -> tuple[float, float, float, float]:
        """Geographic bbox enclosing the grid (for pre-filtering vector reads)."""
        from pyproj import Transformer

        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        tr = self.transform
        pts = []
        n = 16
        for i in range(n + 1):
            f = i / n * self.size
            pts += [tr @ (f, 0), tr @ (f, self.size), tr @ (0, f), tr @ (self.size, f)]
        lons, lats = t.transform([p[0] for p in pts], [p[1] for p in pts])
        lons = [v for v in lons if math.isfinite(v)]
        lats = [v for v in lats if math.isfinite(v)]
        return (
            max(-180.0, min(lons) - margin_deg),
            max(-90.0, min(lats) - margin_deg),
            min(180.0, max(lons) + margin_deg),
            min(90.0, max(lats) + margin_deg),
        )

    def manifest(self, tileset: Tileset, entities: list[tuple[int, str, str]]) -> LevelManifest:
        oe, on = self.origin
        return LevelManifest(
            level=1,
            entity="pais:BR",
            name="Brasil e vizinhos",
            width=self.size,
            height=self.size,
            cell_meters=float(self.cell_meters),
            crs=self.crs,
            origin_x=oe,
            origin_y=on,
            rotation_deg=ROTATION_DEG,
            tileset_id=tileset.id,
            tileset_version=tileset.version,
            entities=entities,
        )


# ----------------------------------------------------------------------------- rasterizing


def rasterize_values(
    geoms: Iterable, values: Iterable[int], spec: GridSpec, *, dtype=np.uint16, all_touched: bool = False
) -> np.ndarray:
    """Burn ``values`` for each geometry into a ``(size, size)`` array; 0 = nothing. Later wins."""
    from rasterio.features import rasterize

    shapes = [(g, int(v)) for g, v in zip(geoms, values) if g is not None and not g.is_empty]
    out = np.zeros((spec.size, spec.size), dtype=dtype)
    if not shapes:
        return out
    return rasterize(shapes, out=out, transform=spec.transform, all_touched=all_touched, fill=0)


def rasterize_mask(geoms: Iterable, spec: GridSpec, *, all_touched: bool = False) -> np.ndarray:
    geoms = list(geoms)
    return rasterize_values(geoms, [1] * len(geoms), spec, dtype=np.uint8, all_touched=all_touched).astype(bool)


def warp_elevation(src_array: np.ndarray, src_transform: Affine, src_crs, spec: GridSpec) -> np.ndarray:
    """Reproject an elevation raster onto the rotated grid (area average), NaN where no data."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    dst = np.full((spec.size, spec.size), np.nan, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(src_array, dtype=np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=spec.transform,
        dst_crs=spec.crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return dst


# ----------------------------------------------------------------------------- inputs


@dataclass
class Level1Inputs:
    """Vector layers already projected to the grid CRS (``GeoDataFrame`` or ``None``).

    * ``countries``: columns ``entity_id``, ``entity_name`` (``pais:XX``), one row per country.
    * ``uf``: columns ``entity_id``, ``entity_name`` (``estado:NN``).
    * ``biomes``: column ``cover`` (terrain code from :func:`biome_code`).
    * ``lakes``, ``rivers``, ``boundaries``, ``roads``: geometry only (already filtered).
    * ``elevation``: ``(size, size)`` float array in metres aligned with the grid, or ``None``.
    """

    countries: object | None = None
    uf: object | None = None
    biomes: object | None = None
    lakes: object | None = None
    rivers: object | None = None
    boundaries: object | None = None
    roads: object | None = None
    elevation: np.ndarray | None = None


def _col(df, *names: str):
    """Case-insensitive column lookup; returns the Series or ``None``."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return df[lower[n.lower()]]
    return None


def country_entity(row_iso: str | None, row_iso_eh: str | None, row_a3: str | None) -> str:
    """``pais:<ISO alpha-2>``; Natural Earth uses ``-99`` for missing, so fall back in order."""
    for v in (row_iso, row_iso_eh):
        if v and v != "-99" and len(v) == 2:
            return f"pais:{v.upper()}"
    return f"pais:{(row_a3 or 'XX').upper()}"


def prepare_countries(gdf):
    iso = _col(gdf, "ISO_A2")
    iso_eh = _col(gdf, "ISO_A2_EH")
    a3 = _col(gdf, "ADM0_A3")
    name_pt = _col(gdf, "NAME_PT")
    name = _col(gdf, "NAME")
    n = len(gdf)
    ids = [
        country_entity(
            None if iso is None else iso.iloc[i],
            None if iso_eh is None else iso_eh.iloc[i],
            None if a3 is None else a3.iloc[i],
        )
        for i in range(n)
    ]
    names = []
    for i in range(n):
        v = name_pt.iloc[i] if name_pt is not None else None
        if not v or v == "-99":
            v = name.iloc[i] if name is not None else ids[i]
        names.append(str(v))
    out = gdf[["geometry"]].copy()
    out["entity_id"] = ids
    out["entity_name"] = names
    return out


def prepare_uf(gdf):
    cd = _col(gdf, "CD_UF")
    nm = _col(gdf, "NM_UF")
    out = gdf[["geometry"]].copy()
    out["entity_id"] = [f"estado:{int(v):02d}" for v in cd]
    out["entity_name"] = [str(v) for v in nm]
    return out


def prepare_biomes(gdf):
    col = _col(gdf, "Bioma", "bioma", "NOME", "nome")
    out = gdf[["geometry"]].copy()
    out["cover"] = [biome_code(v) for v in col] if col is not None else CAMPO
    return out


def _read_vector(zip_path: Path, columns: tuple[str, ...], bbox, layer: str | None = None):
    """Read the (single) shapefile layer in a zip via GDAL's vsizip, filtered by lon/lat bbox."""
    import geopandas as gpd
    import pyogrio

    vsi = f"/vsizip/{Path(zip_path).resolve()}"
    if layer is None:
        layers = pyogrio.list_layers(vsi)
        polyish = [row[0] for row in layers if row[1] is not None and row[1] != "Unknown"] or [row[0] for row in layers]
        if not polyish:
            raise RuntimeError(f"{zip_path}: no readable layer")
        layer = polyish[0]
    info = pyogrio.read_info(vsi, layer=layer)
    have = {f.lower(): f for f in info["fields"]}
    cols = [have[c.lower()] for c in columns if c.lower() in have]
    missing = [c for c in columns if c.lower() not in have]
    if missing:
        log.warning("%s: missing columns %s (have %s)", zip_path.name, missing, list(info["fields"]))
    gdf = gpd.read_file(vsi, layer=layer, columns=cols, bbox=bbox, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def _project(gdf, spec: GridSpec, simplify_deg: float | None):
    if gdf is None or len(gdf) == 0:
        return gdf
    if simplify_deg:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(simplify_deg, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf.to_crs(spec.crs)


def load_inputs(spec: GridSpec, cache_dir: Path, *, use_elevation: bool = True) -> Level1Inputs:
    """Read the cached sources (see :mod:`.sources`), filter and project them to the grid CRS."""
    from . import sources

    bbox = spec.lonlat_bounds()
    # Vertices closer than ~a tenth of a cell are noise at this scale; simplifying in degrees
    # before projecting keeps the reprojection of the huge IBGE layers quick.
    simp = spec.cell_meters / 10.0 / 111_000.0
    inp = Level1Inputs()

    def read(key: str, prep=None, simplify=simp):
        p = sources.path_for(key, cache_dir)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run `bramapa-pipeline fetch` first")
        t0 = time.perf_counter()
        gdf = _read_vector(p, sources.SOURCES[key].fields, bbox, sources.SOURCES[key].layer)
        if prep is not None:
            gdf = prep(gdf)
        gdf = _project(gdf, spec, simplify)
        log.info("%s: %d features in %.1fs", key, len(gdf), time.perf_counter() - t0)
        return gdf

    inp.countries = read("ne_10m_admin_0_countries", prepare_countries)
    inp.uf = read("ibge_uf_2022", prepare_uf)
    inp.biomes = read("ibge_biomas", prepare_biomes)

    lakes = read("ne_10m_lakes", simplify=None)
    sr = _col(lakes, "scalerank")
    if sr is not None:
        big_area = lakes.geometry.area >= 2 * spec.cell_meters**2
        lakes = lakes[(sr.astype(float) <= LAKE_MAX_SCALERANK) | big_area]
    inp.lakes = lakes

    rivers = read("ne_10m_rivers_lake_centerlines", simplify=None)
    sr = _col(rivers, "scalerank")
    if sr is not None:
        rivers = rivers[sr.astype(float) <= RIVER_MAX_SCALERANK]
    inp.rivers = rivers

    inp.boundaries = read("ne_10m_admin_0_boundary_lines_land", simplify=None)

    roads = read("ne_10m_roads", simplify=None)
    rt = _col(roads, "type")
    if rt is not None:
        roads = roads[rt.isin(ROAD_TYPES)]
    else:
        log.warning("ne_10m_roads has no `type` column; skipping roads")
        roads = roads.iloc[0:0]
    inp.roads = roads

    if use_elevation:
        p = sources.path_for("etopo2022", cache_dir)
        if p.exists():
            try:
                inp.elevation = load_elevation(p, spec)
            except Exception as e:  # noqa: BLE001 - optional input, never block the build
                log.warning("elevation unusable (%s: %s); no montanha", p, e)
        else:
            log.warning("no elevation raster at %s; no montanha", p)
    return inp


def load_elevation(path: Path, spec: GridSpec) -> np.ndarray:
    import rasterio
    from rasterio.transform import from_bounds

    from .sources import ETOPO_BBOX

    t0 = time.perf_counter()
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        tr, crs = ds.transform, ds.crs
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        if tr.is_identity or crs is None:
            # ArcGIS exports are sometimes plain TIFFs: fall back to the bbox we asked for.
            tr = from_bounds(*ETOPO_BBOX, ds.width, ds.height)
            crs = "EPSG:4326"
    arr[~np.isfinite(arr)] = np.nan
    elev = warp_elevation(arr, tr, crs, spec)
    log.info("elevation: %dx%d px warped in %.1fs", arr.shape[1], arr.shape[0], time.perf_counter() - t0)
    return elev


# ----------------------------------------------------------------------------- building


@dataclass
class Level1Grid:
    """Full-grid result before autotiling. Arrays are ``(size, size)`` indexed ``[y, x]``."""

    terrain: np.ndarray  # uint8 codes, see TERRAIN_CODES
    overlay: np.ndarray  # uint8 codes, see OVERLAY_CODES
    entity: np.ndarray  # uint16 index into `entities`
    entities: list[tuple[int, str, str]] = field(default_factory=list)
    objects: list[tuple[int, int, str, int]] = field(default_factory=list)  # (x, y, name, flags)

    def cover_names(self) -> np.ndarray:
        lut = np.array([TERRAIN_CODES.get(i, "") for i in range(max(TERRAIN_CODES) + 1)])
        return lut[self.terrain]


def _neighbour_any(mask: np.ndarray) -> np.ndarray:
    """True where any 4-neighbour of the cell is True (grid border counts as False)."""
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def beach(terrain: np.ndarray) -> np.ndarray:
    """Apply the one-cell beach rule in place: land next to ``oceano`` becomes ``areia``."""
    ocean = terrain == OCEANO
    land = ~ocean & (terrain != AGUA)
    terrain[land & _neighbour_any(ocean)] = AREIA
    return terrain


def fill_from_neighbours(index: np.ndarray, holes: np.ndarray, passes: int = 3) -> np.ndarray:
    """Give ``holes`` cells (index 0) the index of a 4-neighbour, repeating up to ``passes`` times.

    Used for slivers where one dataset's outline runs past another's; cells with no indexed
    neighbour within ``passes`` cells (isolated islands) stay 0.
    """
    out = index.copy()
    for _ in range(passes):
        todo = holes & (out == 0)
        if not todo.any():
            break
        cand = np.zeros_like(out)
        for src_sl, dst_sl in (
            ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),  # from E
            ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),  # from W
            ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),  # from S
            ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),  # from N
        ):
            c = cand[dst_sl]
            v = out[src_sl]
            np.copyto(c, v, where=(c == 0) & (v != 0))
        out = np.where(todo & (cand != 0), cand, out)
    return out


def internal_borders(index: np.ndarray) -> np.ndarray:
    """One-cell-wide lines between adjacent cells with different *non-zero* indices.

    Marks the cell whose E or S neighbour differs, so each border is drawn once (one side).
    """
    out = np.zeros(index.shape, dtype=bool)
    a, b = index[:, :-1], index[:, 1:]
    out[:, :-1] |= (a != b) & (a != 0) & (b != 0)
    a, b = index[:-1, :], index[1:, :]
    out[:-1, :] |= (a != b) & (a != 0) & (b != 0)
    return out


def tree_mask(forest: np.ndarray, permille: int = TREE_PERMILLE) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic sparse selection of forest cells: ``(keep mask, flip mask)``."""
    h, w = forest.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.uint32)
    hsh = (xx * np.uint32(73856093)) ^ (yy * np.uint32(19349663))
    hsh = (hsh ^ (hsh >> np.uint32(13))) * np.uint32(0x5BD1E995)
    hsh ^= hsh >> np.uint32(15)
    keep = forest & ((hsh % np.uint32(1000)) < permille)
    flip = ((hsh >> np.uint32(20)) & np.uint32(1)) == 1
    return keep, flip


def index_entities(raw: np.ndarray, labels: dict[int, tuple[str, str]]) -> tuple[np.ndarray, list[tuple[int, str, str]]]:
    """Renumber a provisional code raster to 1..k in order of first appearance (row-major).

    ``labels[code] = (entity_id, name)``; code 0 stays 0. Codes sharing an entity id merge.
    """
    flat = raw.ravel()
    codes, first = np.unique(flat, return_index=True)
    order = np.argsort(first, kind="stable")
    remap = np.zeros(int(codes.max()) + 1 if codes.size else 1, dtype=np.uint16)
    entities: list[tuple[int, str, str]] = []
    by_id: dict[str, int] = {}
    for code in codes[order]:
        code = int(code)
        if code == 0:
            continue
        eid, name = labels[code]
        idx = by_id.get(eid)
        if idx is None:
            idx = len(entities) + 1
            if idx > 0xFFFF:
                raise ValueError("more than 65535 entities")
            entities.append((idx, eid, name))
            by_id[eid] = idx
        remap[code] = idx
    return remap[raw], entities


def build_grid(spec: GridSpec, inp: Level1Inputs) -> Level1Grid:
    """Turn projected inputs into cover/overlay/entity grids. No I/O, all vectorised."""
    size = spec.size
    t0 = time.perf_counter()

    # --- entities + land --------------------------------------------------------------------
    labels: dict[int, tuple[str, str]] = {}
    raw = np.zeros((size, size), dtype=np.uint16)
    land = np.zeros((size, size), dtype=bool)
    if inp.countries is not None and len(inp.countries):
        c = inp.countries
        codes = np.arange(1, len(c) + 1)
        for k, (eid, name) in enumerate(zip(c["entity_id"], c["entity_name"]), start=1):
            labels[k] = (str(eid), str(name))
        raw = rasterize_values(c.geometry, codes, spec)
        land |= raw != 0
    uf_index = np.zeros((size, size), dtype=np.uint16)
    if inp.uf is not None and len(inp.uf):
        u = inp.uf
        base = 1000
        codes = np.arange(base + 1, base + len(u) + 1)
        for k, (eid, name) in zip(codes, zip(u["entity_id"], u["entity_name"])):
            labels[int(k)] = (str(eid), str(name))
        uf_center = rasterize_values(u.geometry, codes, spec)
        uf_touch = rasterize_values(u.geometry, codes, spec, all_touched=True)
        # Brazil per IBGE wins over Natural Earth. NE-Brazil cells the IBGE polygons miss (the
        # two outlines disagree by up to a few km along the border) take a touching state, then
        # a neighbouring cell's state; only isolated rocks stay as a `pais:BR` fallback entity.
        br_codes = [k for k, (eid, _n) in labels.items() if eid == "pais:BR" and k < base]
        ne_brazil = np.isin(raw, br_codes) if br_codes else np.zeros_like(land)
        uf_index = np.where(uf_center != 0, uf_center, np.where(ne_brazil, uf_touch, 0)).astype(np.uint16)
        uf_index = fill_from_neighbours(uf_index, ne_brazil & (uf_index == 0))
        raw = np.where(uf_index != 0, uf_index, raw).astype(np.uint16)
        land |= uf_index != 0
    entity, entities = index_entities(raw, labels)
    brazil = uf_index != 0

    # --- terrain ------------------------------------------------------------------------------
    terrain = np.full((size, size), OCEANO, dtype=np.uint8)
    terrain[land] = CAMPO
    if inp.biomes is not None and len(inp.biomes):
        b = inp.biomes
        # Codes are >= 1, so 0 keeps meaning "no biome here"; cells the centre sampling misses
        # (biome edges vs. the IBGE state outline) take any biome polygon touching them.
        cov = np.asarray(b["cover"], dtype=int)
        bio = rasterize_values(b.geometry, cov, spec, dtype=np.uint8)
        bio_touch = rasterize_values(b.geometry, cov, spec, dtype=np.uint8, all_touched=True)
        bio = np.where(bio != 0, bio, bio_touch)
        sel = brazil & (bio != 0)
        terrain[sel] = bio[sel]
    if inp.elevation is not None:
        high = land & (np.nan_to_num(inp.elevation, nan=-1.0) > MOUNTAIN_METERS)
        terrain[high] = MONTANHA
    if inp.lakes is not None and len(inp.lakes):
        lake = rasterize_mask(inp.lakes.geometry, spec)
        terrain[lake] = AGUA
    beach(terrain)

    # --- overlay (ascending priority, later overwrites) ----------------------------------------
    overlay = np.zeros((size, size), dtype=np.uint8)
    on_land = terrain != OCEANO
    if inp.rivers is not None and len(inp.rivers):
        overlay[rasterize_mask(inp.rivers.geometry, spec, all_touched=True) & on_land] = RIO
    if inp.roads is not None and len(inp.roads):
        overlay[rasterize_mask(inp.roads.geometry, spec, all_touched=True) & on_land] = RODOVIA
    overlay[internal_borders(uf_index)] = LIMITE_ESTADO
    if inp.boundaries is not None and len(inp.boundaries):
        overlay[rasterize_mask(inp.boundaries.geometry, spec, all_touched=True) & on_land] = LIMITE_PAIS

    # --- objects ------------------------------------------------------------------------------
    keep, flip = tree_mask(terrain == FLORESTA)
    ys, xs = np.nonzero(keep)
    objects = [
        (int(x), int(y), "arvore", OBJECT_FLAG_FLIP_H if f else 0) for x, y, f in zip(xs, ys, flip[ys, xs])
    ]

    log.info(
        "grid %dx%d built in %.1fs: %d land cells, %d entities, %d trees",
        size, size, time.perf_counter() - t0, int(land.sum()), len(entities), len(objects),
    )
    return Level1Grid(terrain, overlay, entity, entities, objects)


# ----------------------------------------------------------------------------- output


def to_chunks(grid: Level1Grid, manifest: LevelManifest, tileset: Tileset) -> list[Chunk]:
    """Autotile the full grid through ``tileset`` and slice into 64×64 chunks (zero-padded)."""
    cs = manifest.chunk_size
    H, W = manifest.chunks_y * cs, manifest.chunks_x * cs
    h, w = grid.terrain.shape
    terrain = np.zeros((H, W), dtype=np.uint16)
    overlay = np.zeros((H, W), dtype=np.uint16)
    entity = np.zeros((H, W), dtype=np.uint16)
    terrain[:h, :w] = apply_autotile(grid.terrain, tileset, "terrain", codes=TERRAIN_CODES)
    overlay[:h, :w] = apply_autotile(grid.overlay, tileset, "overlay", codes=OVERLAY_CODES)
    entity[:h, :w] = grid.entity

    per_chunk: dict[tuple[int, int], list[ChunkObject]] = {}
    for x, y, name, flags in grid.objects:
        per_chunk.setdefault((x // cs, y // cs), []).append(ChunkObject(x % cs, y % cs, tileset.object_id(name), flags))

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
                entity=entity[sl].copy(),
                objects=per_chunk.get((cx, cy), []),
            )
        )
    return chunks


def write_level(grid: Level1Grid, spec: GridSpec, tileset: Tileset, out: Path) -> tuple[LevelManifest, int, int]:
    """Write ``level.toml`` + ``chunks/``; returns ``(manifest, n_chunks, total_bytes)``."""
    manifest = spec.manifest(tileset, grid.entities)
    chunks = to_chunks(grid, manifest, tileset)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_level_toml(out / "level.toml", manifest)
    total = 0
    for ch in chunks:
        total += write_chunk(out / "chunks" / chunk_filename(ch.cx, ch.cy), ch).stat().st_size
    return manifest, len(chunks), total


def write_preview(grid: Level1Grid, path: Path) -> Path:
    """One pixel per cell RGB PNG of the cover grid in *grid space* (so Brazil looks rotated 45°).

    Overlays are drawn on top: borders white, rivers light blue, roads dark grey.
    """
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    lut = np.zeros((max(TERRAIN_CODES) + 1, 3), dtype=np.uint8)
    for code, name in TERRAIN_CODES.items():
        lut[code] = PREVIEW_COLOURS[name]
    rgb = lut[grid.terrain]
    rgb[grid.overlay == RIO] = (120, 190, 255)
    rgb[grid.overlay == RODOVIA] = (60, 60, 60)
    rgb[grid.overlay == LIMITE_ESTADO] = (255, 255, 255)
    rgb[grid.overlay == LIMITE_PAIS] = (255, 80, 80)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = rgb.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", driver="PNG", width=w, height=h, count=3, dtype="uint8") as dst:
            dst.write(np.moveaxis(rgb, 2, 0))
    return path


__all__ = [
    "BIOME_COVER",
    "GridSpec",
    "Level1Grid",
    "Level1Inputs",
    "OVERLAY_CODES",
    "TERRAIN_CODES",
    "beach",
    "biome_code",
    "build_grid",
    "country_entity",
    "fill_from_neighbours",
    "index_entities",
    "internal_borders",
    "load_inputs",
    "prepare_biomes",
    "prepare_countries",
    "prepare_uf",
    "rasterize_mask",
    "rasterize_values",
    "to_chunks",
    "tree_mask",
    "warp_elevation",
    "write_level",
    "write_preview",
]
