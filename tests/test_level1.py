"""Builder unit tests on tiny synthetic GeoDataFrames: no network, no cached downloads."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, box

from bramapa_pipeline import level1 as L
from bramapa_pipeline.chunk import read_chunk
from bramapa_pipeline.level import read_level_toml
from bramapa_pipeline.tileset import builtin_tileset

SIZE = 16
SPEC = L.GridSpec(size=SIZE, cell_meters=7000.0)


def _lonlat(e: float, n: float) -> tuple[float, float]:
    t = Transformer.from_crs(SPEC.crs, "EPSG:4326", always_xy=True)
    return t.transform(e, n)


def _gdf(geoms, **cols):
    return gpd.GeoDataFrame({**cols, "geometry": geoms}, crs=SPEC.crs)


def _cell_square(x: int, y: int, shrink: float = 0.3) -> Polygon:
    """A polygon safely inside cell (x, y) (a rotated square around the cell centre)."""
    e, n = SPEC.cell_center(x, y)
    r = SPEC.cell_meters * shrink
    return Polygon([(e + r, n), (e, n + r), (e - r, n), (e, n - r)])


def _cells_polygon(cells: list[tuple[int, int]]) -> Polygon:
    """Union of full cells (in grid space, via the transform) -> one polygon."""
    from shapely.ops import unary_union

    t = SPEC.transform
    polys = [Polygon([t @ (x, y), t @ (x + 1, y), t @ (x + 1, y + 1), t @ (x, y + 1)]) for x, y in cells]
    return unary_union(polys).buffer(1.0)


# ----------------------------------------------------------------------------- transform


def test_center_cell_hits_projected_centre():
    e, n = SPEC.cell_center(SIZE // 2, SIZE // 2)
    ce, cn = SPEC.center_projected
    assert math.isclose(e, ce, abs_tol=1e-3) and math.isclose(n, cn, abs_tol=1e-3)
    lon, lat = _lonlat(e, n)
    assert math.isclose(lon, -53.0, abs_tol=1e-6) and math.isclose(lat, -14.0, abs_tol=1e-6)


def test_axes_point_south_east_and_south_west():
    c = SPEC.cell_meters / math.sqrt(2)
    e00, n00 = SPEC.cell_center(0, 0)
    e11, n11 = SPEC.cell_center(1, 1)
    e10, n10 = SPEC.cell_center(1, 0)
    e01, n01 = SPEC.cell_center(0, 1)
    # (0,0) is north of (1,1), same easting.
    assert n00 > n11 and math.isclose(e00, e11, abs_tol=1e-6)
    assert math.isclose(n00 - n11, 2 * c, rel_tol=1e-9)
    # +x goes south-east, +y goes south-west, each step c/sqrt2 in both E and N.
    assert math.isclose(e10 - e00, c, rel_tol=1e-9) and math.isclose(n10 - n00, -c, rel_tol=1e-9)
    assert math.isclose(e01 - e00, -c, rel_tol=1e-9) and math.isclose(n01 - n00, -c, rel_tol=1e-9)
    # Cell (0,0) is the northernmost cell of the whole grid.
    ns = [SPEC.cell_center(x, y)[1] for x in range(SIZE) for y in range(SIZE)]
    assert max(ns) == pytest.approx(n00)


def test_affine_matches_formula():
    oe, on = SPEC.origin
    c = SPEC.cell_meters / math.sqrt(2)
    for x, y in [(0, 0), (3, 1), (7, 9)]:
        e, n = SPEC.transform @ (x, y)
        assert math.isclose(e, oe + (x - y) * c, rel_tol=1e-12)
        assert math.isclose(n, on - (x + y) * c, rel_tol=1e-12)


def test_rasterize_lands_in_the_right_cell():
    # Polygons inside cells (2, 5) and (9, 9) light exactly those cells, through the rotated affine.
    m = L.rasterize_values([_cell_square(2, 5), _cell_square(9, 9)], [1, 2], SPEC)
    assert m[5, 2] == 1 and m[9, 9] == 2
    assert np.count_nonzero(m) == 2


# ----------------------------------------------------------------------------- rules


def test_biome_mapping_and_normalisation():
    assert L.biome_code("Amazônia") == L.FLORESTA
    assert L.biome_code("Mata Atlântica") == L.FLORESTA
    assert L.biome_code("Cerrado") == L.SAVANA
    assert L.biome_code("Caatinga") == L.CAATINGA
    assert L.biome_code("Pampa") == L.CAMPO
    assert L.biome_code("Pantanal") == L.PANTANAL
    assert L.biome_code("AMAZONIA") == L.FLORESTA
    assert L.biome_code("???") == L.CAMPO


def test_beach_rule_one_cell():
    t = np.full((5, 5), L.OCEANO, dtype=np.uint8)
    t[1:4, 1:4] = L.FLORESTA
    t[2, 2] = L.AGUA  # a lake in the middle (not next to the ocean, and lakes never become sand)
    L.beach(t)
    assert t[2, 2] == L.AGUA
    assert t[1, 1] == L.AREIA and t[1, 2] == L.AREIA and t[3, 3] == L.AREIA
    big = np.full((7, 7), L.OCEANO, dtype=np.uint8)
    big[1:6, 1:6] = L.CAMPO
    L.beach(big)
    assert big[3, 3] == L.CAMPO  # interior untouched
    assert big[2, 2] == L.CAMPO  # diagonal-only contact is not "4-neighbourhood"
    assert big[1, 3] == L.AREIA


def test_internal_borders_one_sided_and_skip_zero():
    idx = np.zeros((4, 4), dtype=np.uint16)
    idx[:, :2] = 1
    idx[:, 2:] = 2
    idx[0, :] = 0  # ocean row: no border against 0
    b = L.internal_borders(idx)
    assert b[1:, 1].all() and not b[1:, 2].any()  # drawn on the W side only
    assert not b[0].any() and not b[:, 0].any() and not b[:, 3].any()


def test_country_entity_fallbacks():
    assert L.country_entity("BR", "BR", "BRA") == "pais:BR"
    assert L.country_entity("-99", "FR", "FRA") == "pais:FR"
    assert L.country_entity("-99", "-99", "ATF") == "pais:ATF"


def test_index_entities_first_appearance_and_merge():
    raw = np.array([[0, 7, 7], [3, 3, 9], [9, 9, 0]], dtype=np.uint16)
    labels = {7: ("pais:UY", "Uruguai"), 3: ("estado:43", "RS"), 9: ("pais:UY", "Uruguai")}
    ent, entities = L.index_entities(raw, labels)
    assert entities == [(1, "pais:UY", "Uruguai"), (2, "estado:43", "RS")]
    assert ent.tolist() == [[0, 1, 1], [2, 2, 1], [1, 1, 0]]


def test_fill_from_neighbours():
    idx = np.array([[1, 0, 0, 2], [1, 0, 0, 2]], dtype=np.uint16)
    holes = np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=bool)
    out = L.fill_from_neighbours(idx, holes)
    assert out.tolist() == [[1, 1, 0, 2], [1, 0, 2, 2]]


def test_tree_mask_is_sparse_and_deterministic():
    forest = np.ones((200, 200), dtype=bool)
    keep, flip = L.tree_mask(forest)
    frac = keep.mean()
    assert 0.02 < frac < 0.04
    keep2, _ = L.tree_mask(forest)
    assert np.array_equal(keep, keep2)
    assert 0.3 < flip[keep].mean() < 0.7


# ----------------------------------------------------------------------------- build_grid


def _inputs() -> L.Level1Inputs:
    # "Brazil" = cells x in 4..11, y in 4..11 (IBGE), split into two states at x < 8 / x >= 8.
    # Natural Earth's Brazil runs one column further west (x = 3) to exercise the sliver rule.
    # "Uruguai" = cells (4..7, 12..13). Everything else is ocean.
    ne_brazil = _cells_polygon([(x, y) for x in range(3, 12) for y in range(4, 12)])
    uruguay = _cells_polygon([(x, y) for x in range(4, 8) for y in range(12, 14)])
    countries = _gdf([ne_brazil, uruguay], entity_id=["pais:BR", "pais:UY"], entity_name=["Brasil", "Uruguai"])
    rs = _cells_polygon([(x, y) for x in range(4, 8) for y in range(4, 12)])
    sp = _cells_polygon([(x, y) for x in range(8, 12) for y in range(4, 12)])
    uf = _gdf([rs, sp], entity_id=["estado:43", "estado:35"], entity_name=["Rio Grande do Sul", "São Paulo"])
    # Biomes: north half (y < 8) Amazônia, south half Cerrado, one pantanal cell.
    amaz = _cells_polygon([(x, y) for x in range(4, 12) for y in range(4, 8)])
    cerr = _cells_polygon([(x, y) for x in range(4, 12) for y in range(8, 12)])
    biomes = _gdf([amaz, cerr], cover=[L.biome_code("Amazônia"), L.biome_code("Cerrado")])
    lakes = _gdf([_cell_square(9, 9)])
    # A river running along y = 6 from x = 5..10 and a road crossing it at x = 6; the state
    # border runs between x = 7 and 8 (drawn on x = 7); the country border between Brazil and
    # Uruguay is drawn on y = 11.
    ye = lambda x, y: SPEC.cell_center(x, y)  # noqa: E731
    rivers = _gdf([LineString([ye(5, 6), ye(10, 6)])])
    roads = _gdf([LineString([ye(6, 4), ye(6, 11)])])
    boundaries = _gdf([LineString([ye(4, 11), ye(7, 11)])])
    elev = np.zeros((SIZE, SIZE), dtype=np.float32)
    elev[10, 10] = 3000.0  # a mountain inside São Paulo (cerrado otherwise)
    elev[0, 0] = 5000.0  # in the ocean: must not become montanha
    return L.Level1Inputs(countries, uf, biomes, lakes, rivers, boundaries, roads, elev)


def test_build_grid_rules():
    g = L.build_grid(SPEC, _inputs())
    t, o, e = g.terrain, g.overlay, g.entity
    # Entities: row-major first appearance; RS (x=4..7) appears before SP in row 4; UY later.
    ids = [eid for _, eid, _ in g.entities]
    assert ids == ["estado:43", "estado:35", "pais:UY"]
    assert e[0, 0] == 0  # ocean
    assert e[5, 5] == 1 and e[5, 9] == 2 and e[12, 5] == 3
    assert e[5, 3] == 1  # NE-Brazil sliver absorbed by the neighbouring state, no pais:BR entity
    # Covers.
    assert t[0, 0] == L.OCEANO
    assert t[6, 6] == L.FLORESTA and t[9, 6] == L.SAVANA
    assert t[9, 9] == L.AGUA
    assert t[10, 10] == L.MONTANHA
    assert t[0, 0] == L.OCEANO  # elevation over the ocean is ignored
    assert t[12, 5] == L.CAMPO and t[13, 4] == L.AREIA  # Uruguay: campo, beach on the coast
    assert t[4, 5] == L.AREIA and t[5, 5] == L.FLORESTA  # beach is one cell, then biome
    # Overlay priority: limite_pais > limite_estado > rodovia > rio.
    assert o[6, 5] == L.RIO and o[5, 6] == L.RODOVIA
    assert o[6, 6] == L.RODOVIA  # road x river: road wins
    assert o[6, 7] == L.LIMITE_ESTADO  # river x state border: state border wins
    assert o[11, 6] == L.LIMITE_PAIS  # road x country border: country border wins
    assert o[11, 7] == L.LIMITE_PAIS  # state border x country border: country border wins
    assert o[8, 7] == L.LIMITE_ESTADO and o[8, 8] == 0  # one-sided
    assert (o[t == L.OCEANO] == 0).all()
    # Trees only on forest.
    for x, y, name, _flags in g.objects:
        assert name == "arvore" and t[y, x] == L.FLORESTA


def test_write_level_and_preview(tmp_path):
    g = L.build_grid(SPEC, _inputs())
    ts = builtin_tileset()
    manifest, n_chunks, total = L.write_level(g, SPEC, ts, tmp_path / "lvl")
    assert n_chunks == 1 and total > 0
    m = read_level_toml(tmp_path / "lvl" / "level.toml")
    assert m.level == 1 and m.entity == "pais:BR" and m.name == "Brasil e vizinhos"
    assert m.rotation_deg == 45.0 and m.crs == "EPSG:5880" and m.cell_meters == 7000.0
    assert (m.origin_x, m.origin_y) == SPEC.origin
    assert m.entities == g.entities
    ch = read_chunk(tmp_path / "lvl" / "chunks" / "0_0.bmc")
    assert ch.entity is not None and ch.entity.shape == (64, 64)
    assert np.array_equal(ch.entity[:SIZE, :SIZE], g.entity)
    assert (ch.entity[SIZE:, :] == 0).all()
    assert ch.terrain[0, 0] == ts.tile_for_cover("terrain", "oceano")
    assert ch.terrain[6, 6] == ts.tile_for_cover("terrain", "floresta")
    assert ch.overlay[11, 7] == ts.tile_for_cover("overlay", "limite_pais")
    assert all(obj.tile_id == ts.object_id("arvore") for obj in ch.objects)
    p = L.write_preview(g, tmp_path / "preview.png")
    assert p.stat().st_size > 0


def test_lonlat_bounds_enclose_grid():
    w, s, e, n = SPEC.lonlat_bounds(margin_deg=0.0)
    for x, y in [(0, 0), (SIZE - 1, 0), (0, SIZE - 1), (SIZE - 1, SIZE - 1)]:
        lon, lat = _lonlat(*SPEC.cell_center(x, y))
        assert w <= lon <= e and s <= lat <= n
