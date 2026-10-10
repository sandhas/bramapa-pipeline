"""Fase 4 unit tests (levels 2/3 helpers): no network, no cached downloads."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from bramapa_pipeline import level1 as L
from bramapa_pipeline import mapbiomas as M
from bramapa_pipeline import osm
from bramapa_pipeline import region as R
from bramapa_pipeline.level import (
    entity_dirname,
    entity_id,
    find_levels,
    is_pseudo_municipio,
    level_dir,
    municipio_uf,
    parse_entity_id,
)

FIXTURE = Path(__file__).parent / "fixtures" / "overpass-tiny.json"


# ----------------------------------------------------------------------------- entity ids / layout


def test_entity_ids_and_layout(tmp_path):
    assert entity_id("estado", 43) == "estado:43"
    assert entity_id("estado", "11") == "estado:11"
    assert entity_id("municipio", 4314902) == "municipio:4314902"
    assert entity_id("pais", "br") == "pais:BR"
    with pytest.raises(ValueError):
        entity_id("estado", 99)
    with pytest.raises(ValueError):
        entity_id("municipio", 123)
    assert parse_entity_id("municipio:4314902") == ("municipio", "4314902")
    assert municipio_uf(4314902) == 43
    assert entity_dirname("municipio:4314902") == "municipio_4314902"
    assert level_dir(tmp_path, 3, "municipio:4314902") == tmp_path / "3" / "municipio_4314902"
    assert level_dir(tmp_path, 2, "estado:43") == tmp_path / "2" / "estado_43"
    assert level_dir(tmp_path, 1, "pais:BR") == tmp_path / "1" / "pais_BR"
    with pytest.raises(ValueError):
        level_dir(tmp_path, 4, "pais:BR")
    # find_levels only lists directories with a level.toml, in level order.
    for d in ("3/municipio_4314902", "1/pais_BR", "2/estado_43", "2/estado_42"):
        (tmp_path / d).mkdir(parents=True)
    for d in ("1/pais_BR", "3/municipio_4314902", "2/estado_43"):
        (tmp_path / d / "level.toml").write_text("")
    assert [str(p.relative_to(tmp_path)) for p in find_levels(tmp_path)] == ["1/pais_BR", "2/estado_43", "3/municipio_4314902"]


def test_pseudo_municipios():
    assert is_pseudo_municipio(4300001) and is_pseudo_municipio(4300002)  # Lagoa Mirim / dos Patos
    assert not is_pseudo_municipio(4310009)  # sequence 1000 + check digit: a real one
    assert not is_pseudo_municipio(4314902) and not is_pseudo_municipio(4300034)


# ----------------------------------------------------------------------------- MapBiomas


def test_mapbiomas_class_table():
    lut = M.cover_lut()
    assert lut[0] == 0
    for cls in (3, 49, 9, 6):
        assert lut[cls] == L.FLORESTA
    assert lut[4] == L.SAVANA
    for cls in (5, 11, 32):
        assert lut[cls] == L.PANTANAL
    for cls in (12, 50, 25, 30, 13, 27):
        assert lut[cls] == L.CAMPO
    for cls in (15, 21):
        assert lut[cls] == L.PASTAGEM
    for cls in (18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48):
        assert lut[cls] == L.AGRICULTURA
    assert lut[23] == L.AREIA and lut[24] == L.URBANO and lut[29] == L.MONTANHA
    assert lut[33] == L.AGUA and lut[31] == L.AGUA
    assert lut[200] == L.CAMPO  # unknown class never leaves a hole
    assert set(M.MAPBIOMAS_COVER.values()) <= set(L.TERRAIN_CODES)


def test_classify_nodata_and_water_off_land():
    classes = np.array([[0, 0, 33], [33, 24, 3]], dtype=np.uint8)
    land = np.array([[False, True, False], [True, True, True]])
    t = M.classify(classes, land)
    assert t.tolist() == [[L.OCEANO, L.CAMPO, L.OCEANO], [L.AGUA, L.URBANO, L.FLORESTA]]


def test_decimation():
    assert M.decimation_for(30.0) == 1
    assert M.decimation_for(700.0) == 5
    assert M.decimation_for(1000.0) == 8


# ----------------------------------------------------------------------------- grid fitting


def _diamond(ce, cn, r):
    return Polygon([(ce + r, cn), (ce, cn + r), (ce - r, cn), (ce, cn - r)])


def test_rotated_extent_and_fit_cell_size():
    # A diamond of half-diagonal 100 km around (5e6, 7e6) is exactly a rotated grid footprint.
    g = _diamond(5_000_000, 7_000_000, 100_000)
    ce, cn, r = R.rotated_extent(g)
    assert (ce, cn) == pytest.approx((5_000_000, 7_000_000))
    assert r == pytest.approx(100_000)
    # Needs c >= r * sqrt2 / size (times the margin), rounded up to 10 m.
    c = R.fit_cell_meters(g, 1024, margin=1.0, min_cell=0)
    assert c == pytest.approx(math.ceil(100_000 * math.sqrt(2) / 1024 / 10) * 10)
    assert R.fit_cell_meters(g, 1024, margin=1.0) == R.MIN_CELL_METERS  # floor applies
    # Every vertex falls inside the fitted grid.
    spec = R.grid_for(g, size=1024, cell_meters=R.fit_cell_meters(g, 1024, min_cell=0))
    inv = ~spec.transform
    for e, n in g.exterior.coords:
        x, y = inv @ (e, n)
        assert -0.01 <= x <= 1024.01 and -0.01 <= y <= 1024.01
    # An axis-aligned box of side 2a has |dE|+|dN| up to 2a at the corners, so r = 2a.
    b = box(0, 0, 20_000, 20_000)
    assert R.rotated_extent(b)[2] == pytest.approx(20_000)


def test_fit_size_multiple_of_64_and_cap():
    g = _diamond(0, 0, 30_000)  # needs 30000*sqrt2/30 = 1414 cells -> 1472 at margin 1.0
    size, cell, capped = R.fit_size(g, 30.0, margin=1.0)
    assert size == 1472 and cell == 30.0 and not capped
    size, cell, capped = R.fit_size(g, 30.0, margin=1.0, cap=1024)
    assert size == 1024 and capped and cell >= 30_000 * math.sqrt(2) / 1024
    assert cell % 10 == 0


# ----------------------------------------------------------------------------- objects


def test_predio_density_monotonic():
    d = np.linspace(0, 20_000, 401)
    p = R.predio_permille(d)
    assert p[0] == 250 and p[-1] == 20
    assert np.all(np.diff(p.astype(int)) <= 0)
    assert R.predio_permille(np.array([5_000.0]))[0] == 80
    assert R.predio_permille(np.array([1_999.0, 2_000.0, 3_500.0]))[2] == pytest.approx(165, abs=1)


def test_city_objects_follow_covers():
    terrain = np.full((100, 100), L.URBANO, dtype=np.uint8)
    terrain[:, :20] = L.FLORESTA
    terrain[:, 20:30] = L.CAMPO
    dist = np.zeros((100, 100))  # everything "in the centre": ~25% predio
    objs = R.city_objects(terrain, dist, tree_permille=100, casa_permille=200, predio=0)
    by = {}
    for x, y, name, _f in objs:
        by.setdefault(name, []).append((x, y))
    for x, y in by["arvore"]:
        assert terrain[y, x] == L.FLORESTA
    for name in ("predio", "casa"):
        for x, y in by[name]:
            assert terrain[y, x] == L.URBANO
    n_urban = 70 * 100
    assert 0.2 < len(by["predio"]) / n_urban < 0.3
    assert 0.1 < len(by["casa"]) / n_urban < 0.2  # 20% of what predio left
    assert not set(by["predio"]) & set(by["casa"])


def test_mask_outline():
    m = np.zeros((5, 5), dtype=bool)
    m[1:4, 1:4] = True
    o = L.mask_outline(m)
    assert o[1:4, 1:4].sum() == 8 and not o[2, 2] and not o[0].any()


# ----------------------------------------------------------------------------- Overpass


def test_overpass_query_and_bbox():
    q = osm.build_query((-51.3, -30.3, -51.0, -29.9), (osm.highway_selector(("motorway", "trunk")), osm.RAIL_SELECTOR))
    assert q.startswith("[out:json][timeout:180];(")
    assert 'way[highway~"^(motorway|trunk)$"](-30.300000,-51.300000,-29.900000,-51.000000);' in q
    assert "way[railway=rail](-30.300000,-51.300000,-29.900000,-51.000000);" in q
    assert q.endswith(");out geom;")
    assert osm.build_query((0, 0, 1, 1), ("way[x=y]",), out="count").endswith("out count;")


def test_overpass_parse_and_classify():
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    lines = osm.parse_lines(doc)
    assert len(lines) == 4  # the node and the 1-vertex way are dropped
    covers = [osm.classify_way(tags) for tags, _c in lines]
    assert covers == ["rodovia", "ferrovia", "rio", None]
    assert osm.classify_way({"highway": "residential"}, osm.ROAD_CLASSES_LEVEL2) is None
    assert osm.classify_way({"highway": "primary"}, osm.ROAD_CLASSES_LEVEL2) == "rodovia"
    gdf = osm.lines_to_gdf(lines, osm.ROAD_CLASSES_LEVEL3)
    assert list(gdf["overlay"]) == ["rodovia", "ferrovia", "rio"]
    assert gdf.crs.to_epsg() == 4326
    assert list(gdf.geometry.iloc[0].coords)[0] == (-51.2, -30.0)


def test_overpass_error_detection():
    assert osm._looks_like_error("<html><body>The server is probably too busy</body></html>") == "server too busy / timed out"
    assert osm._looks_like_error("<!DOCTYPE HTML><html>500 Internal Server Error</html>") == "HTML error page"
    assert osm._looks_like_error('{"elements": []}') is None
